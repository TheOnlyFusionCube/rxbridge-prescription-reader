"""LoRA fine-tune of LFM2.5-VL-3B to transcribe handwriting in 10 scripts.

benchmark.py remains the verifier, so this trainer never reads a test.jsonl:
only train.jsonl and val.jsonl are loaded, and the metric used for the inline
check is the same CER function benchmark.py scores with. A number printed by
this script therefore means the same thing as one printed by the benchmark.

The model is a hybrid: 22 short-conv layers and 8 full-attention layers, so LoRA
is applied to the attention projections, the conv in/out projections and the
SwiGLU feed-forward, and never to lm_head (262M params, in the output space).

Run a short smoke first; only commit to a long run once the loss curve and the
inline CER both move in the right direction:

    python3 train_lora.py --data-root htr --out checkpoints/lora --steps 200 \\
        --per-lang 200 --rank 16 --eval-samples 24
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import unicodedata
from pathlib import Path

import torch
from PIL import Image

PROMPT = (
    "Read the handwriting in this image and transcribe it exactly as written, "
    "preserving the original language and spelling. Reply with only the "
    "transcribed text, nothing else."
)

LANGS = ["en", "fr", "th", "ar", "de", "fa", "ru", "vi", "hi", "ur"]

DEFAULT_MODEL = "LiquidAI/LFM2.5-VL-3B"

LORA_TARGETS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.out_proj",
    "conv.in_proj",
    "conv.out_proj",
    "feed_forward.w1",
    "feed_forward.w2",
    "feed_forward.w3",
]

PASS_BAR = 0.25


def normalise(text: str) -> str:
    if not text:
        return ""
    return " ".join(unicodedata.normalize("NFKC", str(text)).split())


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    r, h = normalise(reference), normalise(hypothesis)
    if not r:
        return 0.0 if not h else 1.0
    return edit_distance(r, h) / len(r)


# ------------------------------------------------------------------ corpus


def load_rows(root: Path, per_lang: int, seed: int = 0) -> list[dict]:
    """Load train+val rows, capped per language. test.jsonl is never opened.

    The cap is what keeps the corpus balanced: without it the six languages
    with ~3200 rows supply 19200 of 22773 samples and en (60), ar (366) and
    vi (944) are effectively drowned out, which defeats the point of a
    multilingual run.
    """
    rows: list[dict] = []
    for lang in LANGS:
        found: list[dict] = []
        for shard in ("train", "val"):
            p = root / lang / f"{shard}.jsonl"
            if not p.exists():
                continue
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    row["_lang"] = lang
                    found.append(row)
        random.Random(seed).shuffle(found)
        found = found[:per_lang]
        rows.extend(found)
        print(f"  {lang}: {len(found)} rows", flush=True)
    print(f"  total: {len(rows)} rows", flush=True)
    return rows


# ------------------------------------------------------------------ batches


def collate(processor, row: dict) -> dict | None:
    """Tokenise one sample, masking everything but the transcript."""
    try:
        image = Image.open(row["image"]).convert("RGB")
    except Exception:
        return None
    answer = normalise(row["text"])
    if not answer:
        return None

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }]
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )

    prompt_enc = processor(text=prompt, images=[image], return_tensors="pt")
    full_enc = processor(text=prompt + answer, images=[image], return_tensors="pt")

    # The prompt is an exact prefix of the full sequence, so the answer tokens
    # are the tail after prompt_ids.shape[1]. Verified on a live sample: 114
    # prompt tokens, 120 with the answer appended.
    n_prompt = prompt_enc["input_ids"].shape[1]
    labels = torch.full_like(full_enc["input_ids"], -100)
    if full_enc["input_ids"].shape[1] <= n_prompt:
        return None
    labels[:, n_prompt:] = full_enc["input_ids"][:, n_prompt:]
    if labels.max().item() < 0:
        return None

    # Pass every processor key through, not just the four I guessed at. The
    # processor also emits pixel_attention_mask and spatial_shapes, and dropping
    # them silently mis-shapes the vision forward pass.
    kept = {
        key: value
        for key, value in full_enc.items()
        if key not in ("input_ids", "attention_mask")
    }
    kept["input_ids"] = full_enc["input_ids"]
    kept["attention_mask"] = full_enc["attention_mask"]
    kept["labels"] = labels
    return kept


def to_device(batch: dict, device: str) -> dict:
    out = {}
    for key, value in batch.items():
        if value is None:
            continue
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def predict(model, processor, image_path: str) -> str:
    """Same call sequence as benchmark.py, so train-time scoring is comparable."""
    image = Image.open(image_path).convert("RGB")
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPT},
        ],
    }]
    prompt = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    enc = processor(text=prompt, images=[image], return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=96, do_sample=False)
    return processor.batch_decode(
        out[:, enc["input_ids"].shape[-1]:], skip_special_tokens=True
    )[0]


# ------------------------------------------------------------------ main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--data-root", default="htr")
    ap.add_argument("--out", default="checkpoints/lora")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-lang", type=int, default=1500)
    ap.add_argument("--log", type=int, default=10)
    ap.add_argument("--eval-samples", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor

    random.seed(args.seed)

    print("loading model", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, trust_remote_code=True
    )
    print("model loaded", flush=True)

    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=LORA_TARGETS,
    )
    model = get_peft_model(model, lora_cfg)
    model.to(args.device)
    # Checkpointing is what makes this fit in 16 GB: without it the activations
    # for 30 layers of a 3B model plus the vision stream OOM before step one.
    # use_cache is incompatible with it, and the default reentrant mode drops
    # LoRA grads because only lora_* params require grad while the wrapped
    # layer inputs do not -- training would run with a flat loss.
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    try:
        model.print_trainable_parameters()
        model.enable_input_require_grads()
    except AttributeError:
        pass
    model.train()

    rows = load_rows(Path(args.data_root), args.per_lang, args.seed)
    if not rows:
        print("no corpus rows found", file=sys.stderr)
        return 1

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0)
    import math

    total = args.steps * args.accum

    def lr_lambda(step: int) -> float:
        step += 1  # LambdaLR calls this after the step counter increments
        if step <= args.warmup:
            return step / args.warmup
        progress = (step - args.warmup) / max(1, total - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    print(f"training: {args.steps} optimiser steps x {args.accum} accumulation "
          f"= {total} forward/backward passes", flush=True)
    t0 = time.time()
    losses: list[float] = []
    accum_left = args.accum
    micro = 0

    try:
        while len(losses) < args.steps * args.accum:
            row = rows[micro % len(rows)]
            micro += 1
            batch = collate(processor, row)
            if batch is None:
                continue
            batch = to_device(batch, args.device)
            out = model(**batch)
            loss = out.loss
            if not torch.isfinite(loss):
                print("non-finite loss, skipping", file=sys.stderr)
                opt.zero_grad()
                accum_left = args.accum
                continue
            (loss / args.accum).backward()
            losses.append(float(loss.detach().cpu()))
            accum_left -= 1
            if accum_left:
                continue

            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            accum_left = args.accum

            n = len(losses)
            if n % args.log == 0:
                el = time.time() - t0
                rate = el / n
                eta = (args.steps * args.accum - n) * rate / 60
                recent = sum(losses[-args.log:]) / min(args.log, len(losses))
                print(
                    f"  pass {n:6d}/{total}  loss {recent:.4f}  "
                    f"{rate:.2f}s/pass  eta {eta:.1f}m",
                    flush=True,
                )
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    processor.save_pretrained(str(out_dir))
    print(f"saved adapter to {out_dir}", flush=True)

    if args.eval_samples:
        print("\n--- inline eval on val rows (same metric as benchmark.py) ---",
              flush=True)
        model.eval()
        per_lang: dict[str, list[float]] = {}
        for lang in LANGS:
            p = Path(args.data_root) / lang / "val.jsonl"
            if not p.exists():
                continue
            sample = [
                json.loads(line)
                for line in p.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            random.Random(args.seed + 99).shuffle(sample)
            sample = sample[: args.eval_samples]
            cers: list[float] = []
            for item in sample:
                try:
                    hyp = predict(model, processor, item["image"])
                except Exception as exc:
                    print(f"    predict failed: {exc}", file=sys.stderr)
                    continue
                cers.append(cer(item["text"], hyp))
            if not cers:
                continue
            mean_cer = sum(cers) / len(cers)
            per_lang[lang] = cers
            verdict = "PASS" if mean_cer <= PASS_BAR else "FAIL"
            print(
                f"  {lang}  n={len(cers):<3} mean CER {mean_cer:.4f}  {verdict}",
                flush=True,
            )
        all_cers = [c for v in per_lang.values() for c in v]
        if all_cers:
            overall = sum(all_cers) / len(all_cers)
            passed = sum(
                1
                for lang, cers in per_lang.items()
                if sum(cers) / len(cers) <= PASS_BAR
            )
            print(
                f"\n  overall mean CER {overall:.4f}  "
                f"languages passed {passed}/{len(per_lang)}",
                flush=True,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
