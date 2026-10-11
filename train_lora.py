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

LM_TARGETS = [
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

VISION_TARGETS = [
    "mlp.fc1",
    "mlp.fc2",
    "patch_embedding",
    "multi_modal_projector.linear_1",
    "multi_modal_projector.linear_2",
]

LORA_TARGETS = LM_TARGETS + VISION_TARGETS

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


def load_rows(root: Path, per_lang: int, seed: int = 0,
              per_lang_chars: int = 40000) -> list[dict]:
    """Load train+val rows, capped per language. test.jsonl is never opened.

    Two caps, both necessary. The row cap stops the six languages with ~3200
    available rows drowning out en (60), ar (366) and vi (944).

    The cap that actually matters is on *characters*, not rows. Measured on the
    built corpus (train+val answer chars): ru 12.1M, fa 3.3M, vi 0.41M, ar
    0.29M, fr 0.15M, de 0.13M, th 0.13M, ur 0.10M, en 62k, hi 65k. The five
    line-form languages (fr/th/de/hi/ur) are ~1% each, while ru+fa are 92% of
    all training characters. Because the loss is per token, a run capped at
    1500 rows per language still taught the model "emit a long Russian essay"
    as its default behaviour: it graded CER 5.66-13.43 on the short-reference
    languages and answered a French reference with fluent Russian prose.

    The paragraph-form corpora (ru, fa, ar, vi, en) are kept but subsampled by
    character rather than dropped -- they are the only evidence those languages
    offer, and the benchmark grades them at median 3688/882/733/447/1452 chars.
    Truncating a paragraph would produce an invalid transcription target, so
    rows are sampled until the budget is spent, never cut mid-way.

    Each language therefore contributes at most `per_lang_chars` answer
    characters, which is what equalises the gradient signal. 40000 is ~10x the
    longest line-form reference (ur max 74, fr max 84) and ~1% of a single
    Russian paragraph, so short languages keep every row and long ones keep
    hundreds of distinct paragraphs.
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
        used = 0
        kept: list[dict] = []
        for row in found:
            n = len(row.get("text") or "")
            if used + n > per_lang_chars:
                continue
            used += n
            kept.append(row)
        rows.extend(kept)
        print(f"  {lang}: {len(kept)} rows, {used} answer chars", flush=True)
    # Shuffle across languages. The training loop reads rows[i % len(rows)]
    # sequentially, so without this the model sees all of one language, then all
    # of the next - a guaranteed forgetting spiral across ten scripts.
    random.Random(seed + 1).shuffle(rows)
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
    # LFM2.5-VL closes an assistant turn with the EOS token. Without it in the
    # labels the model never learns a stop rule: th/hi/ur score exact lines and
    # then ramble (CER 3-6), and ru never terminates at all. Appending the eos
    # string before tokenising keeps the prompt an exact prefix of the full
    # sequence, so the prompt-prefix mask below stays correct.
    answer = answer + processor.tokenizer.eos_token

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


def predict(model, processor, image_path: str, device: str) -> str:
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
    # benchmark.py moves the batch to the device; omitting it here makes the
    # first eval sample fail with 'mat1 is on cpu', and every sample after it.
    enc = to_device(
        processor(text=prompt, images=[image], return_tensors="pt"), device
    )
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
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-lang", type=int, default=1500)
    ap.add_argument("--per-lang-chars", type=int, default=40000,
                    help="max answer chars per language; equalises gradient weight")
    ap.add_argument("--log", type=int, default=10)
    ap.add_argument("--eval-samples", type=int, default=20)
    # A previous run OOM'd at pass 5000/8000 and lost every minute because the
    # only save happened at the end.
    ap.add_argument("--save-every", type=int, default=100)
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

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_adapter() -> None:
        model.save_pretrained(str(out_dir))
        processor.save_pretrained(str(out_dir))

    rows = load_rows(Path(args.data_root), args.per_lang, args.seed,
                     per_lang_chars=args.per_lang_chars)
    if not rows:
        print("no corpus rows found", file=sys.stderr)
        return 1
    save_adapter()
    print(f"baseline adapter saved to {out_dir}", flush=True)

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

    failures = 0
    abort: RuntimeError | None = None
    try:
        while len(losses) < args.steps * args.accum:
            row = rows[micro % len(rows)]
            micro += 1
            batch = collate(processor, row)
            if batch is None:
                failures += 1
                if failures >= 100:
                    abort = RuntimeError(f"{failures} consecutive unusable samples")
                    break
                continue
            batch = to_device(batch, args.device)
            # Catch only OutOfMemoryError, and only around the memory-hungry
            # calls: the first run died inside backward() on a transient spike.
            # Catching RuntimeError here instead would swallow genuine shape and
            # remote-code bugs and turn a broken run into a silent no-op, which
            # is the failure mode smoke_grads.py exists to catch.
            try:
                out = model(**batch)
                (out.loss / args.accum).backward()
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("  OOM on this sample; skipping", file=sys.stderr)
                opt.zero_grad()
                accum_left = args.accum
                failures += 1
                if failures >= 50:
                    abort = RuntimeError(f"{failures} consecutive OOM skips")
                    break
                continue
            failures = 0
            if not torch.isfinite(out.loss):
                print("non-finite loss, skipping", file=sys.stderr)
                opt.zero_grad()
                accum_left = args.accum
                continue
            losses.append(float(out.loss.detach().cpu()))
            accum_left -= 1
            if accum_left:
                continue

            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            accum_left = args.accum

            n = len(losses)
            if args.save_every and (n // args.accum) % args.save_every == 0:
                save_adapter()
                print(f"  checkpoint at opt step {n // args.accum}", flush=True)
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

    save_adapter()
    print(f"saved adapter to {out_dir}", flush=True)
    if abort is not None:
        raise abort

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
                    hyp = predict(model, processor, item["image"], args.device)
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
