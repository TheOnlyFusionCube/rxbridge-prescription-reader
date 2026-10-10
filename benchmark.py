"""Multilingual handwriting benchmark: the verifier.

For each language this scores a model's transcription accuracy on a held-out
sample that was never used in training. Reported metrics are CER (character
error rate, normalised so it is comparable across scripts) and exact match.

A benchmark that only reported the language a model is best at would not be a
verifier, so this grades every available language and requires each one to clear
the bar before the run is called a pass.

Usage:
    python3 benchmark.py --model LiquidAI/LFM2.5-VL-3B --adapter checkpoints/lora
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import unicodedata
from pathlib import Path

LANGUAGES: list[tuple[str, str]] = [
    ("en", "English"), ("fr", "French"), ("de", "German"), ("ru", "Russian"),
    ("zh", "Chinese"), ("th", "Thai"), ("vi", "Vietnamese"),
    ("ar", "Arabic"), ("ko", "Korean"), ("ur", "Urdu"),
    ("or", "Odia"), ("fa", "Persian"), ("hi", "Hindi"),
    ("es", "Spanish"), ("pt", "Portuguese"), ("it", "Italian"),
]

CER_BAR = 0.25

PROMPT = (
    "Read the handwriting in this image and transcribe it exactly as written, "
    "preserving the original language and spelling. Reply with only the "
    "transcribed text, nothing else."
)


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


def exact_match(reference: str, hypothesis: str) -> bool:
    return normalise(reference).lower() == normalise(hypothesis).lower()


def build_split(root: str, seed: int = 0, per_lang: int = 40) -> dict[str, list[dict]]:
    """Read the held-out test shard per language."""
    root_path = Path(root)
    out: dict[str, list[dict]] = {}
    for code, _name in LANGUAGES:
        shard = root_path / code / "test.jsonl"
        if not shard.exists():
            continue
        rows = [json.loads(l) for l in shard.read_text().splitlines() if l.strip()]
        random.Random(seed).shuffle(rows)
        out[code] = rows[:per_lang]
    return out


def run_benchmark(model_id, adapter, data_root, per_lang, device, out_path):
    import torch
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id, dtype=torch.bfloat16, trust_remote_code=True
    )
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model = model.to(device).eval()

    samples = build_split(data_root, per_lang=per_lang)
    results: list[dict] = []
    per_language: dict[str, dict] = {}

    for code, name in LANGUAGES:
        rows = samples.get(code)
        if not rows:
            continue
        cers: list[float] = []
        hits = 0
        # One fixed budget for every language, and it must not depend on the
        # reference. An earlier version scaled the budget by len(row["text"]),
        # which handed the harness the answer length: ru/fa/en/ar were given
        # 4-15x more tokens than the zero-shot baseline had, chosen using the
        # ground truth. That is leakage, and it invalidated the comparison.
        #
        # 128 tokens is kept because it is what the zero-shot baseline used, so
        # this is the only value that makes the two arms comparable. It truncates
        # the paragraph-form corpora (ru refs median 3688 chars) - that is a
        # known, reported limitation of the harness, not something to hide by
        # reading the reference.
        budget = 128
        for row in rows:
            image = Image.open(row["image"]).convert("RGB")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": PROMPT},
                ],
            }]
            text = processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
            enc = processor(text=text, images=[image], return_tensors="pt").to(device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=budget, do_sample=False)
                prediction = processor.batch_decode(
                    out[:, enc["input_ids"].shape[-1] :], skip_special_tokens=True
                )[0]
            c = cer(row["text"], prediction)
            cers.append(c)
            hits += exact_match(row["text"], prediction)
            results.append({
                "lang": code, "reference": row["text"],
                "prediction": prediction, "cer": round(c, 4),
            })

        n = len(cers)
        mean_cer = sum(cers) / n
        per_language[code] = {
            "name": name,
            "n": n,
            "mean_cer": round(mean_cer, 4),
            "median_cer": round(sorted(cers)[n // 2], 4),
            "exact_match": round(hits / n, 4),
            "pass": mean_cer <= CER_BAR,
        }
        print(
            f"{code}  {name:<12} n={n:<4} CER={mean_cer:.4f}  exact={hits/n:.2f}"
            f"  {'PASS' if per_language[code]['pass'] else 'FAIL'}",
            flush=True,
        )

    graded = list(per_language.values())
    passed = [x for x in graded if x["pass"]]
    summary = {
        "model": model_id,
        "adapter": str(adapter or "none"),
        "languages_graded": len(graded),
        "languages_passed": len(passed),
        "bar": CER_BAR,
        "overall_mean_cer": round(sum(x["mean_cer"] for x in graded) / max(len(graded), 1), 4),
        "verdict": "PASS" if len(passed) == len(graded) else "FAIL",
        "per_language": per_language,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=1)
    )
    print()
    print("=" * 64)
    print(f"LANGUAGES: {len(graded)} graded, {len(passed)} passed (bar {CER_BAR})")
    print(f"OVERALL CER: {summary['overall_mean_cer']:.4f}")
    print(f"VERDICT: {summary['verdict']}")
    print(f"wrote {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="LiquidAI/LFM2.5-VL-3B")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--data-root", default="htr")
    ap.add_argument("--per-lang", type=int, default=40)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="eval/BENCHMARK.json")
    args = ap.parse_args()

    summary = run_benchmark(
        args.model, args.adapter or None, args.data_root,
        args.per_lang, args.device, args.out,
    )
    sys.exit(0 if summary["verdict"] == "PASS" else 1)


if __name__ == "__main__":
    main()
