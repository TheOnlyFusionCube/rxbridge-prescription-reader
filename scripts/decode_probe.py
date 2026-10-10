"""Does a decode-time stall penalty help, now that the model has a stop rule?

Measured on the VAL shards only, which are training data (load_rows reads
train+val), so nothing here is selected against the held-out test shards that
benchmark.py grades. That is a real limitation and is disclosed wherever the
result is used: val is contaminated, so this probe can tell us a penalty does
not hurt, but it cannot prove a test-shard gain.

An earlier probe tested repetition_penalty against adapter_v2 and it made
things 2.6x worse. That was before the EOS stop-rule fix (5e44158). This probe
re-tests against adapter_v4, which now stops, to see whether the answer changed
with the fix or whether the penalty is simply the wrong lever for this model.

Prints only CER values; no transcript text enters the log.
"""

import argparse
import json
import random
import sys

import torch
from PIL import Image

DEFAULT_ROOT = "/home/yoka-aiserver11/rxbridge"
DEFAULT_ADAPTER = f"{DEFAULT_ROOT}/lfm/adapter_v4"
PROMPT = ("Read the handwriting in this image and transcribe it exactly as written, "
          "preserving the original language and spelling. Reply with only the "
          "transcribed text, nothing else.")
# The languages that still fail at the 128-token budget. de/fr/th already pass
# and are not re-measured here.
LANGS = ["vi", "hi", "ur", "ar", "fa"]
N = 15


def cer(ref: str, hyp: str) -> float:
    a = " ".join((ref or "").split())
    b = " ".join((hyp or "").split())
    if not a:
        return 0.0 if not b else 1.0
    if a == b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1] / len(a)


# 1.0 / 0 is exactly what benchmark.py defaults to and what every earlier run
# used, so the first row here is the control the others are judged against.
SETTINGS = [
    ("default", {}),
    ("penal1.3", {"repetition_penalty": 1.3}),
    ("ngram3", {"no_repeat_ngram_size": 3}),
    ("pen1.2+ngram3", {"repetition_penalty": 1.2, "no_repeat_ngram_size": 3}),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import PeftModel

    rows = []
    for lang in LANGS:
        rs = [json.loads(l) for l in open(f"{args.root}/htr/{lang}/val.jsonl")
              if l.strip()]
        random.Random(0).shuffle(rs)
        for r in rs[:N]:
            r["lang"] = lang
            rows.append(r)

    proc = AutoProcessor.from_pretrained(args.adapter)
    base = AutoModelForImageTextToText.from_pretrained(
        "LiquidAI/LFM2.5-VL-3B", dtype=torch.bfloat16, trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base, args.adapter).to("cuda").eval()
    print(f"adapter loaded, {len(rows)} val rows x {len(SETTINGS)} settings",
          flush=True)

    res = {}
    for name, kw in SETTINGS:
        by = {}
        for r in rows:
            img = Image.open(r["image"]).convert("RGB")
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": PROMPT}]}]
            text = proc.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)
            enc = proc(text=text, images=[img], return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=128,
                                     do_sample=False, **kw)
            pred = proc.batch_decode(
                out[:, enc["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
            by.setdefault(r["lang"], []).append(cer(r["text"], pred))
        res[name] = {l: sum(v) / len(v) for l, v in by.items()}
        print(f"  done {name}", flush=True)

    print()
    print("%-14s %s  %s" % ("setting",
                            "  ".join("%-7s" % l for l in LANGS), "MEAN"))
    for name, _ in SETTINGS:
        d = res[name]
        mean = sum(d.values()) / len(d)
        print("%-14s %s  %.4f" % (name,
                                  "  ".join("%-7.4f" % d[l] for l in LANGS), mean))
    return 0


if __name__ == "__main__":
    sys.exit(main())
