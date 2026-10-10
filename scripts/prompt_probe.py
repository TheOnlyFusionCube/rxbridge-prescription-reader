"""Does the PROMPT WORDING close the stop-rule gap that decode levers could not?

Every decode lever is now measured and at its ceiling: the 1024-token budget
made things worse (mean CER 0.5279 -> 0.9604), penalties bought no new pass and
hurt vi (0.2616 -> 0.2648), and the post-hoc 1024+penalty combination reaches
4 languages only by adding vi, while ur degrades to 3.2033. All of that acts on
the model AFTER it has already decided to emit a 500-char essay. The one lever
never tested is the text the model reads before the image, which sets that
decision in the first place.

The measured failure mode is a stop-signal gap: fr emitted 34/40 predictions
longer than 380 chars against 48-char references before the EOS fix, and the
post-1024 run still shows a language emitting a 500-char essay where a 40-char
line belongs. So this probe tests three prompt variants that all end in an
explicit stop instruction:

  base    - the exact prompt every published run used (control)
  oneline - base + "Output exactly one line of text and stop."
  stop    - base + "Stop immediately after the last character of the
             transcription."

Measured on the VAL shards only, which are training data (train_lora.py
load_rows reads train+val), so nothing here is selected against the held-out
test shards benchmark.py grades. That is a real limitation and is disclosed
wherever the result is used: val is contaminated, so this probe can show a
prompt helps on data the model trained on, but it cannot prove a test-shard
gain. A prompt chosen here must be re-verified on a held-out shard before any
claim that it generalizes.

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
BASE_PROMPT = ("Read the handwriting in this image and transcribe it exactly as written, "
               "preserving the original language and spelling. Reply with only the "
               "transcribed text, nothing else.")
# The 7 languages that still fail after the 1024+penalty run. de/fr/th/vi
# already pass there and are not re-measured.
LANGS = ["ur", "fa", "ar", "ru", "en", "hi", "vi"]
N = 15

VARIANTS = [
    ("base", BASE_PROMPT),
    ("oneline", BASE_PROMPT + " Output exactly one line of text and stop."),
    ("stop", BASE_PROMPT + " Stop immediately after the last character of "
                           "the transcription."),
]


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
    print(f"adapter loaded, {len(rows)} val rows x {len(VARIANTS)} prompts",
          flush=True)

    res = {}
    for name, prompt in VARIANTS:
        by = {}
        for r in rows:
            img = Image.open(r["image"]).convert("RGB")
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}]}]
            text = proc.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False)
            enc = proc(text=text, images=[img], return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=128,
                                     do_sample=False)
            pred = proc.batch_decode(
                out[:, enc["input_ids"].shape[-1]:], skip_special_tokens=True)[0]
            by.setdefault(r["lang"], []).append(cer(r["text"], pred))
        res[name] = {l: sum(v) / len(v) for l, v in by.items()}
        print(f"  done {name}", flush=True)

    print()
    print("%-10s %s  %s" % ("prompt",
                            "  ".join("%-7s" % l for l in LANGS), "MEAN"))
    for name, _ in VARIANTS:
        d = res[name]
        mean = sum(d.values()) / len(d)
        print("%-10s %s  %.4f" % (name,
                                  "  ".join("%-7.4f" % d[l] for l in LANGS), mean))
    return 0


if __name__ == "__main__":
    sys.exit(main())
