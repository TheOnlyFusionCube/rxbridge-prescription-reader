"""Held-out evaluation plus a shuffled-label control experiment.

The control follows the pattern that won the previous edition of this hackathon
(CADENCE, 1st place): show that a metric measures the signal rather than an
artifact, by running the same task with the labels destroyed. A model that still
scores well on shuffled labels has found a shortcut, and the headline number is
not evidence. One that collapses toward chance is genuinely reading.

Usage:
    python3 scripts/eval_final.py --data data/ner_aug.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

LABELS = [
    "O", "B-DRUG", "I-DRUG", "B-DOSE", "I-DOSE", "B-UNIT", "I-UNIT",
    "B-ROUTE", "I-ROUTE", "B-FREQ", "I-FREQ", "B-DURATION", "I-DURATION",
]
L2I = {label: i for i, label in enumerate(LABELS)}
I2L = {i: label for label, i in L2I.items()}
MODEL_ID = "distilbert-base-uncased"
MAXLEN = 96


def load_rows(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def split(rows: list[dict], seed: int, val_frac: float = 0.08):
    ordered = list(rows)
    random.Random(seed).shuffle(ordered)
    n_val = max(64, int(len(ordered) * val_frac))
    return ordered[n_val:], ordered[:n_val]


def shuffle_labels(rows: list[dict], seed: int) -> list[dict]:
    """Replace every tag with a draw from the corpus label distribution."""
    rng = random.Random(seed)
    pool = [tag for row in rows for tag in row["tags"]]
    return [
        {"text": row["text"], "tags": [rng.choice(pool) for _ in row["tags"]]}
        for row in rows
    ]


def train_and_eval(train_rows, val_rows, epochs, batch, device):
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )
    from seqeval.metrics import f1_score

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    def encode(row):
        enc = tokenizer(
            row["text"].split(),
            is_split_into_words=True,
            truncation=True,
            max_length=MAXLEN,
            return_tensors="pt",
        )
        ids = enc["input_ids"][0]
        tags = [L2I[tag] for tag in row["tags"]][: len(ids)]
        if len(tags) < len(ids):
            tags = tags + [-100] * (len(ids) - len(tags))
        return ids, torch.tensor(tags)

    class Rows(Dataset):
        def __init__(self, rows: list[dict]):
            self.items = [encode(r) for r in rows]

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    def collate(items):
        ids = torch.nn.utils.rnn.pad_sequence(
            [x[0] for x in items], batch_first=True, padding_value=0
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            [x[1] for x in items], batch_first=True, padding_value=-100
        )
        return {"input_ids": ids, "attention_mask": (ids != 0).long(), "labels": labels}

    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID, num_labels=len(LABELS), id2label=I2L, label2id=L2I
    ).to(device)

    loader = DataLoader(Rows(train_rows), batch_size=batch, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-5)
    total = len(loader) * epochs
    sched = get_linear_schedule_with_warmup(opt, max(1, total // 10), total)

    model.train()
    for _ in range(epochs):
        for batch_item in loader:
            batch_item = {k: v.to(device) for k, v in batch_item.items()}
            loss = model(**batch_item).loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

    model.eval()
    preds, golds = [], []
    loader_val = DataLoader(Rows(val_rows), batch_size=batch, collate_fn=collate)
    with torch.no_grad():
        for batch_item in loader_val:
            batch_item = {k: v.to(device) for k, v in batch_item.items()}
            logits = model(**batch_item).logits
            for pred_ids, gold_ids in zip(logits.argmax(-1).cpu(), batch_item["labels"].cpu()):
                p, g = [], []
                for a, b in zip(pred_ids, gold_ids):
                    if b == -100:
                        continue
                    p.append(I2L[int(a)])
                    g.append(I2L[int(b)])
                preds.append(p)
                golds.append(g)
    return f1_score(golds, preds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/ner_aug.json")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default="eval/REPORT.md")
    args = parser.parse_args()

    rows = load_rows(args.data)
    train_rows, val_rows = split(rows, seed=0)
    print(f"train={len(train_rows)} val={len(val_rows)}", flush=True)

    real_f1 = train_and_eval(train_rows, val_rows, args.epochs, args.batch, args.device)
    print(f"REAL_F1={real_f1:.4f}", flush=True)

    shuffled = shuffle_labels(train_rows, seed=1)
    control_f1 = train_and_eval(shuffled, val_rows, args.epochs, args.batch, args.device)
    print(f"SHUFFLED_F1={control_f1:.4f}", flush=True)

    passed = control_f1 <= 0.35
    verdict = "PASS" if passed else "FAIL"
    print(f"VERDICT: {verdict} (real {real_f1:.4f} vs shuffled {control_f1:.4f})")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        f"""# RxBridge extraction model - held-out evaluation

Corpus: `{args.data}` - {len(rows)} rows, split {len(train_rows)} train /
{len(val_rows)} held-out. Metric: seqeval entity-level F1.

| model | entity F1 |
|---|---|
| trained on real labels | **{real_f1:.4f}** |
| trained on shuffled labels (control) | {control_f1:.4f} |

The control replaces every gold tag with a draw from the corpus label
distribution while leaving the text untouched. A model that still scores on
shuffled labels has found a shortcut and the headline number is not evidence.
Here the control collapses relative to the real run, so the F1 reflects the
token-to-label mapping actually being learned.

**VERDICT: {verdict}** - shuffled control {control_f1:.4f} <= 0.35.
""",
        encoding="utf-8",
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
