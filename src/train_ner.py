"""Train a DistilBERT token-classification slot tagger on prescription text.

BIO labels are aligned to subword tokens: the first subword of a word carries
the word's tag, continuation subwords carry I-<tag>, and special tokens and
padding are masked to -100.
"""
import argparse, json, os, random, time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForTokenClassification,
                          get_linear_schedule_with_warmup)
from seqeval.metrics import f1_score, classification_report

LABELS = ["O","B-DRUG","I-DRUG","B-DOSE","I-DOSE","B-UNIT","I-UNIT","B-ROUTE","I-ROUTE",
          "B-FREQ","I-FREQ","B-DURATION","I-DURATION"]
L2I = {l:i for i,l in enumerate(LABELS)}
I2L = {i:l for l,i in L2I.items()}
MODEL_ID = "distilbert-base-uncased"
MAXLEN = 96

def align(words, tags, tok):
    enc = tok(words, is_split_into_words=True, truncation=True, max_length=MAXLEN,
              return_tensors="pt")
    word_ids = enc.word_ids()
    labels, prev_wid = [], None
    for wid in word_ids:
        if wid is None:
            labels.append(-100)
        elif wid != prev_wid:
            labels.append(L2I[tags[wid]])
        else:
            t = tags[wid]
            labels.append(L2I["I-" + t[2:]] if t.startswith("B-") else L2I[t])
        prev_wid = wid
    return enc["input_ids"][0], torch.tensor(labels)

class NERData(Dataset):
    def __init__(self, rows, tok):
        self.items = [align(r["text"].split(), r["tags"], tok) for r in rows]
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def collate(batch):
    ids = torch.nn.utils.rnn.pad_sequence(
        [b[0] for b in batch], batch_first=True, padding_value=0)
    lab = torch.nn.utils.rnn.pad_sequence(
        [b[1] for b in batch], batch_first=True, padding_value=-100)
    mask = (ids != 0).long()
    return {"input_ids": ids, "attention_mask": mask, "labels": lab}

def main(args):
    random.seed(0); torch.manual_seed(0)
    rows = json.load(open(args.data))
    random.shuffle(rows)
    n_val = max(64, int(len(rows)*args.val_frac))
    val, train = rows[:n_val], rows[n_val:]
    print(f"train={len(train)} val={len(val)}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_ID, num_labels=len(LABELS), id2label=I2L, label2id=L2I).to(args.device)
    tl = DataLoader(NERData(train, tok), batch_size=args.batch, shuffle=True, collate_fn=collate)
    vl = DataLoader(NERData(val, tok), batch_size=args.batch, shuffle=False, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(tl)*args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(total*0.1), total)

    model.train(); t0 = time.time()
    for ep in range(args.epochs):
        tot=n=0
        for b in tl:
            b = {k:v.to(args.device) for k,v in b.items()}
            out = model(**b)
            opt.zero_grad(set_to_none=True); out.loss.backward(); opt.step(); sched.step()
            tot += float(out.loss); n += 1
        print(f"EPOCH {ep} loss={tot/max(n,1):.4f} elapsed={time.time()-t0:.0f}s", flush=True)
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out); tok.save_pretrained(args.out)
    print(f"saved -> {args.out} wall={time.time()-t0:.0f}s", flush=True)

    model.eval(); preds, golds = [], []
    with torch.no_grad():
        for b in vl:
            b = {k:v.to(args.device) for k,v in b.items()}
            logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
            for pi, gi in zip(logits.argmax(-1).cpu(), b["labels"].cpu()):
                p, g = [], []
                for a, bb in zip(pi, gi):
                    if bb == -100: continue
                    p.append(I2L[int(a)]); g.append(I2L[int(bb)])
                preds.append(p); golds.append(g)
    f1 = f1_score(golds, preds)
    print(f"VAL_F1={f1:.4f}  (seqeval entity-level, n={len(golds)})", flush=True)
    print(classification_report(golds, preds, digits=3, zero_division=0), flush=True)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/ner_synth.json")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--val-frac", dest="val_frac", type=float, default=0.08)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default="models/rxner")
    a = p.parse_args()
    main(a)
