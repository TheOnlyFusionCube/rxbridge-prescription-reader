"""Add OCR-noise augmentation to the NER corpus.

docTR's real output on prescriptions contains character swaps, dropped
characters, merged words and split words (e.g. "Clotrimazol (500 mg)" ->
"Mletzina 20"). A tagger trained only on clean text cannot survive that, so
we corrupt clean synthetic spans the same way docTR does.
"""
import json, random, sys

SUBS = {"o":"0","i":"1","l":"1","e":"c","a":"o","s":"5","t":"f","n":"m","c":"e","u":"v","m":"n","z":"s"}
SPLIT_WORDS = ["Amoxicillin","Metformin","Ciprofloxacin","Hydrochlorothiazide","Levothyroxine"]

def corrupt_word(w, rng):
    if len(w) < 3: return w
    r = rng.random()
    if r < 0.22:                                   # substitution
        i = rng.randrange(len(w))
        return w[:i] + SUBS.get(w[i].lower(), rng.choice("aeiou")) + w[i+1:]
    if r < 0.38:                                   # deletion
        i = rng.randrange(1, len(w))
        return w[:i] + w[i+1:]
    if r < 0.50:                                   # duplication
        i = rng.randrange(len(w))
        return w[:i] + w[i] + w[i:]
    if r < 0.62:                                   # transposition
        i = rng.randrange(len(w)-1)
        return w[:i] + w[i+1] + w[i] + w[i+2:]
    return w

def maybe_split(w, rng):
    if w in SPLIT_WORDS and rng.random() < 0.30:
        i = rng.randrange(3, len(w)-2)
        return [w[:i], w[i:]]
    return [w]

def augment_row(row, rng):
    """Corrupt tokens while keeping the BIO tags aligned."""
    words, tags = row["text"].split(), row["tags"]
    out_w, out_t = [], []
    for w, t in zip(words, tags):
        if t == "O" and rng.random() < 0.35:
            continue                               # drop a filler word
        pieces = maybe_split(w, rng)
        if len(pieces) == 1 and rng.random() < 0.55:
            pieces = [corrupt_word(w, rng)]
        for p in pieces:
            out_w.append(p)
            out_t.append(t if not t.startswith("B-") else "I-" + t[2:])
    if not out_w:
        return None
    return {"text": " ".join(out_w), "tags": out_t}

def main(src, dst, mult=3, seed=0):
    rows = json.load(open(src))
    rng = random.Random(seed)
    out = list(rows)
    made = 0
    for _ in range(mult):
        for r in rows:
            a = augment_row(r, rng)
            if a: out.append(a); made += 1
    json.dump(out, open(dst, "w"), ensure_ascii=False)
    print(f"clean={len(rows)} augmented=+{made} total={len(out)} -> {dst}")
    for r in out[:2] + out[len(rows):len(rows)+2]:
        print("  ", r["text"][:88])

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv)>3 else 3)
