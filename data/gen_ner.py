"""Generate BIO-tagged prescription text with known ground truth.

Text-level, so the tagger learns slot structure rather than render artifacts.
Surface forms vary widely so no fixed template can be memorised.
"""
import json, random, sys

DRUGS = ["Amoxicillin","Amoxicilina","Amoxicilline","Metformin","Metformina","Lisinopril",
 "Lisinoprilo","Atorvastatin","Atorvastatina","Ibuprofen","Ibuprofeno","Paracetamol",
 "Omeprazole","Omeprazol","Azithromycin","Azitromicina","Amlodipine","Sertraline",
 "Warfarin","Levothyroxine","Levotiroxina","Prednisone","Prednisona","Gabapentin",
 "Ciprofloxacin","Ciprofloxacino","Ciprofloxacine","Metoprolol","Hydrochlorothiazide",
 "Duloxetine","Losartan","Simvastatin","Clotrimazole","Clotrimazol","Fenobarbital"]
DOSES = ["250","500","850","1000","5","10","20","40","75","100","150","200","400","600"]
UNITS = ["mg","mcg","g","ml","UI"]
ROUTES = ["oral","via oral","sublingual","topical","IV","IM","nasal","rectal"]
FREQS = [("BID","twice daily"),("TID","three times daily"),("QID","four times daily"),
 ("QD","once daily"),("q8h","every 8 hours"),("q12h","every 12 hours"),("q6h","every 6 hours"),
 ("PRN","as needed"),("OD","right eye"),("OS","left eye"),("OU","both eyes")]
DURS = ["x 7 days","for 7 days","por 7 dias","x 10 days","for 10 days","durante 5 dias",
 "3 days","14 days","x 5 d","during 3 days"]

SIG_FORMS = [
 "Take one {form} by mouth {freq} for {n} days",
 "Take {n2} {form}s {freq}",
 "Take one {form} every {h} hours as needed",
 "Apply {topical} to the affected area {freq}",
 "Dissolve in water and take {freq}",
 "One {form} in the morning and one at night",
]
COUNT = {"one":"1","two":"2","three":"3","1":"one","2":"two","3":"three"}
NL_FREQ = {"twice daily":2,"three times daily":3,"once daily":1,"four times daily":4}

LABELS = ["O","B-DRUG","I-DRUG","B-DOSE","I-DOSE","B-UNIT","I-UNIT","B-ROUTE","I-ROUTE",
          "B-FREQ","I-FREQ","B-DURATION","I-DURATION"]

def emit(words, tag):
    return [(w, ("B-" if i == 0 else "I-") + tag) for i, w in enumerate(words.split())]

def duration(text):
    out = []
    for p in text.split():
        out += emit(p, "DURATION") if p.isdigit() else [(p, "O")]
    return out


def nat_lang_sample():
    """Natural-language Sig lines, the way real prescriptions are written."""
    drug = random.choice(DRUGS)
    dose = random.choice(DOSES)
    unit = random.choice(UNITS)
    form = random.choice(["capsule","tablet","capsule","tablet"])
    freq = random.choice(["twice daily","three times daily","once daily","four times daily"])
    n = random.choice([5,7,10,14])
    pairs = emit(drug, "DRUG") + emit(dose, "DOSE") + emit(unit, "UNIT")
    if random.random() < 0.45:
        pairs += emit(random.choice(ROUTES), "ROUTE")
    pairs += emit("Take", "O")
    word = random.choice(["one","two","three"])
    pairs += emit(word, "O")
    pairs += emit(form, "O")
    pairs += emit("by mouth", "O")
    pairs += emit(freq, "FREQ")
    pairs += emit("for", "O") + emit(str(n), "DURATION") + emit("days", "O")
    pairs.append((".", "O"))
    return pairs

def sample():
    pairs = []
    for _ in range(random.choices([1,2,3], weights=[0.72,0.22,0.06])[0]):
        pairs += emit(random.choice(DRUGS), "DRUG")
        pairs += emit(random.choice(DOSES), "DOSE")
        pairs += emit(random.choice(UNITS), "UNIT")
        if random.random() < 0.45:
            pairs += emit(random.choice(ROUTES), "ROUTE")
        if random.random() < 0.80:
            raw, _ = random.choice(FREQS)
            pairs += emit(raw, "FREQ")
        if random.random() < 0.65:
            pairs += duration(random.choice(DURS))
        pairs.append((".", "O"))
    return pairs

def main(n, path, seed=0):
    random.seed(seed)
    rows = [{"text": " ".join(t for t,_ in sample()),
             "tags": [g for _,g in sample2] } if False else None for _ in range(0)]
    rows = []
    for _ in range(n):
        p = sample()
        rows.append({"text": " ".join(t for t,_ in p), "tags": [g for _,g in p]})
    with open(path, "w") as f: json.dump(rows, f, ensure_ascii=False)
    lab = set()
    for r in rows: lab |= set(r["tags"])
    assert lab <= set(LABELS), f"unexpected labels {lab - set(LABELS)}"
    print(f"generated {n} -> {path} | labels seen: {sorted(lab)}")
    print("sample:", rows[0]["text"][:100])

if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2], int(sys.argv[3]) if len(sys.argv)>3 else 0)
