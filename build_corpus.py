"""Build per-language train/val/test shards from the downloaded HF datasets.

Every language gets three shards. ``test`` is never used by training, so the
benchmark is a real held-out measurement rather than a score on seen data.

Output rows: {"image": "<abs path>", "text": "<transcription>"}

Usage:
    python3 build_corpus.py --repo /home/yoka-aiserver11/rxbridge
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

# language -> list of (relative path under repo, reader kind)
SOURCES: dict[str, list[tuple[str, str]]] = {
    "en": [("ml/amr", "amr"), ("ml/iam", "tgz"), ("ml/rxwords", "rxwords")],
    "fr": [("ml/rimes", "parquet")],
    "th": [("htr/th", "parquet")],
    "ar": [("htr/ar", "parquet")],
    "de": [("htr/de", "parquet")],
    "fa": [("htr/fa", "parquet")],
    "ru": [("htr/ru", "parquet")],
    "vi": [("htr/vi", "parquet")],
    "zh": [("htr/zh", "parquet")],
    "ko": [("htr/ko", "ko")],
    "ur": [("htr/ur", "csv")],
    "or": [("htr/or", "auto")],
}

TRAIN_CAP = 3000
VAL_CAP = 200
TEST_CAP = 200
MIN_PER_LANG = 12


# ------------------------------------------------------------------ helpers


def _dump_png(img, out_dir: Path, cache: dict) -> str | None:
    """Materialise a parquet image cell as a PNG on disk, memoised by hash."""
    import hashlib

    if isinstance(img, dict):
        raw = img.get("bytes") or img.get("data")
    elif isinstance(img, (bytes, bytearray)):
        raw = bytes(img)
    else:
        return None
    if not raw:
        return None
    key = hashlib.sha1(raw).hexdigest()[:16]
    hit = cache.get(key)
    if hit:
        return hit
    try:
        with Image.open(io.BytesIO(raw)) as im:
            im.load()
            if im.mode not in ("RGB", "L"):
                im = im.convert("RGB")
            out_dir.mkdir(parents=True, exist_ok=True)
            path = out_dir / f"{key}.png"
            if not path.exists():
                im.save(path)
            cache[key] = str(path)
            return str(path)
    except Exception:
        return None


# ------------------------------------------------------------------ readers


def read_parquet(root: Path, cache: dict):
    """Generic reader: parquet shards with an image column and a text column."""
    cols_text = ("text", "label", "transcription", "ground_truth")
    for parquet in sorted(root.rglob("*.parquet")):
        try:
            import pandas as pd

            df = pd.read_parquet(parquet)
        except Exception:
            continue
        if not len(df):
            continue
        img_col = next((c for c in ("image", "img", "pixel_values") if c in df.columns), None)
        txt_col = next((c for c in cols_text if c in df.columns), None)
        if not img_col or not txt_col:
            continue
        out_dir = parquet.parent / "_png"
        for _, row in df.iterrows():
            text = str(row[txt_col]).strip()
            if not text:
                continue
            path = _dump_png(row[img_col], out_dir, cache)
            if path:
                yield {"image": path, "text": text}


def read_amr(root: Path, cache: dict):
    """AMR: htr/<id>.png paired with truth/<id>.txt via metadata.csv."""
    meta = root / "metadata.csv"
    if not meta.exists():
        return
    with meta.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            truth = root / row["truth_file"]
            if not truth.exists():
                continue
            text = truth.read_text(encoding="utf-8").strip()
            if not text:
                continue
            for part in row["htr_file"].split(";"):
                img = root / part.strip()
                if img.exists():
                    yield {"image": str(img), "text": text}


def read_tgz(root: Path, cache: dict):
    """IAM: IAM_lines.tgz -> <form>/<form>-<nnnn>.png + ascii/<form>/<form>-<nnnn>.txt"""
    tgz = root / "IAM_lines.tgz"
    if not tgz.exists():
        return
    out = root / "_lines"
    if not out.exists():
        out.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(str(tgz), str(out))
    for txt in sorted(out.rglob("*.txt")):
        form = txt.parent.name
        for i, line in enumerate(txt.read_text(errors="ignore").splitlines()):
            line = line.strip()
            if not line:
                continue
            img = out / form / f"{form}-{i:04d}.png"
            if img.exists():
                yield {"image": str(img), "text": line}


def read_rxwords(root: Path, cache: dict):
    """KWS-style: images/*.png with a lines.csv / labels csv alongside."""
    imgs = sorted(root.rglob("*.png")) + sorted(root.rglob("*.jpg"))
    labels = list(root.rglob("*.csv")) + list(root.rglob("*.jsonl"))
    for lab in labels:
        rows: dict[str, str] = {}
        if lab.suffix == ".jsonl":
            for line in lab.read_text().splitlines():
                try:
                    rows.update({json.loads(line)["image"]: json.loads(line)["text"]})
                except Exception:
                    pass
        else:
            for row in csv.DictReader(lab.open(encoding="utf-8")):
                k = row.get("image") or row.get("file") or row.get("filename") or ""
                v = (row.get("text") or row.get("label") or "").strip()
                if k and v:
                    rows[Path(k).name] = v
        for img in imgs:
            text = rows.get(img.name, "")
            if text:
                yield {"image": str(img), "text": text}


def read_ko(root: Path, cache: dict):
    """Korean dataset ships .jpg with a sibling .json holding the text."""
    for jpg in sorted(root.rglob("*.jpg")):
        js = jpg.with_suffix(".json")
        if not js.exists():
            continue
        try:
            text = str(json.loads(js.read_text()).get("text", "")).strip()
        except Exception:
            continue
        if text:
            yield {"image": str(jpg), "text": text}


def read_csv(root: Path, cache: dict):
    """CSV with image path + text columns."""
    for csv_file in sorted(root.rglob("*.csv")):
        try:
            with csv_file.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    img_key = next(
                        (k for k in ("image", "file", "path", "filename") if k in row), None
                    )
                    txt_key = next((k for k in ("text", "label", "transcription") if k in row), None)
                    if not img_key or not txt_key:
                        continue
                    text = str(row[txt_key]).strip()
                    img = csv_file.parent / str(row[img_key])
                    if text and img.exists():
                        yield {"image": str(img), "text": text}
        except Exception:
            continue


def read_auto(root: Path, cache: dict):
    yield from read_parquet(root, cache)


READERS = {
    "parquet": read_parquet,
    "amr": read_amr,
    "tgz": read_tgz,
    "rxwords": read_rxwords,
    "ko": read_ko,
    "csv": read_csv,
    "auto": read_auto,
}


# ------------------------------------------------------------------ core


def write_shards(lang: str, rows: list[dict], out_root: Path, seed: int = 0):
    random.Random(seed).shuffle(rows)
    n_test = min(TEST_CAP, max(MIN_PER_LANG, len(rows) // 5))
    n_val = min(VAL_CAP, max(4, len(rows) // 10))
    test = rows[:n_test]
    val = rows[n_test : n_test + n_val]
    train = rows[n_test + n_val : n_test + n_val + TRAIN_CAP]

    out_dir = out_root / lang
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, shard in (("train", train), ("val", val), ("test", test)):
        p = out_dir / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as f:
            for r in shard:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(train), len(val), len(test)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/yoka-aiserver11/rxbridge")
    ap.add_argument("--root", default="htr")
    args = ap.parse_args()

    repo = Path(args.repo)
    data_root = repo / args.root
    cache: dict[str, str] = {}
    report = []

    for lang, sources in SOURCES.items():
        rows: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for rel, kind in sources:
            path = repo / rel
            if not path.exists():
                continue
            reader = READERS[kind]
            try:
                for row in reader(path, cache):
                    if not row["text"]:
                        continue
                    key = (row["image"], row["text"][:80])
                    if key in seen:
                        continue
                    seen.add(key)
                    rows.append(row)
                print(f"  {lang}/{rel}: +{len(rows)} running total", flush=True)
            except Exception as exc:
                print(f"  {lang}/{rel}: READER FAILED {exc}", file=sys.stderr)

        if len(rows) < MIN_PER_LANG:
            report.append((lang, 0, 0, 0, f"SKIP ({len(rows)} rows)"))
            continue
        n_tr, n_va, n_te = write_shards(lang, rows, data_root)
        report.append((lang, n_tr, n_va, n_te, "OK"))
        print(f"{lang:5s} total={len(rows):<6} train={n_tr:<5} val={n_va:<4} test={n_te}", flush=True)

    print()
    print(f"{'lang':5s} {'train':>6} {'val':>5} {'test':>5}  status")
    for lang, a, b, c, status in report:
        print(f"{lang:5s} {a:>6} {b:>5} {c:>5}  {status}")
    ok = sum(1 for r in report if r[4] == "OK")
    print()
    print(f"LANGUAGES READY: {ok}")


if __name__ == "__main__":
    main()
