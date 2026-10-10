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
import json
import random
import shutil
import sys
from pathlib import Path

# language -> list of (relative path under repo, reader kind)
# Only languages with verified image+transcript pairs are listed. Odia ships
# character-classification metadata with no images; Korean ships 53 polygon
# regions inside 5 images, too few for a split.
SOURCES: dict[str, list[tuple[str, str]]] = {
    # en had 74 rows (53 train), the one graded language not supply-saturated.
    # IAM pairs are line-granular while the en test shard is paragraph form, so
    # this buys script reading, not paragraph structure; test/val are restored
    # byte-identical after the build so the verifier is unchanged.
    "en": [("ml/amr", "amr"), ("fresh/iam_voxel", "iam_voxel")],
    "fr": [("ml/rimes_fr", "parquet")],
    "th": [("htr/th", "parquet")],
    "ar": [("htr/ar", "parquet")],
    "de": [("htr/de", "parquet")],
    "fa": [("htr/fa", "parquet")],
    "ru": [("htr/ru", "parquet")],
    "vi": [("htr/vi", "parquet")],
    "hi": [("htr/ml", "hi_zip")],
    "ur": [("fresh/ur", "ur_tgz")],
}

TRAIN_CAP = 3000
VAL_CAP = 200
TEST_CAP = 200
MIN_PER_LANG = 12
ROW_CAP = TRAIN_CAP + VAL_CAP + TEST_CAP


# ------------------------------------------------------------------ helpers


_IMAGE_SUFFIX = {
    b"\xff\xd8\xff": ".jpg",
    b"\x89PNG\r\n\x1a\n": ".png",
    b"GIF87a": ".gif",
    b"GIF89a": ".gif",
}


def _dump_image(img, out_dir: Path, cache: dict) -> str | None:
    """Write an image cell's bytes to disk verbatim, memoised by content hash."""
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
    # Decoding and re-encoding each cell cost seconds on the large handwritten
    # line shards, which is what stalled Arabic; the bytes are already a valid
    # image and the benchmark opens them with Image.open either way.
    suffix = next(
        (s for magic, s in _IMAGE_SUFFIX.items() if raw.startswith(magic)), ".bin"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{key}{suffix}"
    if not path.exists():
        path.write_bytes(raw)
    cache[key] = str(path)
    return str(path)


# ------------------------------------------------------------------ readers


def _text_from_messages(messages) -> str:
    """Conversation-format shards: the assistant turn carries the transcript."""
    # pandas hands nested struct columns back as ndarray, which fails the list
    # check below and silently blanks every row; tolist restores native dicts.
    if hasattr(messages, "tolist"):
        messages = messages.tolist()
    if not isinstance(messages, (list, tuple)):
        return ""
    for msg in reversed(messages):
        if not isinstance(msg, dict) or msg.get("role") == "user":
            continue
        content = msg.get("content")
        # pandas hands the nested list back as ndarray too, so coerce it before
        # the isinstance checks below or every part is skipped and the row is
        # silently blanked.
        if hasattr(content, "tolist"):
            content = content.tolist()
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                p.get("text") for p in content
                if isinstance(p, dict) and p.get("type") == "text" and p.get("text")
            ]
            if parts:
                return " ".join(str(p) for p in parts).strip()
    return ""


def read_parquet(root: Path, cache: dict):
    """Generic reader: parquet shards with an image column and a text column."""
    cols_text = ("text", "label", "transcription", "ground_truth", "content")
    for parquet in sorted(root.rglob("*.parquet")):
        try:
            import pandas as pd

            df = pd.read_parquet(parquet)
        except Exception:
            continue
        if not len(df):
            continue
        img_col = next((c for c in ("image", "img", "pixel_values") if c in df.columns), None)
        if not img_col:
            continue
        msg_col = "messages" if "messages" in df.columns else None
        txt_col = next((c for c in cols_text if c in df.columns), None)
        if not msg_col and not txt_col:
            continue
        out_dir = parquet.parent / "_png"
        for _, row in df.iterrows():
            text = _text_from_messages(row[msg_col]) if msg_col else str(row[txt_col]).strip()
            if not text:
                continue
            path = _dump_image(row[img_col], out_dir, cache)
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


def read_iam_voxel(root: Path, cache: dict):
    """Voxel51 IAM line set: samples.json pairs data/NNNNN.png with a transcript.

    English had 74 rows in total -- 53 of them train -- against 3000+ for every
    other graded language, so the language could not be learned however the
    decoder was tuned. This source adds ~5,600 labeled images of English
    handwriting. It is line-granular while the en test shard is paragraph form,
    which is a disclosed mismatch and not a like-for-like addition: it buys the
    model a chance to read English script, it does not buy it paragraph
    structure. Wherever an en result uses this source, that is stated.
    """
    meta = root / "samples.json"
    if not meta.exists():
        return
    payload = json.loads(meta.read_text(encoding="utf-8"))
    for item in payload.get("samples", []):
        rel = str(item.get("filepath", ""))
        text = str(item.get("assistant", "")).strip()
        if not rel or not text:
            continue
        img = root / rel
        if img.exists():
            yield {"image": str(img), "text": text}


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


def read_hi_zip(root: Path, cache: dict):
    """Hindi: annotation JSON rows paired with PNGs baked into per-chunk zips."""
    import zipfile

    ann_dir = root / "annotations"
    zip_dir = root / "images" / "hindi"
    if not ann_dir.is_dir() or not zip_dir.is_dir():
        return
    texts: dict[str, str] = {}
    for chunk in sorted(ann_dir.glob("*.json")):
        try:
            rows = json.loads(chunk.read_text(encoding="utf-8"))
        except Exception:
            continue
        for row in rows:
            rid = str(row.get("id", "")).strip()
            text = str(row.get("text", "")).strip()
            if rid and rid not in texts:
                texts[rid] = text
    if not texts:
        return

    out_dir = root / "_png"
    made = 0
    for zf_path in sorted(zip_dir.glob("*.zip")):
        try:
            with zipfile.ZipFile(zf_path) as zf:
                for name in zf.namelist():
                    rid = Path(name).stem
                    text = texts.get(rid)
                    if not text:
                        continue
                    out = out_dir / f"{rid}.png"
                    if not out.exists():
                        with zf.open(name) as src, out.open("wb") as dst:
                            shutil.copyfileobj(src, dst)
                    made += 1
                    yield {"image": str(out), "text": text}
                    if made >= ROW_CAP:
                        return
        except Exception:
            continue


def read_ur_tgz(root: Path, cache: dict):
    """Urdu: CSVs name img paths that live inside ouhdl_v1.0_core.tar.gz."""
    import tarfile

    tgz = next(iter(sorted(root.glob("*.tar.gz"))), None)
    if tgz is None:
        return
    wanted: dict[str, str] = {}
    for csv_path in sorted(root.glob("*.csv")):
        try:
            with csv_path.open(encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    img = str(row.get("img", "")).strip()
                    text = str(row.get("line", "")).strip()
                    if img and text:
                        wanted.setdefault(img, text)
        except Exception:
            continue
    if not wanted:
        return

    out_dir = root / "_png"
    made = 0
    with tarfile.open(tgz, "r:gz") as tf:
        for member in tf:
            if not member.isfile() or member.name not in wanted:
                continue
            src = tf.extractfile(member)
            if src is None:
                continue
            out = out_dir / Path(member.name).name
            if not out.exists():
                with out.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            made += 1
            yield {"image": str(out), "text": wanted[member.name]}
            if made >= ROW_CAP:
                return


def read_ur_tgz(root: Path, cache: dict):
    """Urdu: CSV rows name images that ship inside the ouhdl core tarball."""
    import tarfile

    tgz = root / "ouhdl_v1.0_core.tar.gz"
    if not tgz.exists():
        return
    pairs: list[tuple[str, str]] = []
    for csv_file in sorted(root.glob("*.csv")):
        with csv_file.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                img_key = next((k for k in ("img", "image", "path", "file") if k in row), None)
                txt_key = next((k for k in ("line", "text", "label") if k in row), None)
                if not img_key or not txt_key:
                    continue
                img = str(row[img_key]).strip()
                text = str(row[txt_key]).strip()
                if img and text:
                    pairs.append((img, text))
    if not pairs:
        return
    out_dir = root / "_png"
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tgz, "r:gz") as tf:
        for name, text in pairs:
            try:
                member = tf.getmember(name)
            except KeyError:
                continue
            path = out_dir / Path(name).name
            if not path.exists():
                with tf.extractfile(member) as src:
                    if src is None:
                        continue
                    path.write_bytes(src.read())
            yield {"image": str(path), "text": text}


def read_hi_zip(root: Path, cache: dict):
    """Hindi: annotation chunk i pairs id+text with entries in images/hindi/hindi_i.zip."""
    ann_dir = root / "annotations"
    zip_dir = root / "images" / "hindi"
    ann_files = sorted(ann_dir.glob("*.json")) if ann_dir.is_dir() else []
    if not ann_files:
        return
    zips = {p.name: p for p in zip_dir.glob("*.zip")} if zip_dir.is_dir() else {}
    out_dir = root / "_png"
    for i, ann in enumerate(ann_files):
        try:
            rows = json.loads(ann.read_text(encoding="utf-8"))
        except Exception:
            continue
        zip_path = zips.get(f"hindi_{i:04d}.zip")
        zf = None
        if zip_path:
            import zipfile

            try:
                zf = zipfile.ZipFile(zip_path)
            except Exception:
                zf = None
        for row in rows:
            rid = str(row.get("id", "")).strip()
            text = str(row.get("text", "")).strip()
            if not rid or not text or zf is None:
                continue
            path = out_dir / f"{rid}.png"
            if not path.exists():
                try:
                    data = zf.read(f"hindi/{rid}.png")
                except KeyError:
                    continue
                out_dir.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            yield {"image": str(path), "text": text}


READERS = {
    "parquet": read_parquet,
    "amr": read_amr,
    "tgz": read_tgz,
    "rxwords": read_rxwords,
    "ko": read_ko,
    "csv": read_csv,
    "auto": read_auto,
    "ur_tgz": read_ur_tgz,
    "hi_zip": read_hi_zip,
    "iam_voxel": read_iam_voxel,
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
                    if len(rows) >= ROW_CAP:
                        break
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
