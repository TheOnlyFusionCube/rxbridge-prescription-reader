"""Rebuild the corpus with the IAM English source, without moving the verifier.

Three things this guards, in order of importance:

1. The other nine languages' shards must come out byte-identical. build_corpus
   only touches what its sources yield, and IAM is wired to en alone, so any
   checksum change elsewhere means something read the corpus differently and
   the whole comparison is void.
2. en's test and val shards are restored from the pre-build backup. Left alone,
   write_shards would draw 200 test rows and 200 val rows from the now-5737-row
   pool and the en benchmark would silently start grading against IAM line
   crops instead of the 14 paragraph-form AMR rows every published English
   number was measured on.
3. Any row whose image appears in the restored test/val is then stripped from
   en's train shard. The rebuild shuffles with seed 0 and caps train at 3000 of
   5737 rows, so some of the 21 original test/val AMR images land in train by
   chance. Removing them is what makes the verifier actually held-out rather
   than held-out-on-paper.

Usage: python3 rebuild_en.py --repo /home/yoka-aiserver11/rxbridge
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

BACKUP = Path("/home/yoka-aiserver11/rxbridge/htr_backup")


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def image_paths(shard: Path) -> set:
    return {json.loads(l)["image"] for l in shard.read_text().splitlines() if l.strip()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="/home/yoka-aiserver11/rxbridge")
    args = ap.parse_args()
    repo = Path(args.repo)
    htr = repo / "htr"

    expected = {}
    for line in (BACKUP / "CHECKSUMS.txt").read_text().splitlines():
        if line.strip():
            digest, rel = line.split()
            expected[rel] = digest

    print("building corpus...", flush=True)
    rc = subprocess.run(
        [sys.executable, str(repo / "build_corpus.py"), "--repo", str(repo)]
    ).returncode
    if rc != 0:
        print("BUILD FAILED", file=sys.stderr)
        return rc

    print("\n--- 1. other languages unchanged ---", flush=True)
    regressions = []
    for rel, want in sorted(expected.items()):
        lang = rel.split("/")[0]
        if lang == "en":
            continue
        got = md5(htr / rel)
        mark = "ok" if got == want else "CHANGED"
        if got != want:
            regressions.append(rel)
        print(f"  {rel:24s} {mark}", flush=True)
    if regressions:
        print(f"\nABORT: {len(regressions)} shard(s) changed: {regressions}",
              file=sys.stderr)
        return 1

    print("\n--- 2. restore en test/val from backup ---", flush=True)
    for name in ("test", "val"):
        src = BACKUP / "htr" / "en" / f"{name}.jsonl"
        dst = htr / "en" / f"{name}.jsonl"
        shutil.copyfile(src, dst)
        n = sum(1 for l in dst.read_text().splitlines() if l.strip())
        print(f"  restored {name} ({md5(dst)[:8]}, {n} rows)", flush=True)

    print("\n--- 3. strip held-out images from en train ---", flush=True)
    held = image_paths(htr / "en" / "test.jsonl") | image_paths(htr / "en" / "val.jsonl")
    train_p = htr / "en" / "train.jsonl"
    rows = [json.loads(l) for l in train_p.read_text().splitlines() if l.strip()]
    kept = [r for r in rows if r["image"] not in held]
    dropped = len(rows) - len(kept)
    with train_p.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    overlap = image_paths(train_p) & held
    print(f"  train rows {len(rows)} -> {len(kept)} (dropped {dropped} held-out images)",
          flush=True)
    print(f"  remaining test/val images in train: {len(overlap)}", flush=True)
    if overlap:
        print(f"ABORT: {len(overlap)} held-out image(s) still in train", file=sys.stderr)
        return 1

    chars = sum(len(r.get("text", "")) for r in kept)
    print(f"\n  en train: {len(kept)} rows, {chars} answer chars", flush=True)
    print("\nDONE - corpus rebuilt, verifier unchanged", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
