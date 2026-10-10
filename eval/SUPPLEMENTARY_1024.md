# Supplementary benchmark: 1024-token generation budget (2026-10-09)

Supplementary, pre-declared fixed-budget measurement. The 128-token headline
benchmark is unchanged by this document; see `submission/EVIDENCE.md` for it.
This run exists to measure what the models do when the budget stops being the
binding constraint for the paragraph-form corpora.

Pre-declaration, fixed before either run started: one budget of 1024 new
tokens, identical for both arms; same held-out split (seed 0, 40 rows per
language, greedy decode); no value chosen by looking at test-shard results.

`benchmark.py` gained `--max-new-tokens` (int, default 128) for this
(commit 48e0e7a); default behaviour is unchanged.

## Which adapter

Two EOS-fixed LoRA retrains of LiquidAI/LFM2.5-VL-3B (commit 5e44158) ran
concurrently on the cluster. The pre-declared rule was to benchmark whichever
reached its final save first, judged by the log line `saved adapter to`, not
by directory existence — both `lfm/adapter_v3` and `lfm/adapter_v4` had
contained complete pre-training baseline saves since ~10:34 CST, which would
have been the wrong artifact to grade.

- `lfm/adapter_v3` (seed 0, 1000 optimiser steps, per-lang-chars 40000, GPU 1)
  saved its final adapter at 11:49:05 CST.
- `lfm/adapter_v4` (accum 8, lr 2e-4, GPU 0) saved at 11:49:32 CST.

`lfm/adapter_v3` finished first and is the adapter benchmarked here.

## Commands

Both run on GPU 2 (free; GPUs 0-1 were training), outputs to /tmp on the
cluster:

```
cd ~/rxbridge
CUDA_VISIBLE_DEVICES=2 python3 benchmark.py --model LiquidAI/LFM2.5-VL-3B \
  --max-new-tokens 1024 --per-lang 40 --device cuda \
  --out /tmp/bench_zs_1024.json

CUDA_VISIBLE_DEVICES=2 python3 benchmark.py --model LiquidAI/LFM2.5-VL-3B \
  --adapter lfm/adapter_v3 --max-new-tokens 1024 --per-lang 40 --device cuda \
  --out /tmp/bench_eos_1024.json
```

## Per-language results (mean CER, pass = CER <= 0.25)

| Lang | n | Zero-shot @1024 | Adapter @1024 | Delta |
|------|----|------------------|----------------|-------|
| en   | 14 | 0.3888 FAIL | 0.3559 FAIL | -0.0329 |
| fr   | 40 | 0.1153 **PASS** | 0.0768 **PASS** | -0.0385 |
| de   | 40 | 0.1244 **PASS** | 0.0612 **PASS** | -0.0632 |
| ru   | 40 | 0.8153 FAIL | 0.8071 FAIL | -0.0082 |
| th   | 40 | 0.9877 FAIL | 0.1867 **PASS** | -0.8010 |
| vi   | 40 | 0.2364 **PASS** | 0.3285 FAIL | +0.0921 |
| ar   | 40 | 0.7357 FAIL | 2.2318 FAIL | +1.4961 |
| ur   | 40 | 1.0284 FAIL | 0.8057 FAIL | -0.2227 |
| fa   | 40 | 0.6583 FAIL | 1.4803 FAIL | +0.8220 |
| hi   | 40 | 1.9467 FAIL | 0.4229 FAIL | -1.5238 |

Zero-shot: 3/10 passed (fr, de, vi), overall mean CER 0.7037.
EOS-fixed adapter: 3/10 passed (fr, de, th), overall mean CER 0.6757.
Both arms FAIL the all-languages bar; the harness verdict is unchanged.

Scrubbed per-row {lang, cer} data: `eval/SUPPLEMENTARY_ZS_1024.json`,
`eval/SUPPLEMENTARY_TRAINED_1024.json` (374 rows each, same scrub as
`eval/BENCHMARK_V2.json`). Raw benchmark output containing reference and
prediction text stays on the cluster at /tmp and is not committed.

Note on filenames: the adapter arm was first written as
`SUPPLEMENTARY_EOS_1024.json` and later re-derived from the same cluster output
under the name `SUPPLEMENTARY_TRAINED_1024.json`, which is the file referenced
by `submission/EVIDENCE.md`. The two were verified to contain identical
per-language CER data against the raw `/tmp/bench_eos_1024.json`, and only the
latter is tracked.

## Caveats

Even at 1024 tokens the paragraph-form corpora (ru, fa, ar, en: references
1000-3700 chars, i.e. roughly 250-900+ tokens) are still truncated, so those
languages cannot pass at either budget by construction — what 1024 changes is
how much of the handicap, not whether it exists. CER above 1.0 now also means
the hypothesis ran longer than the reference: with room to ramble, some rows
generate to the full budget, so these numbers measure runaway generation as
well as transcription error, and the EOS fix trades one failure mode for
another (th 0.9877 → 0.1867 with 80% exact match, but ar 0.7357 → 2.2318 as
the adapter's hypothesis length grows). The hi improvement (mean 1.9467 →
0.4229, exact match 0.375 → 0.575, median 0.2667 → 0.0000) is both effects at
once: the median reaching 0.0000 with exact match up to 0.575 is genuinely
better short-row transcription, and the remaining gap between a 0.0000 median
and a 0.4229 mean is a minority of long-output rows that still run away. en
has n=14 because the shard holds only 14 test rows. Both arms used the same
split, seed, budget, and greedy decode; the runs happened about an hour apart
on GPU 2 while GPUs 0-1 finished training, and nothing in this document was
chosen by looking at the test shards — the budget, the arm pairing, and the
first-final-save adapter rule were all fixed before the runs.
