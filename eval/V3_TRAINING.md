# RxBridge - v3 retrain: the EOS fix (2026-10-09)

## Root cause
collate() tokenised prompt + answer with NO end token. LFM2.5-VL closes
assistant turns with <|EOT|> (id 124900). The model never saw a stop
signal, so th/hi/ur scored exact lines then rambled (CER 3-6) and ru
never terminated. Fix: append processor.tokenizer.eos_token to the
answer before tokenising (commit 5e44158). The prompt stays an exact
prefix, so the label mask is unchanged. Covered by
tests/test_train_eos.py (CPU, stubbed processor, no download; 2/2 pass).

## Training run
arc-cluster GPU 1, launched 19:22 PDT, finished ~20:53 PDT.
Same recipe as v2 (only the EOS in labels differs):
    python3 train_lora.py --data-root htr --out lfm/adapter_v3 \
        --steps 1000 --per-lang-chars 40000 --seed 0
1000 optimiser steps x 8 accumulation = 8000 passes, ~0.62 s/pass.
Corpus unchanged: 5409 rows, ~40000 answer chars per language (same
shuffle and seed as v2, so row selection is identical).

## Loss curve (mean of last 10 passes, from lfm/v3.log)
pass 80: 1.62 | pass 2000: 2.30 | pass 4000: 0.58 | pass 7990: 1.36
Oscillates 0.6-2.3 throughout; no collapse, same shape as v2. The fix
targets the stop rule, not the loss level.

## VAL stop-rate (val rows only, 8 rows/lang th/hi/ur, 256-token
budget; stopped = EOS 124900 emitted before the budget)
| adapter | th | hi | ur | overall |
|---|---|---|---|---|
| adapter_v2 | 7/8 | 6/8 | 0/8 | 13/24 (54%) |
| adapter_v3 | 8/8 | 8/8 | 8/8 | 24/24 (100%) |

Urdu, which never stopped under v2, now stops on every row. Thai and
Hindi went from partial to complete termination. Measured with a probe
script (lfm/val_check.py, cluster-side, not committed); val.jsonl is
training data - test.jsonl was never opened.

## Inline val eval after training (val rows = training data, NOT
held-out)
de 0.0329 PASS, fr 0.0621 PASS, th 0.2554, vi 0.3392, en 0.4115,
hi 0.4530, ur 0.7962, ar 0.8266, fa 0.8293, ru 0.9417;
overall mean CER 0.5006, 2/10 languages. Reported as-is, no tuning.

## Not measured here
No test-set benchmark was run. The parallel TUI session owns
benchmark.py and will grade lora_v3 on the held-out test shards once
its budget-leak fix is committed. Adapter path:
~/rxbridge/lfm/adapter_v3 on arc-cluster (adapter_model.safetensors
+ tokenizer/processor files). Training log: ~/rxbridge/lfm/v3.log
