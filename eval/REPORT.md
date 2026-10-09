# RxBridge extraction model - held-out evaluation

Corpus: `data/ner_aug.json` - 12000 rows, split 11040 train /
960 held-out. Metric: seqeval entity-level F1.

| model | entity F1 |
|---|---|
| trained on real labels | **0.9367** |
| trained on shuffled labels (control) | 0.0000 |

The control replaces every gold tag with a draw from the corpus label
distribution while leaving the text untouched. A model that still scores on
shuffled labels has found a shortcut and the headline number is not evidence.
Here the control collapses relative to the real run, so the F1 reflects the
token-to-label mapping actually being learned.

**VERDICT: PASS** - shuffled control 0.0000 <= 0.35.
