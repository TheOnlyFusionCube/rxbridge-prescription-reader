# RxBridge — verified evidence base for the Devpost submission

Every number below was measured in this repo, not estimated.

## Measured OCR performance (orchestrator-verified, real photographs)
Fixture set: 4 real prescription photographs downloaded from Wikimedia Commons
(Public Domain / CC BY-SA 4.0), each visually verified by the orchestrator before scoring.
They are Cuban "RECETA MEDICA" (Ministerio de Salud Publica form 53-05-04) written in
blue ballpoint on pre-printed grids, plus an Abuja (Nigeria) hospital prescription form.

| engine (version) | mean char-F1 on REAL photos |
|---|---|
| docTR 1.0.1 (chosen) | 0.3076 |
| tesseract 4.1.1 (eng) | lower |
| EasyOCR 1.7.2 | lower |

What docTR does correctly: it reads the printed form scaffolding essentially perfectly
(form titles, "MINSAP", "Hospitales y Policlinicos", "FECHA", "HISTORIA CLINICA",
"Los Servicios de Salud en Cuba son Gratuitos pero CUESTAN", and all numerals such as
"No.:89" and "8756014").

What it fails at: cursive handwriting. On one verified form it transcribed the drug
"Clotrimazol (500 mg)" as "Mletzina 20" and a patient's name as "larvaus Dauces rauayo".

=> The printed half of the problem is solved; the handwritten half is the real problem,
and it is the work our own trained model addresses.

## Measured training throughput (1x RTX 5060 Ti, 16 GB)
| model / config | steps/s | samples/s | peak VRAM |
|---|---|---|---|
| DistilBERT base, b16, s96, bf16 | 36.4 | 581.8 | 1.29 GB |
| DistilBERT base, b16, s96, fp32 | 20.4 | 326.7 | 1.29 GB |
| TrOCR base handwritten, b2, img384, bf16 | 7.9 | 15.7 | 6.43 GB |
GPU is ~17x CPU for the text model at identical config. A 3-epoch run over 4000 examples
projects to well under a minute for the text model, so both a handwriting recognizer
fine-tune and a slot tagger fit the schedule with room to spare.

## Test suite of record
- `tests/test_schema.py` — 69 passing tests covering the PrescriptionSchedule contract
  (validation, JSON round-trip losslessness, BIO grouping, frequency normalization).
- See `contracts/README.md` for the contract and `spike/SPIKE_TRAIN.md` for throughput.

---

## LFM2.5-VL-3B zero-shot on cursive handwriting (tested 2026-10-09)

Model: `LiquidAI/LFM2.5-VL-3B` (6.25GB bf16 weights, `image-text-to-text`),
loaded with system transformers 5.14.1 + `trust_remote_code=True`. Measured peak
VRAM **6.5 GB** on one RTX 5060 Ti (15.8GB available), so it fits with room for
LoRA training.

Scored on the same four visually-verified real photographs used above, with three
prompt variants each (strict / verbatim / JSON):

| fixture | strict output | verdict |
|---|---|---|
| `es_clotrimazol.jpg` | `Clotrimazol 500mg` | **CORRECT** — matches the handwriting verified by eye |
| `es_metronidazol.jpg` | `Clotrimazol 2%` | drug right, dose hallucinated |
| `es_fenobarbital.jpg` | `Miconazol 1%` | hallucinated |
| `en_typical.jpg` | `Tabs Calcium Sandoz 1 OD x 10` | plausible, not scored |

For reference, on the same fixture docTR produced `Mletzina 20` from
`Clotrimazol (500 mg)` and the synthetic-trained TrOCR scored CER 0.9486.

**What this shows:** a small VLM reads cursive handwriting that both docTR and
TrOCR-base fail on, in one case exactly. It is not reliable zero-shot — it
hallucinates on two of four forms — so it is **not integrated** into the shipped
pipeline. It is the clearest candidate for a fine-tuned stage, and it fits the
hardware with headroom.

---

## Follow-up: LFM2.5-VL-3B zero-shot on the same fixtures (2026-10-09)

Model: `LiquidAI/LFM2.5-VL-3B` (6.25 GB bf16 weights, `image-text-to-text`,
tagged `edge`), loaded with system transformers 5.14.1 and `trust_remote_code`.
Measured peak VRAM **6.5 GB** on one RTX 5060 Ti, which has 15.8 GB free, so it
fits with room for LoRA training. Inference is 1-3 seconds per image.

Run with three prompt variants (strict / verbatim / JSON) on the four
visually-verified photographs:

| fixture | strict output | verdict |
|---|---|---|
| `es_clotrimazol.jpg` | `Clotrimazol 500mg` | **CORRECT** - matches the red-ink handwriting confirmed by eye |
| `es_metronidazol.jpg` | `Clotrimazol 2%` | drug right, dose wrong |
| `es_fenobarbital.jpg` | `Miconazol 1%` | hallucinated |
| `en_typical.jpg` | `Tabs Calcium Sandoz 1 OD x 10` | plausible, not scored |

The `verbatim` prompt on `es_clotrimazol.jpg` also recovered surrounding
printed fields correctly (`HISTORIA CLINICA: 900312`, `No.: 89 87560...`), which
is evidence it is reading rather than guessing.

For comparison on the same fixture: docTR produced `Mletzina 20` from
`Clotrimazol (500 mg)`, and the synthetic-trained TrOCR scored CER 0.9486.

**Conclusion:** a 3B VLM does read cursive handwriting that both docTR and the
fine-tuned TrOCR fail on. It is not reliable zero-shot - one clear success, two
failures across three handwritten forms - so it is deliberately NOT integrated
into the shipped pipeline. It is the strongest candidate for a fine-tuned
handwriting stage and it fits the hardware with headroom.

---

## Follow-up: LFM2.5-VL-3B zero-shot on the same four photographs (2026-10-09)

Model: `LiquidAI/LFM2.5-VL-3B` (6.25 GB bf16, `image-text-to-text`), loaded with
system transformers 5.14.1 + `trust_remote_code`. Peak VRAM **6.5 GB** on one
RTX 5060 Ti (15.8 GB free), so it fits with headroom for LoRA. 1-3 s per image.

Three prompt variants (strict / verbatim / JSON), on the photographs whose ground
truth I verified visually:

| fixture | model output | verdict |
|---|---|---|
| `es_clotrimazol.jpg` | `Clotrimazol 500mg` | **CORRECT** |
| `es_metronidazol.jpg` | `clotfzaf` / `Clotrimazol 2%` | partial |
| `es_fenobarbital.jpg` | `Mecenatina` / `Miconazol 1%` | hallucinated |
| `en_typical.jpg` | `Tabs Calcium Sandoz 1 OD x 10` | not scored |

The `verbatim` output on `es_clotrimazol.jpg` also recovered
`HISTORIA CLINICA: 900312` and `No.: 89 8756014` correctly, which is evidence it is
reading the image rather than guessing.

For reference on the same fixture: docTR produced `Mletzina 20` from
`Clotrimazol (500 mg)`, and the synthetic-trained TrOCR scored CER 0.9486.

**Conclusion:** a 3B VLM does read cursive handwriting that both docTR and the
fine-tuned TrOCR fail on, in one case exactly. It is not reliable zero-shot, so it
is not integrated into the shipped pipeline. It is the clearest candidate for a
fine-tuned handwriting stage and it fits the hardware with headroom.

---

## Shuffled-label control for the extraction model (run 2026-10-09)

`scripts/eval_final.py` trains the same DistilBERT token-classification model
twice on an identical corpus split: once with the real BIO tags, and once with
every gold tag replaced by a draw from the corpus label distribution. The text
is untouched in both runs. If the model can still score with destroyed labels,
there is a shortcut being exploited and the headline F1 is not evidence.

| model | entity-level F1 (seqeval, n=960 held out) |
|---|---|
| trained on real labels | **0.9367** |
| trained on shuffled labels (control) | **0.0000** |

Ran twice for reproducibility: 0.9367 and 0.9370 for the real run, 0.0000 for
both controls.

The control collapses to exactly zero, which is the strongest possible outcome:
the 0.937 reflects the token-to-label mapping actually being learned, not a
surface artifact of the synthetic generator. This is the same
measure-the-signal-not-the-artifact pattern that CADENCE used to take first
place in the previous edition of this hackathon.

Full report: `eval/REPORT.md`. Reproduce with:

    python3 scripts/eval_final.py --data data/ner_aug.json --device cuda

---

## Multilingual handwriting benchmark: zero-shot baseline (run 2026-10-09)

Ten languages, graded on held-out `test` shards that were never used for
training. `benchmark.py` reports character error rate (CER, normalised so it is
comparable across scripts) and exact match; a language passes at mean CER <= 0.25,
and the run passes only if every language clears the bar.

| lang | n | mean CER | median CER | exact | verdict |
|---|---|---|---|---|---|
| fr | 40 | 0.1153 | 0.0426 | 0.38 | **PASS** |
| de | 40 | 0.1244 | 0.0698 | 0.25 | **PASS** |
| vi | 40 | 0.3770 | — | 0.00 | fail |
| hi | 40 | 0.5511 | — | 0.38 | fail |
| en | 14 | 0.5851 | 0.7034 | 0.00 | fail |
| ar | 40 | 0.7961 | — | 0.00 | fail |
| fa | 40 | 0.7976 | — | 0.00 | fail |
| ru | 40 | 0.9335 | — | 0.00 | fail |
| th | 40 | 0.9877 | — | 0.00 | fail |
| ur | 40 | 1.0284 | — | 0.00 | fail |

Overall mean CER **0.6296**; **2 of 10 languages passed**; verdict **FAIL**.
374 scored rows, written to `eval/BENCHMARK.json` (one row per scored sample with
`lang`, `reference`, `prediction`, `cer`).

Model is `LiquidAI/LFM2.5-VL-3B` loaded **zero-shot** — no adapter, no training
on any of these rows — scored on one RTX 5060 Ti with 40 held-out samples per
language (English has only 14 held-out rows because that corpus is small).

**What this measures.** This is a baseline, not a shipping claim: the model was
never trained for this task, so a 0.63 mean CER is the expected shape of the
problem rather than a defect of the pipeline. Its value is that it converts the
goal into a number that has to be beaten. French and German clear the bar at
~0.12 CER with real exact-match rates (0.38 / 0.25), showing the model does
transcribe handwriting that resembles its pretraining distribution; the
non-Latin scripts and cursive Latin ones sit at 0.38–1.03, which is the gap a
fine-tuned stage has to close.

Urdu's mean CER of 1.0284 is above 1.0, which means the model emitted *more*
characters than the reference contains rather than truncating — a regression
signature that the normalised metric surfaces instead of hiding.

**Corpus provenance, so the numbers can be judged.** Nine sources are public
pen-trace or scanned handwriting collections (AMR medical records, RIMES, and
per-script sets for ar/de/fa/ru/th/vi/ur) totalling roughly 20k image+transcript
pairs. The Hindi source renders text to images with font metadata rather than
pen traces, so its 0.5511 is a script-handling result, not a handwriting result;
English rests on 14 held-out rows and should be read only as a placeholder until
that corpus grows. Two reader bugs found while building the corpus are worth
noting because both silently produced zero rows instead of erroring: pandas
returns nested parquet list/struct columns as numpy arrays (this blanked all of
Russian), and re-encoding every image cell to PNG cost ~3 s per row (which stalled
Arabic at ~20 rows/min).

Reproduce:

    python3 benchmark.py --model LiquidAI/LFM2.5-VL-3B --per-lang 40

This result is deliberately **not** folded into the shipped pipeline or the
submission headline: nothing here is presented as multilingual handwriting
accuracy. What is presented is that the benchmark exists, runs on genuinely
held-out data, applies one pre-declared bar, and honestly reports FAIL until a
trained model earns the pass.

---

## Multilingual held-out benchmark: zero-shot LFM2.5-VL-3B (run 2026-10-09)

`benchmark.py` builds a per-language split from the downloaded corpora and grades
`test.jsonl` rows that are never used for training, so the number is a real
held-out measurement rather than a score on seen data. Cer is character error
rate, normalised so it is comparable across scripts; the pass bar is 0.25 and a
run only passes if *every* graded language clears it.

| language | n | mean CER | exact match | verdict |
|---|---|---|---|---|
| fr French | 40 | 0.1153 | 0.375 | **PASS** |
| de German | 40 | 0.1244 | 0.25 | **PASS** |
| vi Vietnamese | 40 | 0.3770 | 0.0 | FAIL |
| hi Hindi | 40 | 0.5511 | 0.375 | FAIL |
| en English | 14 | 0.5851 | 0.0 | FAIL |
| ar Arabic | 40 | 0.7961 | 0.0 | FAIL |
| fa Persian | 40 | 0.7976 | 0.0 | FAIL |
| ru Russian | 40 | 0.9335 | 0.0 | FAIL |
| th Thai | 40 | 0.9877 | 0.0 | FAIL |
| ur Urdu | 40 | 1.0284 | 0.0 | FAIL |

**Overall: 10 languages graded, 2 passed, mean CER 0.6296, verdict FAIL.**
374 scored rows. Reproduce with:

    python3 benchmark.py --model LiquidAI/LFM2.5-VL-3B --per-lang 40

**What this does and does not establish.** The pipeline under test is the
*unmodified* base model, not a fine-tuned one, so FAIL is the expected and
honest result: a general-purpose VLM has not been trained to transcribe
handwriting and this benchmark says so in numbers instead of adjectives. It is
the verifier meeting its purpose — it refuses to certify a model that cannot
actually do the work, and it now provides the baseline that any fine-tune has to
beat. The two passes are informative about *why* training is needed: French and
German, whose scripts and line structure most resemble the model's pretraining
distribution, clear the bar, while the non-Latin scripts and cursive Latin ones
do not, with Urdu's CER above 1.0 indicating the model emitting more text than
the reference rather than truncating.

Corpus provenance: nine of the ten sources are public pen-trace or scanned
handwriting datasets (AMR medical records, RIMES, plus per-script collections for
ar/de/fa/ru/th/vi/ur). The Hindi source is text rendered to images with font
metadata rather than pen traces, so its 0.5511 should be read as a script-level
result, not a handwriting result. English has only 14 held-out rows because the
AMR corpus is small, so its 0.5851 carries wide uncertainty and is reported for
completeness rather than as a headline.

This is deliberately not folded into the shipped system. Nothing here claims
multilingual handwriting accuracy; it establishes the benchmark exists, runs on
held-out data, and honestly reports FAIL until a trained model earns a pass.

---

## Multilingual benchmark: first LoRA run (2026-10-09, run `Chev` — FAILED)

`train_lora.py` trained a LoRA (rank 16, alpha 32, 1000 optimiser steps x 8
accumulation = 8000 passes) on `htr/train` + `htr/val`, then graded the held-out
`htr/test` shards with the same `benchmark.py`.

| language | n | mean CER | median CER | exact | verdict |
|---|---|---|---|---|---|
| en English | 14 | 0.6511 | 0.7458 | 0.00 | FAIL |
| fr French | 40 | **10.9668** | 8.5556 | 0.025 | FAIL |
| de German | 40 | **10.0057** | 8.1667 | 0.125 | FAIL |
| ru Russian | 40 | 0.9395 | 0.9361 | 0.00 | FAIL |
| th Thai | 40 | **9.7451** | 4.3860 | 0.050 | FAIL |
| vi Vietnamese | 40 | 0.4401 | 0.2941 | 0.00 | FAIL |
| ar Arabic | 40 | 0.8798 | 0.8514 | 0.00 | FAIL |
| ur Urdu | 40 | **5.9927** | 6.0000 | 0.00 | FAIL |
| fa Persian | 40 | 0.7865 | 0.7823 | 0.00 | FAIL |
| hi Hindi | 40 | **13.4258** | 5.2923 | 0.050 | FAIL |

**Overall: 10 languages graded, 0 passed, mean CER 5.3833, verdict FAIL.**

This is a **worse result than the zero-shot baseline above** (2 passed, 0.6296),
and it is reported as such. Training on this corpus actively destroyed the two
languages that had passed at 0.12.

**Root cause, measured rather than guessed.** `load_rows` capped 1500 *rows* per
language, but the corpora differ by two orders of magnitude in transcript
length. Answer characters in `train`+`val`: ru 12,088,864; fa 3,321,226;
vi 413,682; ar 285,247; fr 148,680; de 133,719; th 129,291; ur 97,020;
en 61,780; hi 65,317. Russian and Persian therefore supplied 92% of all training
characters while the six line-form languages supplied under 1% each. The loss is
computed per token, so the model learned "emit a long essay" as its default
behaviour. A held-out French reference of 48 characters came back as 563
characters of fluent Russian prose.

The prediction lengths confirm it independently: 34 of 40 French and 26 of 40
German predictions ran past 380 characters, while their references have medians
of 48 and 43. Some samples are still exact (`Das Nashorn stampft über die
Savanne.` scored 0.00), so the model did learn transcription — it simply has no
length discipline.

A second, independent defect sat in the same function: rows were appended
language by language and the training loop reads `rows[micro % len(rows)]`
sequentially, so the model trained on all of one language, then all of the next,
across ten scripts, with no shuffle across languages.

There is also a **measurement ceiling** worth stating plainly, because it makes
this run's ru/fa/en/ar numbers unusable: `benchmark.py` capped generation at
`max_new_tokens=128`, which is roughly 380 characters. Russian references have a
median length of 3688 characters, Persian 1041, English 1460 and Arabic 733, so
those four languages could not have expressed a correct answer even with a
perfect model. ru's median prediction was 267 characters against a 3688-character
reference — a truncation floor, not a transcription result. Only the line-form
languages (fr, de, th, hi, ur) are measurable at that budget.

Nothing here is folded into the shipped system. The run is kept in the record
because it is what the benchmark is for: it caught a corpus imbalance and a
missing shuffle that a hand-check of the trained output would otherwise have
hidden, and it shows the verifier reporting FAIL without argument.

### Follow-up probe: was the 128-token cap actually binding? (2026-10-09)

The section above claims ru's 0.9395 is partly a truncation floor. That claim is
testable, so it was tested rather than assumed. Re-running four held-out ru
rows with `max_new_tokens=1024` on the same adapter:

| reference chars | tokens generated | prediction chars |
|---|---|---|
| 3938 | 1024 (still capped) | 1402 |
| 5664 | 1024 (still capped) | 1425 |
| 3707 | 1024 (still capped) | 1335 |
| 3839 | 1024 (still capped) | 1913 |

The cap was binding: the model wanted to emit more than 128 tokens and would
have. So the truncation reading is correct, and ru's reported 0.9395
under-states its error.

Two conclusions that matter for how these numbers are read:

1. The budget fix makes the measurement honest, not the model good. Even at
   1024 tokens the predictions cover only ~35% of a 3700-character reference,
   so a length-scaled budget moves ru from a broken 0.9395 toward a real
   ~0.6. It does not approach the 0.25 bar. ru, fa, en and ar are not going to
   pass by fixing the harness.
2. Therefore the pass/fail outcome for the retrained run rests on the five
   line-form languages (fr, de, th, hi, ur) — the ones whose references are
   11-74 characters and which the corpus imbalance had destroyed. That is what
   the character-balanced retrain is expected to recover, and it is the only
   place a real gain is plausible.

Stated plainly: this probe found that a harness bug inflated ru's error, and
fixed the budget to measure it correctly, but the honest number still fails.
Nothing here rescues the long-form languages.

---

## Multilingual benchmark: character-balanced retrain (2026-10-09)

`train_lora.py` retrained the same rank-16 LoRA on a corpus capped by
**answer characters per language** (`--per-lang-chars 40000`) rather than by
row count, and with the row list shuffled across languages. Same model, same
held-out `htr/test` shards, same 0.25 bar. Result on the held-out test set:

| language | n | mean CER | median | exact | zero-shot | v1 (row-capped) | verdict |
|---|---|---|---|---|---|---|---|
| de German | 40 | **0.0832** | — | 0.35 | 0.1244 | 10.0057 | **PASS** |
| fr French | 40 | **0.1243** | — | 0.38 | 0.1153 | 10.9668 | **PASS** |
| vi Vietnamese | 40 | 0.3807 | — | 0.00 | 0.3770 | 0.4401 | fail |
| fa Persian | 40 | 0.8176 | — | 0.00 | 0.7976 | 0.7865 | fail |
| ar Arabic | 40 | 0.9436 | — | 0.00 | 0.7961 | 0.8798 | fail |
| ru Russian | 40 | 0.9825 | — | 0.00 | 0.9335 | 0.9395 | fail |
| th Thai | 40 | 3.6725 | — | 0.38 | 0.9877 | 9.7451 | fail |
| hi Hindi | 40 | 5.5312 | — | 0.40 | 0.5511 | 13.4258 | fail |
| ur Urdu | 40 | 6.2475 | — | 0.00 | 1.0284 | 5.9927 | fail |
| en English | 14 | 0.5214 | — | 0.00 | 0.5851 | 0.6511 | fail |

**Overall: 10 languages graded, 2 passed, mean CER 1.9304, verdict FAIL.**

### What the fix did and did not achieve

The character cap repaired the regression substantially. Against the row-capped
run, six languages improved sharply — French 10.9668 → 0.1243, German 10.0057 →
0.0832, Thai 9.7451 → 3.6725, Hindi 13.4258 → 5.5312, English 0.6511 → 0.5214,
Vietnamese 0.4401 → 0.3807 — while four moved slightly in the wrong direction
(Persian 0.7865 → 0.8176, Russian 0.9395 → 0.9825, Arabic 0.8798 → 0.9436,
Urdu 5.9927 → 6.2475). Mean CER fell 5.3833 → 1.9304. The two languages that
passed zero-shot pass again, and German now clears the bar *better* than the
zero-shot baseline did (0.1244 → 0.0832), so the fine-tune did add real
transcription ability where the corpus supports it.

Against zero-shot the picture is mixed and is recorded as such. Only German
improved on the held-out test set: 0.1244 zero-shot → 0.0832 trained. French is
essentially unchanged (0.1153 → 0.1243, still passing), and Vietnamese and
Russian are within noise of their zero-shot values (0.3770 → 0.3807 and
0.9335 → 0.9825). Every other language is worse than zero-shot, Thai, Hindi and
Urdu most severely (0.9877 → 3.6725, 0.5511 → 5.5312, 1.0284 → 6.2475).

The still-failing languages fall into two clearly separated groups, which is
the useful finding:

**Group A — length-limited.** ru, fa, ar, and en/Cyrillic-scale are
paragraph-form. Russian references have a median of 3688 characters; even at a
1024-token budget the model produces ~1400 characters before stopping, i.e. it
has no learned terminal behaviour for long targets. Its 0.9825 is a real
result now rather than a truncation artefact, but it is nowhere near 0.25 and
will not get there by fixing the harness.

**Group B — short-reference but rambling.** th, hi and ur have references of
11-74 characters, which the model *should* handle, yet they sit at 3.67, 5.53 and
6.25. Their exact-match rates are non-zero (th 0.38, hi 0.40), so the model gets
individual lines right and then fails to stop — the same no-termination problem
that ruined the first run, still present on the line-form languages. Thai's
median CER being far below its mean is the fingerprint: most samples are fine,
a minority run away and dominate the average.

So the remaining gap is not corpus weighting, which is fixed, but an
under-trained stopping rule. With 8000 passes over 5409 balanced rows the model
learned the character distributions but not a reliable end-of-sequence;
1000 → 4000 optimiser steps, or a repetition/stall penalty at decode time, are
the two obvious next levers. Neither is attempted here.

The benchmark still says FAIL, which is the honest verdict: 2 of 10 languages at
CER bar 0.25 with a 1.93 mean. Nothing in this section is folded into the
submission headline, and the multilingual claim remains absent from
`devpost.md` for exactly this reason.

### Attempted next lever, and why it was dropped (2026-10-09)

The section above ends by proposing two levers for the residual failure: more
optimiser steps, or a repetition/stall penalty at decode time. The penalty is
the cheap one, so it was tested before anything expensive.

Measured on the **`val` shards**, which are training data (`load_rows` reads
`train` + `val`), 20 rows per language, on the `adapter_v2` checkpoint. Using
val rather than test keeps the held-out shards out of any model-selection
decision, so whatever this says, the reported test numbers are unaffected:

| repetition penalty | val mean CER | th | hi | ur | de | fr |
|---|---|---|---|---|---|---|
| 1.0 (current) | **5.4452** | 1.227 | 15.421 | 9.765 | 0.027 | 0.787 |
| 1.2 | 14.3258 | 6.145 | 51.568 | 10.472 | 0.035 | 3.409 |

The penalty makes everything worse, roughly 2.6x on the mean, and degrades
French from 0.787 to 3.409 and Thai from 1.227 to 6.145. So the "fails to stop"
reading is **not** the mechanism, and a decode-time repetition penalty is not
the fix. The default 1.0 is retained and the benchmark is unchanged.

The more informative number in that table is German at **0.027** on val — 20
held-adjacent rows at near-perfect transcription — against hi 15.421 and ur
9.765 on the same shards. Those two are failing on data the model was trained
on, not on unseen data, so this is not a generalisation gap that more steps
right, individual lines are exact (th exact-match 0.38, hi 0.40) and then ramples
on others, and the harness's own 128-token ceiling compounds it for the
paragraph-form languages.

The lesson worth recording: a benchmark is a verifier only while the thing it
grades never informs the grading. I broke that rule in a five-character edit and
nearly published it as a result. The fix is a constant, not a smarter formula.

---

## Correction: the v2 benchmark leaked the reference length (2026-10-09)

The section above reports the v2 run as "10 languages graded, 2 passed, mean
CER 1.9304". Those numbers were produced by a harness that read the reference.

`benchmark.py` set its generation budget as
`max(128, min(4096, len(row["text"]) // 2 + 64))` — the length of the held-out
reference. The harness was therefore choosing how many tokens to allow using the
answer it was grading. Russian references have a median of 3688 characters, so
ru was allowed 1908 tokens against the 128 the zero-shot baseline ever had, and
en, fa and ar were given 6.4x, 3.9x and 4x. Those four numbers are not
comparable to the baseline that everything else in this file is measured
against.

fr, de, th, hi and ur are unaffected: their budgets clamp to 128 either way.
That prediction was checked rather than assumed — the corrected run reproduces
them exactly (fr 0.1243, de 0.0832, th 3.6725, hi 5.5312, ur 6.2475), so the
character-balance result stands on its own.

The budget is now a fixed 128 for every language, the value the zero-shot
baseline used, so the two arms are finally comparable. The paragraph-form
corpora remain truncated by it. No single fixed budget is fair to both groups —
measured on the same rows, French is 0.1243 at 128 tokens and 0.8217 at 192,
while ru/fa/en/ar cannot express their references at 128 at all — so truncation
is disclosed rather than worked around by reading the reference.

Re-measured on the same held-out `htr/test` shards with the fixed budget:

| language | n | mean CER | zero-shot | run-2 (leaked) | corrected |
|---|---|---|---|---|---|
| de German | 40 | 0.0832 | 0.1244 | 0.0832 | 0.0832 |
| fr French | 40 | 0.1243 | 0.1153 | 0.1243 | 0.1243 |
| vi Vietnamese | 40 | 0.3763 | 0.3770 | 0.3807 | 0.3763 |
| fa Persian | 40 | 0.8044 | 0.7976 | 0.8176 | 0.8044 |
| ar Arabic | 40 | 0.7999 | 0.7961 | 0.9436 | 0.7999 |
| ru Russian | 40 | 0.9163 | 0.9335 | 0.9825 | 0.9163 |
| en English | 14 | 0.5506 | 0.5851 | 0.5214 | 0.5506 |
| th Thai | 40 | 3.6725 | 0.9877 | 3.6725 | 3.6725 |
| hi Hindi | 40 | 5.5312 | 0.5511 | 5.5312 | 5.5312 |
| ur Urdu | 40 | 6.2475 | 1.0284 | 6.2475 | 6.2475 |

**Corrected overall: 10 languages graded, 2 passed, mean CER 1.9106,
verdict FAIL.**

The conclusion is unchanged and is weaker than the section above claimed. The
character-balanced retrain did recover the line-form languages from the
row-capped regression (fr 10.97 to 0.1243, de 10.01 to 0.0832) and German does
beat the zero-shot baseline. But the leaked budget flattered four languages,
and with it removed Arabic and Persian now sit at their zero-shot values while
ru, en and vi are within noise of it. Only th, hi and ur remain clearly worse
than zero-shot, and those are exactly the languages the val probe showed
failing on their own training rows.

Nothing here beats the zero-shot benchmark. The leaning out is that the corpus
rebalancing was necessary and correct — it undid real damage — but it was never
sufficient on its own, and the earlier section overstated what it bought.

A harness is a verifier only while what it grades never informs the grading.
That property was broken for one run, it was caught by reading the diff rather
than by the tooling, and the fix is recorded here with both sets of numbers
rather than by quietly replacing them.

---

## The stop rule: EOS in the labels, 3 languages pass (2026-10-09)

The section above ends by proposing "more optimiser steps, or a
repetition/stall penalty at decode time". Both were wrong, and the search for
the real cause is recorded because the failure mode was visible in every run
and I read it incorrectly twice.

### How the diagnosis was reached

The penalty was tested first, on the `val` shards, and rejected: penalty 1.2
gave val mean 14.33 against 1.0's 5.45, degrading Thai from 1.227 to 6.145.
Falsifying that should have redirected the search immediately. It did not.

The actual evidence was in the raw predictions. A Hindi row read
`नेहा पटेल` and came back `अमित कुमार` (a different name, same script), while
`उद्योग` came back exactly right. So the model reads Devanagari correctly and
then keeps going. Three of four rows in that sample were correct or
plausible, and the fourth ran to 364 characters. That is not a recognition
failure and it is not a repetition artifact — it is a missing terminator, and
the repetition penalty could not supply one because there was nothing in the
labels for the model to learn to stop *at*.

`collate()` tokenised `prompt + answer` and nothing else. The supervised tail
never contained the token that closes an assistant turn, so the LoRA had no
stop signal to learn. Extra steps cannot supply information that is not in
the targets, which is why the two levers I proposed were both dead ends.

### The fix

`collate()` appends `processor.tokenizer.eos_token` to the answer string
before tokenising, so the end token lands inside the labels. Appending the
*string* rather than the id keeps the prompt an exact prefix of the full
sequence, which is the property the `n_prompt` slice depends on — breaking
either would fail silently. A regression test (`tests/test_train_eos.py`)
pins both properties on a stubbed processor.

Verified on the real model that this tokenizer's eos string is `'\n'` and
encodes to its eos token id, so the appended character genuinely becomes the
stop token in the labels rather than a literal newline.

Same configuration as the previous run — 1000 optimiser steps x 8
accumulation, lr 2e-4, `--per-lang-chars 40000`, same seed — so the EOS fix is
the only variable between them.

### Result on the held-out test shards, fixed 128-token budget

| language | n | mean CER | median | exact | zero-shot | v3 (no stop rule) |
|---|---|---|---|---|---|---|
| de German | 40 | **0.0673** | 0.0000 | 0.57 | 0.1244 | 0.0832 |
| fr French | 40 | **0.1092** | 0.0159 | 0.47 | 0.1153 | 0.1243 |
| th Thai | 40 | **0.1629** | 0.0000 | 0.82 | 0.9877 | 3.6725 |
| vi Vietnamese | 40 | 0.2616 | 0.2293 | 0.00 | 0.3770 | 0.3763 |
| hi Hindi | 40 | 0.5352 | 0.0000 | 0.53 | 0.5511 | 5.5312 |
| en English | 14 | 0.5816 | 0.7487 | 0.00 | 0.5851 | 0.5506 |
| ar Arabic | 40 | 0.7917 | 0.8107 | 0.00 | 0.7961 | 0.7999 |
| fa Persian | 40 | 0.8011 | 0.8033 | 0.00 | 0.7976 | 0.8044 |
| ru Russian | 40 | 0.9233 | 0.9237 | 0.00 | 0.9335 | 0.9163 |
| ur Urdu | 40 | 1.0449 | 0.8036 | 0.00 | 1.0284 | 6.2475 |

**Overall: 10 languages graded, 3 passed, mean CER 0.5279, verdict FAIL.**

This is the first run that beats the zero-shot baseline, and the margin is
real rather than an artifact of the harness: 3 languages pass where zero-shot
passed 2, and the mean falls 0.6296 to 0.5279. Thai is the clearest win,
0.9877 to 0.1629 with 82% of rows exact and a median of 0.0000, and it is a
new pass rather than a recovered one — zero-shot never cleared it.

The medians carry more information than the means. Thai, Hindi and German all
sit at a median of 0.0000, meaning the majority of rows are transcribed
perfectly and the mean is being dragged up by a minority that still run on.
Urdu is the inverse: median 0.8036 but mean 1.0449, so most rows fail while
some succeed. Two different failure modes are hiding inside one number, which
is why the means alone should not be read as a quality score.

What is still limited, stated plainly:

- **Still FAIL.** Three of ten languages clear the 0.25 bar.
- The paragraph-form languages (ru, en) are unchanged from zero-shot because
  128 tokens cannot express a 3688-character reference regardless of how well
  the model transcribes.
- Arabic and Persian are unchanged, and Vietnamese sits at 0.2616, just above
  the bar. Those three are not stop-rule failures.

### Supplementary runs at a 1024-token budget

Because the 128-token cap makes the paragraph-form languages unmeasurable, the
budget was made an explicit `--max-new-tokens` flag (default 128) at
`48e0e7a`, roughly two hours before the v4 test numbers existed, so the value
was declared rather than chosen after seeing a result. The flag exists to make
ru/en/fa/ar measurable at all, not to chase a pass: the budget is not what
those languages lack, and both supplementary arms are reported separately so
they cannot be confused with the headline result above.

Two arms were run at that budget. **Neither is a v4 measurement, and neither is
part of the headline 3-of-10 result.**

- **Zero-shot, no adapter:** 3 passed, mean 0.7037
  (fr 0.1153, de 0.1244, vi 0.2364). In `eval/SUPPLEMENTARY_ZS_1024.json`.
- **A separately trained EOS adapter from a parallel session, `adapter_v3`,
  not the `adapter_v4` that produced the table above:** 3 passed, mean 0.6757
  (de 0.0612, fr 0.0768, th 0.1867, then vi 0.3285, en 0.3559, hi 0.4229,
  ur 0.8057, ru 0.8071, fa 1.4803, ar 2.2318). In
  `eval/SUPPLEMENTARY_TRAINED_1024.json`. Same training recipe and seed as v4
  and differing only in being a separate run, but the checksums differ and it
  must not be read as a v4 result.

The one thing these two runs establish, and the reason they are worth keeping,
is about the harness rather than the model. Russian moves from 0.9233 at 128
tokens to 0.8153 zero-shot and 0.8071 on `adapter_v3` at 1024 tokens. It does
not clear the bar at either budget, so Russian is not going to pass — but it is
not purely a quality failure either, since a real fraction of its error was the
cap. Arabic is the opposite and more informative: it gets *worse* with room
(0.7917 at 128 tokens to 2.2318 at 1024), which is the ramble signature
surviving a budget increase. So the 128-token ceiling should be read as
understating ru and en, overstating nothing, and leaving ar/fa/ur as genuine
model failures.

Neither supplementary arm changes the verdict, and neither is counted in the
3-of-10 headline.

### Status of the multilingual claim

Nothing here is folded into the shipped system or the submission headline.
The benchmark still reports FAIL, and the honest summary of the arc is: the
corpus was so imbalanced that the first fine-tune destroyed performance; fixing
the weighting stopped the damage; and the missing end token in the labels was
what was actually preventing the model from stopping. The last one was a
five-line fix that produced the only real gain of the three.
