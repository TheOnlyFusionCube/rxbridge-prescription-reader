# RxBridge — verified evidence base for the Devpost submission

Every number below was measured in this repo, not estimated, and each one names
the artifact or command that produced it. Two early figures are the exception
and are called out here because they are not labelled at their point of use:
the docTR char-F1 of 0.3076 and the TrOCR CER of 0.9486 were measured in an
earlier session whose eval harness is not in this repository. The four fixtures
they were scored against are still in `fixtures_real/`, but nothing here
re-derives those two numbers, so they rest on that earlier run rather than on
anything a judge can re-execute.

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
374 scored rows in `eval/BENCHMARK.json`. Reproduce with:

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
374 scored rows in `eval/BENCHMARK_TRAINED.json`.

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
374 scored rows in `eval/BENCHMARK_V2.json`.

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
right, individual lines are exact (th exact-match 0.38, hi 0.40) and then runs on
others, and the harness's own 128-token ceiling compounds it for the
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
verdict FAIL.** 374 scored rows in `eval/BENCHMARK_V3.json`.

The conclusion is unchanged and is weaker than the section above claimed. The
character-balanced retrain did recover the line-form languages from the
row-capped regression (fr 10.97 to 0.1243, de 10.01 to 0.0832) and German does
beat the zero-shot baseline. But the leaked budget flattered four languages,
and with it removed Arabic and Persian now sit at their zero-shot values while
ru, en and vi are within noise of it. Only th, hi and ur remain clearly worse
than zero-shot, and those are exactly the languages the val probe showed
failing on their own training rows.

Nothing here beats the zero-shot benchmark. The lesson is that the corpus
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

| language | n | mean CER | median | exact | zero-shot | `adapter_v2` (no EOS) |
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
374 scored rows in `eval/BENCHMARK_V4.json`.

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
cap. English moves the same way and further (0.5816 to 0.3559). So the
128-token ceiling understates ru and en.

The one movement that runs the other way is Arabic, and it is worth being
precise about because the comparison is not clean. Zero-shot Arabic *improves*
with room, 0.7961 at 128 tokens to 0.7357 at 1024. But `adapter_v3` at 1024
tokens scores 2.2318, against 0.7917 for `adapter_v4` at 128 and 0.7999 for the
pre-EOS `adapter_v2` at 128. Those last two are different adapters, so this is
not a within-adapter budget effect and is not presented as one — there is no
`adapter_v3` run at 128 tokens to compare against. What it does show is that
the trained EOS adapter rambles on Arabic specifically when given room, which is
the rampaging signature surviving into high budgets and is a reason not to raise
the cap globally.

So the 128-token ceiling should be read as understating ru and en, being
neutral-to-helpful for Arabic on an untrained model, and leaving ar (trained),
fa and ur as genuine model failures that more tokens do not fix.

Neither supplementary arm changes the verdict, and neither is counted in the
3-of-10 headline.

### Status of the multilingual claim

Nothing here is folded into the shipped system or the submission headline.
The benchmark still reports FAIL, and the honest summary of the arc is: the
corpus was so imbalanced that the first fine-tune destroyed performance; fixing
the weighting stopped the damage; and the missing end token in the labels was
what was actually preventing the model from stopping. The last one was a
five-line fix that produced the only real gain of the three.

---

## End-to-end verification on a printed prescription (re-measured 2026-10-09)

`README.md` and `submission/devpost.md` both state a printed-prescription
result and then point here. The numbers lived only in `devpost.md`, with no
verification behind them, so they were re-derived by running the pipeline.

Measured with `scripts/run_api.sh` under `.venv` (docTR 1.0.1, DistilBERT
tagger), `POST /extract` with `fixtures_real/printed_full.jpg`, graded against
`fixtures_real/printed_full_gt.json`:

| claim | measured |
|---|---|
| OCR lines | 20 of 20, confidence 0.8356 to 0.9933 |
| drugs extracted | 3 of 3, matching the ground truth by name, dose and frequency |
| drug-level confidence | 0.6125, 0.6550, 0.6333 |
| overall confidence | 0.6336 |
| duration | recovered for 1 of 3 (Amoxicillin, 10 days); the other two null |
| warnings | none |

The three extracted drugs are Amoxicillin 500 mg twice daily, Paracetamol
500 mg every 8 hours as needed, and Omeprazole 20 mg once daily, which is what
the ground truth states. No case is populated that the form does not support.

Two corrections to what `devpost.md` previously claimed, both now applied
there and in `README.md`:

1. "0.87-0.99 confidence" was wrong at the floor. The measured minimum is
   0.8356, which rounds to 0.84. The upper bound and the 20/20 count are
   correct.
2. "3/3 drugs extracted with dose, frequency, day-grid times and duration"
   read as duration for all three. Only Amoxicillin carries a duration, because
   it is the only one the form gives one; the other two return null. The API is
   declining to guess rather than failing to extract, so this is stated as the
   measured scope instead of the fuller-sounding original.

Reproduce:

    bash scripts/run_api.sh
    curl -s -X POST http://127.0.0.1:8010/extract \
      -F "image=@fixtures_real/printed_full.jpg"

---

## Web frontend build and bundle size (verified 2026-10-09)

`README.md` status line claims the Next.js PWA "builds clean, 105-108 kB
first-load JS". Verified against a fresh build rather than the stale `.next`
directory already on disk:

    cd web && ./node_modules/.bin/next build

Result: compiled successfully, types checked, 4 static pages generated, no
warnings or errors. Next.js's own report:

| route | First Load JS |
|---|---|
| `/` | 108 kB |
| `/_not-found` | 106 kB |
| shared by all routes | 105 kB |

So the claimed range is exact: 105 kB is the shared floor and 108 kB the root
route ceiling, both as reported gzipped by Next.js itself.

Two earlier hand-computations from the stale build chunks were wrong, for
different reasons, and neither is kept. Summing `rootMainFiles` plus the `/page`
entry gave 208.4 kB because that entry already repeats the shared chunks, so the
shared ones were counted twice; a separate sum that added `polyfills` and
`framework` as well gave 233 kB and over-counted those too. Next.js's own report
is the source of truth here, not the chunk arithmetic.

---

## docTR printed-form confidence, corrected (2026-10-09)

`README.md` and `submission/devpost.md` both claimed docTR reads the printed
scaffolding "at ~0.99 confidence". Measured from the same e2e run as the table
above - 20 OCR lines on `fixtures_real/printed_full.jpg`:

| confidence | lines |
|---|---|
| min | 0.8356 |
| max | 0.9933 |
| >= 0.99 | 1 of 20 |
| >= 0.95 | 10 of 20 |

So ~0.99 was one line, not the printed half generally. Both files now state
0.84-0.99, which is the measured range. The corrected claim survives: docTR
handles the printed half and fails on the cursive, which is what the 0.3076
char-F1 and the `Mletzina 20` substitution show.

---

## The EOS-fixed model measured at a 1024-token budget (2026-10-10)

`adapter_v4` -- the model the EOS stop-rule fix produced -- was measured at a
1024-token budget, the same pre-declared budget the supplementary zero-shot run
used, so the comparison is arm-for-arm. Held-out `htr/test`, 40 rows per
language, identical to every other run in this file.

| language | n | CER @128 | CER @1024 | verdict |
|---|---|---|---|---|
| de German | 40 | 0.0673 | 0.0673 | **PASS** |
| fr French | 40 | 0.1092 | 0.1092 | **PASS** |
| th Thai | 40 | 0.1629 | 0.1629 | **PASS** |
| en English | 14 | 0.5816 | 0.4351 | fail |
| hi Hindi | 40 | 0.5352 | 0.5352 | fail |
| vi Vietnamese | 40 | 0.2616 | 0.6013 | fail |
| ru Russian | 40 | 0.9233 | 0.8115 | fail |
| fa Persian | 40 | 0.8011 | 1.4822 | fail |
| ar Arabic | 40 | 0.7917 | 2.1959 | fail |
| ur Urdu | 40 | 1.0449 | 3.2033 | fail |

**Overall: 3 passed, mean CER 0.9604 at 1024 tokens against 0.5279 at 128.**
Scrubbed per-row data in `eval/BENCHMARK_V4_1024.json`; the raw output was
deleted after scrubbing rather than kept.

### What this run settles

The budget was **not** what held the failing languages back, which was the
hypothesis this run was launched to test. The pass set is identical at both
budgets -- de, fr, th clear the bar either way -- and every other language
still fails at 1024.

Only English genuinely improved, 0.5816 to 0.4351, and it is the clearest
remaining case of the 128-token ceiling truncating an answer: its references
have a median of 1452 characters, which 128 tokens cannot express. That part of
the earlier reading stands. English is still not passing.

The line-form languages that already fit in 128 tokens did not move at all.
Hindi is byte-identical at both budgets, which is what a reference with a
median of 11 characters should do when the budget stops being the binding
constraint.

Every language that got worse did so because more room lets it run longer.
Arabic went 0.7917 to 2.1959, Urdu 1.0449 to 3.2033, Persian 0.8011 to 1.4822,
Vietnamese 0.2616 to 0.6013. Vietnamese is the sharpest illustration: at 128
tokens it sat 0.0116 above the bar, and the budget increase pushed it 0.34
further away rather than over. That is the same rambling behaviour the EOS fix
addressed for Thai, surviving at high budgets for scripts where the stop rule
did not fully take.

So the residual gap for ru, fa, ar, ur, vi and hi is not the harness ceiling
and not corpus weighting. It is that the model still fails to stop, and more
generation budget converts that into a larger number rather than a better one.
The next lever is strengthening the stop signal itself -- more optimiser steps,
or a decode-time repetition/stall penalty applied identically to both arms and
declared before the run -- not a larger token budget.

Neither budget changes the headline: 3 of 10 languages pass and the verdict is
FAIL.


---

## Decode-time stall penalties on the EOS-fixed model (2026-10-10)

The section above names a decode-time repetition/stall penalty as the next lever.
That was tested, and this is the pre-declared version of it: the penalty values
were committed to benchmark.py (64cbfeb) before any test-shard number was
looked at, so the ordering is declared-then-measured rather than the reverse.

Choosing the values was done on the val shards, which are training data
(load_rows reads train+val). That limitation is disclosed rather than buried:
val can show a penalty does not hurt, but it cannot prove a held-out gain. It
was optimistic -- val said vi would land at 0.2415, below the bar; held-out it
came in at 0.2648, above it. `scripts/decode_probe.py` holds the probe.

Held-out htr/test, 40 rows per language, 128-token budget, adapter_v4:

| language | unpenalised | penalised | verdict |
|---|---|---|---|
| de German | 0.0673 | 0.0679 | **PASS** |
| fr French | 0.1092 | 0.1100 | **PASS** |
| th Thai | 0.1629 | 0.1721 | **PASS** |
| vi Vietnamese | 0.2616 | 0.2648 | fail |
| fa Persian | 0.8011 | 0.7885 | fail |
| ar Arabic | 0.7917 | 0.7468 | fail |
| hi Hindi | 0.5352 | 0.5464 | fail |
| ur Urdu | 1.0449 | 0.8073 | fail |
| ru Russian | 0.9233 | 0.9163 | fail |
| en English | 0.5816 | 0.5200 | fail |

**Overall: 3 passed, mean CER 0.4940 against 0.5279 unpenalised.** Scrubbed
per-row data in `eval/BENCHMARK_PENAL.json`.

The mean improves but the passes do not move, which is the distinction this
benchmark exists to make. The objective is not mean CER -- it is every language
clearing 0.25 -- so a lower average that leaves three languages failing is not
progress by the run's own rule.

Where the improvement comes from is the long-form languages, not the ones near
the bar. English, Arabic, Persian, Urdu and Russian all improve, because the
penalty shortens the run-on answers that inflated them. The languages that were
close to passing move the wrong way: vi 0.2616 to 0.2648, th 0.1629 to 0.1721,
hi 0.5352 to 0.5464. Penalising repetition costs accuracy on the short
references where the answer is right and must not be trimmed.

So the penalty is a net negative for the objective. It helps the languages that
are far from passing and slightly hurts the ones closest, which is the opposite
of what was wanted. Both earlier conclusions it contradicts are worth stating:
against adapter_v2 the penalty cost 2.6x and was rightly rejected; here it does
not destroy the run, because the EOS fix already stopped the runaway repetition
the penalty was invented to suppress. It is now unnecessary rather than
harmful, and the stop-rule fix is what did the real work.

The flags remain in benchmark.py at no-op defaults so the numbers above and
every earlier result stay reproducible.


---

## The EOS-fixed model at a 1024-token budget with penalties (2026-10-10)

Combining the 1024-token budget with the decode-time penalties is the first
configuration to move four languages under the bar. It is recorded here with
the caveat that it needs, because it was not a pre-declared arm.

| language | 128 tok, no penalty | 128 tok, penalties | 1024 tok, no penalty | 1024 tok, penalties |
|---|---|---|---|---|
| de German | 0.0673 | 0.0679 | 0.0673 | 0.0679 |
| fr French | 0.1092 | 0.1100 | 0.1092 | 0.1100 |
| th Thai | 0.1629 | 0.1721 | 0.1629 | 0.1721 |
| vi Vietnamese | 0.2616 | 0.2648 | 0.6013 | **0.2060** |
| hi Hindi | 0.5352 | 0.5464 | 0.5352 | 0.5464 |
| en English | 0.5816 | 0.5200 | 0.4351 | 0.4610 |
| fa Persian | 0.8011 | 0.7885 | 1.4822 | 0.7264 |
| ar Arabic | 0.7917 | 0.7468 | 2.1959 | 0.7360 |
| ru Russian | 0.9233 | 0.9163 | 0.8148 | 0.8148 |
| ur Urdu | 1.0449 | 0.8073 | 3.2033 | 3.2033 |

**Overall: 4 passed, mean CER 0.4648, verdict FAIL.** Scrubbed per-row data in
`eval/BENCHMARK_PENAL_1024.json`.

The four passing languages are de, fr, th and vi. The fourth is the amount:
Vietnamese was 0.2616 at 128 tokens without penalties and 0.2648 with them,
both just above the bar, and only the combination of a larger budget with the
penalties takes it under, at 0.2060. Neither setting does it alone.

The reason this row is disclosed rather than headlined: the budget and the
penalty values were each pre-declared in code before any test number was seen,
but the *combination* was chosen after seeing the 128-token results. That is
selection on the test shards, and it means the vi figure is an optimistic
estimate of what that configuration would do on data it was not chosen
against. The three languages that pass in all four columns are the ones the
pre-declared settings established.

Urdu is the clearest failure of the combination and worth stating plainly. It
improved from 1.0449 to 0.8073 with penalties at 128 tokens, then went to
3.2033 with the larger budget, and the penalties did not recover it. Penalising
repetition does not help a language the model answers by emitting a long
answer it was never taught to stop: it needs the stop signal itself, which is
what the EOS fix addressed for Thai and only partly took hold elsewhere.

So the standing is 4 of 10 languages at CER 0.25, mean 0.4648, verdict FAIL.
The retrains launched after this run exist to attack the remaining six with
more data rather than more decoding, since the decoding levers are now
measured and their ceiling is visible.


---

## Prompt wording on the EOS-fixed model (2026-10-10)

With budget and penalties both measured and at their ceiling, the one lever
left that acts *before* the model decides to emit a 500-char essay is the text
it reads alongside the image. The stop-rule gap is a decision the model makes
early, and every decode-time lever can only shorten the damage after the fact.
Three prompt variants were tested on the seven languages that still fail after
the 1024+penalty run, 15 val rows each, 128 tokens, greedy, adapter_v4:

  base    - the exact prompt every published run used (control)
  oneline - base + "Output exactly one line of text and stop."
  stop    - base + "Stop immediately after the last character of the
             transcription."

| prompt | ur | fa | ar | ru | en | hi | vi | MEAN |
|---|---|---|---|---|---|---|---|---|
| base | 0.8208 | 0.8004 | 0.8041 | 0.9292 | 0.3436 | 0.4592 | 0.2827 | 0.6343 |
| oneline | 0.8241 | 0.8666 | 0.8564 | 0.9309 | 0.3641 | 0.4592 | 0.2659 | 0.6525 |
| stop | 0.7583 | 0.8046 | 0.8113 | 0.9305 | 0.3380 | 0.4592 | 0.2655 | 0.6239 |

**Verdict: prompt wording is not the lever.** Measured on the val shards, which
are training data, so none of this is selected against held-out test; it is a
ceiling check, not a result. `prompt_probe.py` on the cluster holds the probe.

The extra instruction does not make the model stop. `oneline` is worse than the
control on four of seven languages and worse on the mean (0.6525 against
0.6343), because "one line" reads as a formatting constraint on a model that is
already emitting a paragraph, and it answers by padding. `stop` is the best of
the three at 0.6239, but the entire gain is 0.0104 of mean CER, and it comes
from ur (0.8208 to 0.7583) and en (0.3436 to 0.3380) -- trimming the long wrong
answers the control produces -- while vi moves 0.2827 to 0.2655 and stays above
the bar. No language crosses 0.25 in any variant.

The instructive part is hi: identical to four decimals across all three prompts
(0.4592). When the model emits the same wrong answer regardless of what it is
told, the instruction is not reaching the decision. That is a capacity and data
statement, not a prompting one, and it agrees with what the decode levers
already showed.

So three different kinds of lever -- budget, penalties, prompt wording -- all
bottom out at the same wall, and the wall is the stop rule for the six
long-form languages. The remaining attack is the one already in flight: more
training data per language, via the two retrains. Nothing here changes any
published number; the control column IS the published prompt, and it stays.


---

## Retrain A: 200k chars per language, held-out at 128 tokens with penalties (2026-10-10)

adapter_v5 was trained with `--per-lang-chars 200000` -- against v4's 40000 --
holding everything else at the v4 settings: 1000 optimiser steps x 8
accumulation, lr 2e-4, rank 16, EOS in the labels, global shuffle. The attack
was data volume, because budget, penalties and prompt wording had each been
measured and each had bottomed out. This is the same held-out protocol as every
run above: htr/test, 40 rows per language, the exact penalty arm the val-shard
probe picked (`--repetition-penalty 1.2 --no-repeat-ngram-size 3`), declared
before the run.

| language | v4 penalised 128 | v5 penalised 128 | verdict |
|---|---|---|---|
| de German | 0.0679 | 0.0418 | **PASS** |
| fr French | 0.1100 | 0.0777 | **PASS** |
| th Thai | 0.1721 | 0.1632 | **PASS** |
| vi Vietnamese | 0.2648 | 0.1945 | **PASS** |
| hi Hindi | 0.5464 | 0.4184 | fail |
| en English | 0.5200 | 0.5926 | fail |
| fa Persian | 0.7885 | 0.7751 | fail |
| ar Arabic | 0.7468 | 0.7110 | fail |
| ur Urdu | 0.8073 | 0.7967 | fail |
| ru Russian | 0.9163 | 0.9111 | fail |

**Overall: 4 passed, mean CER 0.4682 against 0.4940. Verdict FAIL.** Scrubbed
per-row data in `eval/BENCHMARK_V5_PENAL.json`.

The pass count is unchanged at four, and that is not the headline. What changed
is which caveat those four carry. Under v4, the fourth pass only existed in the
1024-token arm, and that arm was chosen after seeing the 128-token results, so
vi's 0.2060 was disclosed as optimistic. Here vi passes at 0.1945 in the
pre-declared 128-token penalty arm, the same configuration that graded v4 at
0.2648. The fourth pass is now clean by the run's own rule, and it was earned
by data volume rather than by moving the decoder.

The movement is concentrated exactly where the corpus diagnosis said it would
be. Vietnamese went 0.2648 to 0.1945, the largest single change in the table,
and it was the language sitting 0.0116 above the bar at v4. Hindi went
0.5464 to 0.4184 and is the only other failing language that moved by more
than 0.1 -- the two paragraph-form languages with the most headroom between
"needs more data" and "needs a different model". Everything else moved by
0.001 to 0.036, which is the range of noise for 40 rows.

The two honest regressions, recorded rather than rounded away. English got
worse, 0.5200 to 0.5926. It is the language whose test shard has only 14 rows
and whose references have a median of 1452 characters, so it is both the
noisiest number in the table and the one most exposed to the earlier finding
that a 128-token budget truncates a correct answer. Russian is essentially
flat at 0.9111 against 0.9163 and remains the worst performer -- it supplied
72.2% of training characters under the old row cap and was the language the
char budget was designed to stop dominating.

So the data hedge produced a real held-out improvement -- a cleaner fourth
pass and the largest CER move since the EOS fix -- without reaching five. The
verdict is still FAIL, and the standing pass count is still four of ten. The
1024-token arm on v5 was measured to complete the picture, and retrain B
(600k chars per language) is the larger-data endpoint of the same attack.


---

## adapter_v5 at the 1024-token budget with penalties (2026-10-10)

Same adapter as the section above, same declared penalty arm, at the 1024-token
budget. This is the second half of the disclosure rule: the v4 numbers in that
column were post-hoc, so the v5 column is measured to show whether the
configuration that produced v4's optimistic vi actually holds up on a better
model, or whether it was a selection artefact.

| language | v4 128 penal | v4 1024 penal | v5 128 penal | v5 1024 penal |
|---|---|---|---|---|
| de German | 0.0679 | 0.0679 | 0.0418 | 0.0418 |
| fr French | 0.1100 | 0.1100 | 0.0777 | 0.0777 |
| th Thai | 0.1721 | 0.1721 | 0.1632 | 0.1632 |
| vi Vietnamese | 0.2648 | 0.2060 | 0.1945 | **0.1313** |
| hi Hindi | 0.5464 | 0.5464 | 0.4184 | 0.4184 |
| en English | 0.5200 | 0.4610 | 0.5926 | 0.5867 |
| fa Persian | 0.7885 | 0.7264 | 0.7751 | 0.7027 |
| ar Arabic | 0.7468 | 0.7360 | 0.7110 | 0.6168 |
| ur Urdu | 0.8073 | 0.8073 | 0.7967 | 0.7967 |
| ru Russian | 0.9163 | 0.8148 | 0.9111 | 0.8025 |

**Overall: 4 passed, mean CER 0.4338 -- the lowest mean of any run here.**
Scrubbed per-row data in `eval/BENCHMARK_V5_PENAL_1024.json`.

The budget now buys something it did not buy for v4. Vietnamese falls from
0.1945 to 0.1313, its best number in the campaign and 0.0753 under the bar, and
it is the only language that moves at all. Arabic 0.7110 to 0.6168 and Russian
0.9111 to 0.8025 improve without crossing. Everything else is unchanged to four
decimals, including Hindi at 0.4184 in both columns and English moving only
0.0059.

That is the clean reading of why v4's 1024 run looked like a trap. For v4 the
same budget made vi worse (0.2648 to 0.6013 without penalties) and in
combination recovered it only to 0.2060 -- the model ran longer, rambled, and
the penalties had to claw the answer back. For v5 the stop rule is solid enough
that extra room is spent on content instead of repetition, and vi uses it: vi's
references have a median of 447 characters, which needs roughly 184 tokens, so
at a 128-token budget the answer was being truncated. The budget was never the
lever; a working stop rule was the precondition for the budget to matter.

English is the honest exception and it cuts the other way: at 128 tokens it was
0.5926 and at 1024 it is 0.5867, meaning the larger budget recovered almost
none of the 0.5200 it scored under v4. Its references have a median of 1452
characters, far beyond even 1024 tokens, so no budget in this campaign reaches
them. That is a corpus problem, not a decoding one.

So v5 holds four passes at both budgets with a mean of 0.4682 at 128 and 0.4338
at 1024, and both numbers are better than v4's equivalent arms. The verdict is
still FAIL.


---

## Retrain B: 600k chars per language, held-out at 128 tokens with penalties (2026-10-10)

The same attack as retrain A with three times the character budget, to find
where the data-volume curve turns over. Everything else identical: 1000
optimiser steps x 8 accumulation, lr 2e-4, rank 16, EOS in the labels, global
shuffle, held-out htr/test at 40 rows per language, the declared penalty arm.

| language | v5 200k penal | v6 600k penal | verdict |
|---|---|---|---|
| de German | 0.0418 | 0.0413 | **PASS** |
| fr French | 0.0777 | 0.0786 | **PASS** |
| vi Vietnamese | 0.1945 | 0.1755 | **PASS** |
| hi Hindi | 0.4184 | 0.3802 | fail |
| en English | 0.5926 | 0.5709 | fail |
| fa Persian | 0.7751 | 0.7574 | fail |
| ar Arabic | 0.7140 | 0.7140 | fail |
| ur Urdu | 0.7967 | 0.8139 | fail |
| ru Russian | 0.9111 | 0.9103 | fail |
| th Thai | 0.1632 | 0.2976 | fail |

**Overall: 3 passed, mean CER 0.4740 against v5's 0.4682 and 4 passes.
Verdict FAIL.** Scrubbed per-row data in `eval/BENCHMARK_V6_PENAL.json`.

More data is not uniformly better, and this is the run that proves it. Thai
went from 0.1632, a pass with 35% headroom under the bar, to 0.2976 -- the
largest regression in the campaign and the only pass that a training change has
ever cost. Thai was the language the EOS fix was invented for; its references
are short (median 11 characters) and it was already solved. Tripling the
character budget diluted a language that was already working, which is the
failure mode the char budget was supposed to prevent and instead produced from
the other direction.

Everything else says the same thing more quietly. Hindi improved to 0.3802, its
best number in any arm, and Arabic, Russian, Persian and Urdu are flat to within
0.04. English improved 0.0217. None of it crossed the bar, and the one language
that did cross it under v5 fell out.

So the volume curve has a peak and 600k is past it: v4 at 40k scores 3 passes
and 0.4940, v5 at 200k scores 4 and 0.4682, v6 at 600k scores 3 and 0.4740.
adapter_v5 is the best model in the campaign on both counts and it is the one
every comparison should be graded against from here.

What the three data arms together establish is that data volume was worth
exactly one pass and a mean improvement of 0.06, and it is now also spent. The
six remaining languages are not going to be closed by re-running the same
trainer with a different number, and the next question is what actually binds
them -- which is what the corpus counts below are for.


---

## Corpus supply: what binds the remaining six (2026-10-10)

The data-volume question only makes sense once you know how much data each
language actually has. Counted over `htr/<lang>/{train,val,test}.jsonl`,
characters of ground-truth text, against the caps the retrains used:

| language | rows | chars | % of 200k cap | % of 600k cap |
|---|---|---|---|---|
| ru Russian | 3400 | 12,856,352 | 100% | 100% |
| fa Persian | 3400 | 3,529,506 | 100% | 100% |
| vi Vietnamese | 1144 | 499,563 | 100% | 83% |
| ar Arabic | 457 | 357,708 | 100% | 59% |
| fr French | 3400 | 157,763 | 78% | 26% |
| de German | 3400 | 141,866 | 70% | 23% |
| th Thai | 3400 | 137,432 | 68% | 22% |
| ur Urdu | 2403 | 105,833 | 52% | 17% |
| en English | 74 | 82,231 | 41% | 13% |
| hi Hindi | 3400 | 69,322 | 34% | 11% |
| bn Bengali | 0 | 0 | 0% | 0% |
| ko Korean | 0 | 0 | 0% | 0% |
| ml Malayalam | 0 | 0 | 0% | 0% |
| or Odia | 0 | 0 | 0% | 0% |
| zh Chinese | 0 | 0 | 0% | 0% |

Read the top four rows against the retrain results and the campaign's entire
data story closes. Russian and Persian have 12.8M and 3.5M characters and score
0.9111 and 0.7751 -- among the worst in the table. They are not data-starved at
any budget in this campaign; the 200k and 600k caps both saturate on them, which
is why v5 and v6 moved ru and fa by 0.005 and 0.015 respectively. Their failure
is model capacity for those scripts, not corpus volume, and no version of this
trainer fixes it. Arabic and Vietnamese are the same story with less margin:
both saturate at 200k, and vi is the language that did cross, at 0.1313 with the
largest budget. There is no remaining data lever for any of them.

The bottom rows are where the actual opportunity is, and they are not a
modelling problem. English has 74 rows and 82k characters, and its references
have a median of 1452 characters -- its ceiling in this corpus is a handful of
long prescriptions, and 40 test rows of those is a noisy verdict, not a
measurement. Hindi has 3400 rows but only 69k characters, a median reference
near 11 characters: it is line-form like Thai, so its rows are cheap in
characters and the 600k cap spent 11%. Both improved when they got more data
(hi 0.5464 to 0.3802 across the three arms) and both are now limited by supply.
Urdu at 105k and 52% is the same shape as hi.

The five empty directories are the finding that matters for the stated goal.
The benchmark grades ten languages and the target is ten-plus; `htr/` contains
`bn`, `ko`, `ml`, `or` and `zh` directories with zero rows in every split. That
means five of the fifteen languages on disk are unmeasured rather than failing,
and the campaign has never tested whether the model can transcribe them at all.
The honest standing is therefore not "6 of 10 fail" but "4 of 10 pass and 5 more
are unmeasured", and the path to a higher pass count is corpus acquisition for
those five, not further tuning of the trainer or the decoder.

Nothing in this section changes a graded number. It is the census that says
where the next unit of work belongs.


---

## English supply: the IAM retrain, held-out at both budgets (2026-10-10)

The census above identifies English as the one graded language that is not
supply-saturated: 74 corpus rows, 82k characters, 41% of the 200k cap, against
3,400 rows for its neighbours. It is also the language that has failed at every
decode configuration ever measured, at 0.5200 to 0.6010, with references of a
median 1,452 characters. This run tests whether that failure is a supply
problem by acquiring English handwriting data and retraining on it.

Corpus acquisition for the five empty languages was attempted first and each
failed for a recorded reason: Bengali gated behind a 403, Chinese's repo 404s,
Korean ships 5 images with polygon regions, Odia ships character-classification
metadata with no line images, and the "multi-lingual" dataset is Hindi-only.
The English source that did exist is the IAM handwriting corpus via a public
5,663-pair distribution: `fresh/iam_voxel` with a `samples.json` pairing each
`data/NNNNN.png` to a transcript. A reader (`read_iam_voxel`, commit 622565d)
and a SOURCES entry were added so en trains from AMR plus IAM.

The rebuild was guarded so it could not move the verifier. `scripts/rebuild_en.py`
verifies the other nine languages' shards are byte-identical against a
pre-build checksum manifest, restores en's test and val from backup, and strips
any train row whose image also appears in the restored test or val. All three
matters: `write_shards` would have drawn 200 test rows and 200 val rows out of
the now-5,737-row pool, silently re-grading English against IAM line crops
instead of the 14 paragraph-form rows every published English number was
measured on; and the seed-0 shuffle with a 3,000-row train cap put 20 of the 21
original held-out AMR images into the new train shard by chance. The rebuild
reported en train 53 to 2,980 rows, test 14 and val 7 restored byte-identical
(`aaa3cc83`, `d310a711`), zero remaining test/val images in train, and the nine
other languages unchanged.

The disclosed caveat, stated before the number: IAM rows are line and
multi-line crops, 3 to 524 characters with a median of 86, while the en test
shard is paragraph form with a median of 1,452. So this experiment can buy the
model English script-reading ability. It cannot buy it paragraph structure, and
it is not a like-for-like addition.

Training mix, verbatim from the two runs' load_rows output. Every language is
identical between them except English; the only variable in this experiment is
English data.

| language | v5 rows | v5 chars | v7 rows | v7 chars |
|---|---|---|---|---|
| en English | 60 | 61,780 | 1,500 | 185,550 |
| fr French | 1,500 | 69,653 | 1,500 | 69,653 |
| th Thai | 1,500 | 61,545 | 1,500 | 61,545 |
| de German | 1,500 | 62,904 | 1,500 | 62,904 |
| hi Hindi | 1,500 | 31,341 | 1,500 | 31,341 |
| ur Urdu | 1,500 | 66,117 | 1,500 | 66,117 |
| ar Arabic | 259 | 199,999 | 259 | 199,999 |
| fa Persian | 194 | 199,649 | 194 | 199,649 |
| ru Russian | 52 | 197,822 | 52 | 197,822 |
| vi Vietnamese | 459 | 199,982 | 459 | 199,982 |
| total | 8,524 | 1,150,792 | 9,964 | 1,274,562 |

English's share of training characters went from 5.4% to 14.6%, a 2.7x raise,
paid for proportionally by the other nine out of a fixed 200k-per-language cap.
The average English row went from 1,030 characters to 124, which is the
granularity mismatch expressed as a number.

Held-out htr/test, 40 rows per language, adapter_v7 against adapter_v5 at the
declared penalty arm, 128 tokens:

| language | v5 | v7 | verdict |
|---|---|---|---|
| de German | 0.0418 | 0.0473 | **PASS** |
| fr French | 0.0777 | 0.0953 | **PASS** |
| th Thai | 0.1632 | 0.1852 | **PASS** |
| vi Vietnamese | 0.1945 | 0.2093 | **PASS** |
| hi Hindi | 0.4184 | 0.3376 | fail |
| en English | 0.5926 | 0.6010 | fail |
| fa Persian | 0.7751 | 0.7716 | fail |
| ar Arabic | 0.7140 | 0.7258 | fail |
| ur Urdu | 0.7967 | 0.8103 | fail |
| ru Russian | 0.9111 | 0.9160 | fail |

**Overall: 4 passed, mean CER 0.4699 against v5's 0.4682. Verdict FAIL.**
Scrubbed per-row data in `eval/BENCHMARK_V7_PENAL.json`.

English failed the purpose of the experiment. It did not improve at the
primary budget, it regressed by 0.0084, and the language that was supposed to
benefit is the one that got worse. Worse, it got worse on its own training
data: the inline val eval -- the 7 original AMR val rows, which the model
trained on -- scored en 0.4217 against v5's 0.4161. A model that cannot
transcribe the seven English rows it was fitted on has not been taught English;
it has been taught something else in the English slot.

The cost is visible in the four passing languages, all of which drifted the
wrong way by a similar margin: fr +0.0176, th +0.0220, vi +0.0148, de +0.0055.
That is the 2.7x gradient share coming out of their budget, and it is why the
experiment is a net negative at this budget even before English is counted.

At the 1024-token budget the same run tells a more interesting story, and it is
the reason the experiment was worth running.

| language | v5 | v7 | verdict |
|---|---|---|---|
| de German | 0.0418 | 0.0473 | **PASS** |
| fr French | 0.0777 | 0.0953 | **PASS** |
| th Thai | 0.1632 | 0.1852 | **PASS** |
| vi Vietnamese | 0.1313 | 0.1376 | **PASS** |
| hi Hindi | 0.4184 | 0.3376 | fail |
| en English | 0.5867 | **0.4369** | fail |
| fa Persian | 0.7027 | 0.6372 | fail |
| ar Arabic | 0.6168 | 0.5725 | fail |
| ur Urdu | 0.8107 | 0.8103 | fail |
| ru Russian | 0.8025 | 0.8102 | fail |

**Overall: 4 passed, mean CER 0.4070 -- the best mean of the campaign, against
v5's 0.4338. Verdict FAIL.** Scrubbed per-row data in
`eval/BENCHMARK_V7_PENAL_1024.json`.

English 0.5867 to 0.4369 is its best number in the campaign, and Arabic 0.5725
and Persian 0.6372 are bests for them too. So the IAM data did teach the model
to emit more English, and at a budget with room to emit it, that shows up. But
0.4369 against a 0.25 bar is not a near miss, and 0.4369 is the best the model
has ever done on this language at any setting. The candour required here is that
even the good version of this result is 0.19 above the bar.

So the supply lever for English is closed, and it is closed by the corpus
rather than by the trainer. What English needs is paragraph-form English
handwriting, and there are 74 rows of it in AMR and none in IAM; the public
IAM distribution is line crops by construction. More of it buys more of the
wrong shape, and the 128-token arm shows that shape actively costing the four
passes.

The standing is unchanged and now measured from every angle: 4 of 10 languages
pass, best model is adapter_v5 at 0.4682 on the declared arm, verdict FAIL. The
six failures are bound by capacity for the saturated scripts (ru, fa, ar
saturate at 200k and did not move when they got 600k) and by reference form for
en and hi. The 10-language goal needs corpora that do not exist on disk: five
languages have zero rows, and English has no paragraph-form data to acquire.


---

## The row cap: a lever no run had moved (declared 2026-10-11)

The census above ends by blaming supply, and for Hindi that was the wrong
diagnosis. It deserves correcting before it is built on.

`load_rows` in train_lora.py caps every language twice. The character cap is
the one this campaign has measured and tuned, from 40k through 200k to 600k.
The row cap sits in front of it and no run has ever moved it:

    random.Random(seed).shuffle(found)
    found = found[:per_lang]        # <-- the row cap, default 1500
    used = 0
    kept = []
    for row in found:               # <-- only then the char cap
        ...

Counted over the built corpus, train+val, what was actually on disk against
what `--per-lang 1500` ever let the trainer see:

| language | train+val on disk | rows ever used | rows never used |
|---|---|---|---|
| fr French | 3,200 | 1,500 | 1,700 |
| th Thai | 3,200 | 1,500 | 1,700 |
| de German | 3,200 | 1,500 | 1,700 |
| hi Hindi | 3,200 | 1,500 | 1,700 |
| fa Persian | 3,200 | 1,500 | 1,700 |
| ru Russian | 3,200 | 1,500 | 1,700 |
| ur Urdu | 2,203 | 1,500 | 703 |
| en English | 2,987 | 60 | 2,927 |
| vi Vietnamese | 944 | 459 | 485 |
| ar Arabic | 366 | 259 | 107 |

So seven of the ten graded languages have never been trained on more than half
their rows. Hindi in particular has 3,200 rows and 65,317 characters on disk
against a 200,000-character cap -- the char cap was never binding on it at all,
and 1,700 Hindi rows have never been seen by a single run. The census called
that "limited by supply"; it is limited by a flag nobody touched.

This is also a different direction from the retrain B result, which is what
makes it worth a run. Retrain B raised the char cap to 600k and that diluted
toward the paragraph-form languages, because ru, fa, ar and vi are the ones the
char cap binds -- and it cost Thai its pass. Raising the row cap while holding
the char cap at 200k does the opposite: the paragraph-form languages cannot use
more rows, because they are char-capped inside the same 1,500 they already had,
so the extra rows go entirely to the line-form languages that have them. fr, th,
de, hi and ur each go from 1,500 rows to every row they own.

Declared before any number: adapter_v8 is retrain A's exact configuration
(1,000 optimiser steps x 8 accumulation, lr 2e-4, rank 16, EOS in the labels,
global shuffle, `--per-lang-chars 200000`) with `--per-lang 3200`, the only
change. Both arms are pre-declared: 128 tokens with the `pen1.2+ngram3` penalty
arm, and 1024 tokens with the same penalties. No held-out shard number has been
looked at, and this paragraph is committed before either is run.

The training mix it produces, verbatim from the run's own load_rows output,
against v5 and v7:

| language | v5 rows | v5 chars | v7 rows | v7 chars | v8 rows | v8 chars |
|---|---|---|---|---|---|---|
| en English | 60 | 61,780 | 1,500 | 185,550 | 1,623 | 199,999 |
| fr French | 1,500 | 69,653 | 1,500 | 69,653 | 3,200 | 148,680 |
| th Thai | 1,500 | 61,545 | 1,500 | 61,545 | 3,200 | 129,291 |
| de German | 1,500 | 62,904 | 1,500 | 62,904 | 3,200 | 133,719 |
| hi Hindi | 1,500 | 31,341 | 1,500 | 31,341 | 3,200 | 65,317 |
| ur Urdu | 1,500 | 66,117 | 1,500 | 66,117 | 2,203 | 97,020 |
| ar Arabic | 259 | 199,999 | 259 | 199,999 | 259 | 199,999 |
| fa Persian | 194 | 199,649 | 194 | 199,649 | 194 | 199,649 |
| ru Russian | 52 | 197,822 | 52 | 197,822 | 52 | 197,822 |
| vi Vietnamese | 459 | 199,982 | 459 | 199,982 | 459 | 199,982 |
| total | 8,524 | 1,150,792 | 9,964 | 1,274,562 | 17,590 | 1,571,478 |

Hindi's training characters go 31,341 to 65,317, a 2.08x raise, and its share of
all training characters goes 2.7% to 4.2%. Urdu goes 1.47x in rows and 5.7% to
6.2% in share. The four char-capped languages are byte-identical to v7, which is
what makes the comparison clean: v8 against v7 changes one thing only, the row
cap. v8 against v5 changes two, because IAM is still in the corpus, and that
comparison is reported second for that reason.

The risk is named in advance rather than discovered afterward. Thai passed at
0.1632 on 1,500 rows and retrain B showed what dilution does to it. Thai now
gets 3,200 rows instead of 1,500, and if Thai regresses past 0.25 this run will
have bought Hindi's shot with Thai's pass, which is the same trade retrain B
made in the other direction. That is a real possibility, not a rhetorical one,
and it is stated here before the measurement rather than after it.


---

## The row cap: measured (2026-10-11)

Declared above, then run. adapter_v8 is retrain A's exact configuration with
`--per-lang 3200` and nothing else; the corpus is the one the IAM section left
behind, and en's test/val shards are still the restored originals. Both arms
use the declared penalties.

Held-out htr/test, 40 rows per language, 128 tokens with penalties, v8 against
v7 (the clean single-variable comparison, since only the row cap differs):

| language | v5 | v7 | v8 | verdict |
|---|---|---|---|---|
| de German | 0.0418 | 0.0473 | 0.0621 | **PASS** |
| fr French | 0.0777 | 0.0953 | 0.0744 | **PASS** |
| th Thai | 0.1632 | 0.1852 | 0.2079 | **PASS** |
| vi Vietnamese | 0.1945 | 0.2093 | 0.2335 | **PASS** |
| hi Hindi | 0.4184 | 0.3376 | 0.3376 | fail |
| en English | 0.5926 | 0.6010 | 0.5499 | fail |
| fa Persian | 0.7751 | 0.7716 | 0.7889 | fail |
| ar Arabic | 0.7140 | 0.7258 | 0.7447 | fail |
| ur Urdu | 0.7967 | 0.8103 | 0.8109 | fail |
| ru Russian | 0.9111 | 0.9160 | 0.9162 | fail |

**Overall: 4 passed, mean CER 0.4726 against v7's 0.4699 and v5's 0.4682.
Verdict FAIL.** Scrubbed per-row data in `eval/BENCHMARK_V8_PENAL.json`.

Hindi did not move. It is 0.3376 against v7's 0.3376, identical to four
decimals, and 0.3376 again at 1024 tokens -- the same score on the same 40 rows
at both budgets, which means the model emits the same answers for Hindi whether
it saw 1,500 Hindi rows or 3,200 of them. The lever aimed at the closest
failing language moved it by zero.

The reason is in the corpus rather than the trainer, and it is the finding that
actually matters. The Hindi source is the "multi-lingual" dataset the download
log recorded as `ml OK 1325.9 MB tahirkiller/multi-lingual-handwritten-ocr-dataset`,
and its annotation rows carry `"font": "DejaVuSans.ttf"` and `"font_size": 34`
-- it is rendered text, not handwriting. Hindi's 3,200 rows are 65,317 characters
of the same font at the same size with different strings. A row cap that
uncapped 1,700 more of those rows gave the model more strings in a distribution
it already had 31,341 characters of, and the score confirms that bought nothing.
The binding constraint on Hindi was never the row count; it is that the corpus
contains no handwritten Hindi at all.

What the extra rows did buy was English's best number at this budget, 0.5499
against v7's 0.6010 and v5's 0.5926. That is real and it is the second time an
English move has shown up, and it still fails by 0.30. French improved to
0.0744, its best since the EOS fix. Both gains are consistent with the declared
mechanism: line-form languages got more gradient.

The cost came out of the same place the declaration predicted. Thai went
0.1852 to 0.2079 and Vietnamese 0.2093 to 0.2335, the latter now 0.017 from the
bar, and German 0.0473 to 0.0621. Thai did not lose its pass, so the worst case
named in the declaration did not occur, but the headroom that made Thai a safe
pass at 35% is down to 17%. Arabic and Persian also drifted the wrong way.

At the 1024-token budget the drift becomes a regression for the paragraph-form
languages:

| language | v5 | v7 | v8 | verdict |
|---|---|---|---|---|
| de German | 0.0418 | 0.0473 | 0.0621 | **PASS** |
| fr French | 0.0777 | 0.0953 | 0.0744 | **PASS** |
| th Thai | 0.1632 | 0.1852 | 0.2079 | **PASS** |
| vi Vietnamese | 0.1313 | 0.1376 | 0.1755 | **PASS** |
| hi Hindi | 0.4184 | 0.3376 | 0.3376 | fail |
| en English | 0.5867 | 0.4369 | 0.4389 | fail |
| fa Persian | 0.7027 | 0.6372 | 0.8900 | fail |
| ar Arabic | 0.6168 | 0.5725 | 0.6872 | fail |
| ur Urdu | 0.8107 | 0.8103 | 0.8109 | fail |
| ru Russian | 0.8025 | 0.8102 | 0.8157 | fail |

**Overall: 4 passed, mean CER 0.4500 against v7's 0.4070 and v5's 0.4338.
Verdict FAIL.** Scrubbed per-row data in `eval/BENCHMARK_V8_PENAL_1024.json`.

Persian at 0.8900 is the worst Persian in the campaign and Arabic at 0.6872 is
worse than both earlier models, so the larger budget now hurts two languages
instead of helping them, where under v7 the same budget produced the campaign's
best mean of 0.4070. The extra line-form rows took gradient from the
paragraph-form languages that the budget was being spent on.

One oddity recorded rather than smoothed over. The inline val eval showed Urdu
collapsing from 0.8312 to 2.4616 -- the model stopped being able to fit Urdu's
own training rows -- yet held-out Urdu is 0.8109 against v5's 0.8103. The val
collapse did not transfer to unseen rows, and since val is training data the
honest reading is that v8 stopped memorising Urdu without generalising any
worse. It is a sign the row redistribution destabilised a language's fit, and
it is worth knowing it does not show up on the metric that counts.

So the row cap was a real lever that had never been pulled, it moved two
languages in the right direction, and it closed no gap. The standing is
unchanged: 4 of 10 languages pass, best model is adapter_v5 at 0.4682 on the
declared arm, verdict FAIL.

That is five distinct levers now measured and spent: generation budget,
decode-time penalties, prompt wording, per-language character volume, and the
per-language row count. The two runs that moved the scoreboard were the EOS
stop-rule fix and the character-balanced corpus, both of which predate this
stretch of work. What remains is not a knob on this trainer. Hindi's corpus is
rendered text, English's is 74 paragraph rows against a line-crop public set,
five languages are empty directories, and Russian and Persian have 12.8M and
3.5M characters available and saturate at 200k without moving -- which is
capacity for those scripts, not anything a flag can reach.


---

## Optimiser passes: the one knob never moved (declared 2026-10-11)

There is a sixth knob, and the campaign has never touched it. Every retrain in
this file has run `--steps 1000 --accum 8`, which is 8,000 forward/backward
passes, and that number was fixed before the corpus started growing. It did
grow: v5 trained on 8,524 rows, v7 on 9,964, v8 on 17,590. So the passes each
row was actually seen fell from 0.94 to 0.80 to 0.45 without anyone choosing
for it to.

| run | rows | passes | views per row |
|---|---|---|---|
| v5 (200k, row cap 1500) | 8,524 | 8,000 | 0.94 |
| v7 (200k + IAM) | 9,964 | 8,000 | 0.80 |
| v8 (row cap 3200) | 17,590 | 8,000 | 0.45 |

That reframes the v8 result rather than just recording it. v8 gave Hindi more
than twice the characters it had ever seen and Hindi did not move; it also gave
each row barely half the attention v5 gave its rows. Both readings were offered
in the last section -- the data is rendered text, and the views halved -- and
this run separates them. If v8's flat Hindi and drifting Thai are an
undertraining artefact of the row expansion, then holding the views per row
fixed should recover v7's standing while keeping the English and French gains
the row cap produced. If Hindi is still flat, the rendered-text explanation
stands and the row cap's value was only ever English.

Declared before any number exists: adapter_v9 is adapter_v8's exact
configuration with `--steps 2000` and nothing else changed. That makes 16,000
passes against 17,590 rows, or 0.91 views per row, within 3% of v5's 0.94 --
the density the campaign's best score was achieved at -- while carrying the
row cap and the IAM corpus that v8 introduced. Both held-out arms are
pre-declared: 128 tokens with the `pen1.2+ngram3` penalty arm, and 1024 tokens
with the same penalties, on the untouched htr/test shards.

The comparison that matters is v9 against v8, which changes one thing (the step
count), and secondarily v9 against v5, which changes the row cap and IAM as
well. The cost side is named in advance again: this is the longest run of the
campaign, roughly 3.5 hours at the measured 0.78 seconds per pass, and the
inline val eval has already shown on a long run that a language's fit can
destabilise (v8's Urdu val collapsed to 2.4616 while its held-out number stayed
flat). If v9 inherits that pattern the honest reading is that longer training
reshuffles which languages fit, not that it helps them.

Three specific things would count as a result, stated before the run so they
cannot be moved afterwards: Hindi below 0.3376 at either budget would falsify
the rendered-text explanation and vindicate the row cap; Thai returning to
0.1852 or better would show the dilution was an artefact of the step count;
and any fifth language crossing 0.25 would be the first since the EOS fix. If
none of the three occurs, the step count joins the other five spent levers and
the campaign has measured everything this trainer exposes.


---

## Optimiser passes: measured (2026-10-11)

Run as declared: adapter_v9 is adapter_v8's exact configuration with
`--steps 2000`, 16,000 passes over the same 17,590 rows, or 0.91 views per row
against v8's 0.45. Same corpus, same caps, same penalties, same untouched held-
out shards. The inline val eval looked like it had worked -- overall 0.4640
against v8's 0.6632, with Urdu recovered from its 2.4616 val collapse to 0.7429
-- and val is where the good news stopped.

Held-out htr/test, 40 rows per language, 128 tokens with penalties, v9 against
v8 and v5:

| language | v5 | v8 | v9 | verdict |
|---|---|---|---|---|
| de German | 0.0418 | 0.0621 | 0.0314 | **PASS** |
| fr French | 0.0777 | 0.0744 | 0.0820 | **PASS** |
| th Thai | 0.1632 | 0.2079 | 0.2586 | fail |
| vi Vietnamese | 0.1945 | 0.2335 | 0.2505 | fail |
| hi Hindi | 0.4184 | 0.3376 | 0.3407 | fail |
| en English | 0.5926 | 0.5499 | 0.5380 | fail |
| fa Persian | 0.7751 | 0.7889 | 0.7702 | fail |
| ar Arabic | 0.7140 | 0.7447 | 0.7278 | fail |
| ur Urdu | 0.7967 | 0.8109 | 0.7711 | fail |
| ru Russian | 0.9111 | 0.9162 | 0.9118 | fail |

**Overall: 2 passed, mean CER 0.4682 against v5's 0.4682 and 4 passes.
Verdict FAIL.** Scrubbed per-row data in `eval/BENCHMARK_V9_PENAL.json`.

All three declared criteria failed, and the run is the worst of the campaign on
the declared arm by pass count. Hindi came in at 0.3407 against the 0.3376 the
declaration named, so it missed by 0.0031 -- close enough that it is worth
saying the miss does not rescue the rendered-text reading but does not confirm
it either. Thai went the wrong way entirely, 0.2586 against the 0.1852 it was
supposed to return to, losing its pass outright. Vietnamese also lost its pass
at 0.2505, three thousandths above the bar. No fifth language crossed 0.25, and
the campaign finished two passes where it started this stretch at four.

The mechanism is now legible in the shape of the table rather than the mean.
German improved to 0.0314, the best number of the campaign for any language,
and English to 0.5380, its best at this budget -- both line-form, both
languages that got more gradient under the row cap. Thai and Vietnamese, also
line-form and also passed under v5, got worse. Four languages in one shape
moving in two directions at once points at what doubling the passes actually
did: not fitting the corpus better, but reallocating which sub-distributions
inside each language the model spends its capacity on. The 8,000-pass run was
long enough to learn the common shapes and stop; at 16,000 passes the optimiser
kept going and rebalanced toward the shapes that appear most often, which for
Thai's short references means the model now knows Thai script better and the
specific 40 held-out rows slightly worse.

The 1024-token arm, for completeness, recovered Vietnamese at 0.1953 and still
did not recover Thai:

| language | v5 | v8 | v9 | verdict |
|---|---|---|---|---|
| de German | 0.0418 | 0.0621 | 0.0314 | **PASS** |
| fr French | 0.0777 | 0.0744 | 0.0820 | **PASS** |
| vi Vietnamese | 0.1313 | 0.1755 | 0.1953 | **PASS** |
| th Thai | 0.1632 | 0.2079 | 0.2586 | fail |
| hi Hindi | 0.4184 | 0.3376 | 0.3407 | fail |
| en English | 0.5867 | 0.4389 | 0.4366 | fail |
| fa Persian | 0.7027 | 0.8900 | 1.0907 | fail |
| ar Arabic | 0.6168 | 0.6872 | 0.7066 | fail |
| ur Urdu | 0.7967 | 0.8109 | 0.7711 | fail |
| ru Russian | 0.8025 | 0.8157 | 0.8006 | fail |

**Overall: 3 passed, mean CER 0.4714. Verdict FAIL.** Scrubbed per-row data in
`eval/BENCHMARK_V9_PENAL_1024.json`.

Persian at 1.0907 is the worst number in the campaign for any language at any
setting, and it is the third consecutive arm in which Persian has degraded under
a larger budget: 0.7027 at v5, 0.8900 at v8, 1.0907 here. Urdu at 0.7711
improved, and Thai and Hindi are identical to four decimals at both budgets,
which is the signature of answers that terminate well inside the smaller
budget. Persian's only move was to get longer and more wrong.

So the step count joins the five spent levers, and the falsification runs in the
direction nobody wanted: more optimiser passes on this corpus does not recover
what the row expansion cost, it costs more. v5 remains the best model in the
campaign on both counts, 4 passes and 0.4682, and every comparison in this file
should be graded against it.

Standing after nine held-out runs across six levers: 4 of 10 languages pass at
CER 0.25, and the verifier says FAIL. The two changes that ever moved the
scoreboard were the EOS stop-rule fix and the character-balanced corpus, both of
which predate this stretch of work. Budget, penalties, prompt wording, character
volume, the row count and the step count have each been declared, measured and
closed. What binds the remaining six is not reachable from this trainer: Hindi's
corpus is rendered text rather than handwriting, English has 74 paragraph rows
and no paragraph-form source to acquire, five languages are empty directories,
and Russian and Persian hold 12.8M and 3.5M characters and saturate the cap
without moving -- capacity for those scripts, which a flag cannot fix.

The benchmark itself is the durable artifact of this campaign. It is stable
across nine runs and every configuration in it, its bar has never moved, the
levers it tested were declared before they were measured, and it has now
correctly reported FAIL more times than it has reported PASS. That is what a
verifier is for.


---

## The vision read path: the lens itself was frozen (declared 2026-10-11)

The previous section closes by saying what binds the remaining six is "not
reachable from this trainer" and names capacity for scripts as the cause. That
claim is now audited rather than asserted, and the audit found one surface it
was wrong about.

`LORA_TARGETS` in train_lora.py has always been an LM-side naming list:
`self_attn.{q,k,v,out}_proj`, `conv.{in,out}_proj`, `feed_forward.w{1,2,3}`.
LFM2.5-VL-3B's vision stack is a Siglip2 encoder, and Siglip2 names its
projections differently: `mlp.fc1`/`fc2`, `embeddings.patch_embedding`, and
`multi_modal_projector.linear_{1,2}`. PEFT matches `target_modules` by name
suffix, so the interesting fact is not that the vision list was absent but
*which* parts of the vision stack the LM-side names accidentally reached.
Building the model exactly as training builds it and inspecting the result:

| component | modules wrapped by LoRA |
|---|---|
| vision_tower.encoder.layers.*.self_attn.{q,k,v,out}_proj | 432 (all 27 layers) |
| vision_tower.encoder.layers.*.mlp.{fc1,fc2} | **0** |
| vision_tower.embeddings.patch_embedding | **0** |
| multi_modal_projector.linear_{1,2} | **0** |

So vision *attention* has been trained in every run of this campaign, and the
57 modules that make up the rest of the read path have not. The most
consequential of them is `multi_modal_projector`: it is the only thing between
the Siglip2 encoder and the language model, which means every Cyrillic,
Arabic, Persian and Devanagari glyph this campaign has ever tried to read was
handed to the LM through the base model's fixed lens, and no gradient has ever
reached that lens. "Capacity for those scripts" is the sentence the census
used; this is the mechanism that sentence was gesturing at.

Declared before any number exists: adapter_v10 is adapter_v7's exact
configuration -- 1,000 optimiser steps x 8 accumulation, lr 2e-4, rank 16,
alpha 32, `--per-lang-chars 200000`, save every 100, the same IAM corpus with
en's restored verifier -- with one change, the five vision target suffixes
added to the list. v10 against v7 therefore changes exactly one thing, the same
standard v8 and v9 were held to. Both held-out arms are pre-declared: 128
tokens with the `pen1.2+ngram3` penalties, and 1024 tokens with the same.

What would count as a result, fixed here first so it cannot be moved after.
The declared budget is 128 tokens with penalties. At that budget the best
number each language has ever recorded is: en 0.5380, ru 0.9111, fa 0.7751,
ar 0.7110, hi 0.3376, ur 0.7711, th 0.1632, vi 0.1945. Any of ru below 0.90,
fa below 0.75, ar below 0.71, or hi below 0.30 would be the first movement
those languages have shown to any lever. Any fifth language crossing 0.25
would be the first new pass since the EOS fix. And th returning below 0.1852
would undo the damage v9 did to it.

The risks are named in advance too, and they are not small. This is the first
change in the campaign that trains the vision tower's non-attention layers,
which triples the trainable parameter count from 28.4M to 662 tensors' worth
and adds optimizer state; the smoke test put peak VRAM at 11.09 GB against 16
GB cards, so it fits, but the margin is thinner than any prior run. More
seriously, the fixed lens may be fixed for a reason: the base model spent its
pretraining aligning Siglip2 features to the LM's embedding space through that
projector, and a rank-16 adapter started from noise may tear that alignment
down faster than it learns anything, which would show up as the passing
languages getting worse while the failing ones stay flat. If that happens, the
honest reading is that the lens was already good and the capacity constraint
lives somewhere the adapter cannot reach, not that the lens needed more steps.
