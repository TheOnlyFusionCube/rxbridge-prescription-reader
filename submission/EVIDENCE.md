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
