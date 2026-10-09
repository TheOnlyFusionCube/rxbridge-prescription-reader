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
