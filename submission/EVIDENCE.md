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
