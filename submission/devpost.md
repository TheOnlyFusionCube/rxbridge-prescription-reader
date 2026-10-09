# RxBridge

**Photograph a prescription. Hear your medicines, in your own language.**

RxBridge turns a photo of a medical prescription into a spoken, structured
medication schedule for patients who cannot read the writing on their own
prescription — because the handwriting is illegible, the form is in a language
they do not read, or they have low literacy or low vision.

It runs on a phone, recognition needs no cloud round trip, and it never gives
medical advice: it only reads back what the prescription says, so the patient
always follows their doctor and pharmacist.

---

## Problem Statement

A prescription is the single most important instruction a patient receives for
their own body, and a large share of patients cannot act on it.

In the countries we tested against, forms like Cuba's `RECETA MEDICA`
(Ministerio de Salud Pública model 53-05-04) are pre-printed grids filled in by
hand in blue ballpoint. The printed scaffolding — form number, "MINSAP",
"Hospitales y Policlinicos", "FECHA", "HISTORIA CLÍNICA" — is machine-readable,
but the drug name, the dose and the frequency are handwritten, and that
handwriting is unreadable to a machine without a model trained for it.

Real photograph evidence, curated with vision and scored (see `EVIDENCE.md`):

| engine | result on 4 real photos |
|---|---|
| docTR 1.0.1 (chosen) | 0.3076 char-F1 — reads the printed scaffolding at ~0.99, fails on the cursive |
| tesseract 4.1.1 | similar, weaker |
| EasyOCR 1.7.2 | degrades further on degraded photos and French |

docTR reads printed scaffolding essentially perfectly (`RECETA MEDICA`,
`MINSAP`, `No.:89`, `8756014`, dates, numerals) and fails on cursive
handwriting — `Clotrimazol (500 mg)` comes back as `Mletzina 20`.

The consequence is not academic. Medication errors caused by unreadable
instructions are a documented cause of preventable harm, and the patients most
affected are the ones who already have least access to interpreters, pharmacist
counselling, or a family member who can translate. A printed English
prescription handed to a Spanish-only reader is no more usable than a
handwritten one handed to a low-vision patient.

RxBridge combines three things: line detection that handles a real phone
photograph of a real form, a slot-tagger that turns OCR text into
drug / dose / frequency / duration, and speech plus a pictogram day-grid that
require no reading at all.

## Solution Overview

Five stages, each independently verifiable:

```
phone photo
    │
    ▼
[1] docTR  ─────────────  detect text lines, recognise printed content
    │                       (in-memory only: no disk write, no cloud call)
    ▼
[2] DistilBERT ──────────  token-classification slot tagger, trained by us
    │                       DRUG / DOSE / UNIT / ROUTE / FREQ / DURATION
    ▼
[3] contracts/schema.py ─  group BIO spans into a validated PrescriptionSchedule
    │                       (69 tests; JSON-Schema lossless round trip)
    ▼
[4] normalise ───────────  "twice daily" → times_of_day [morning, night]
    │                       "q8h" → every 8 hours; BID/TID/QID/PRN; ES/FR
    ▼
[5] Next.js PWA ─────────  pictogram day-grid + Web Speech read-aloud
                            (target language chosen by the patient)
```

Two honest fallbacks keep the service usable rather than broken:

- **Tagger fallback.** The learned DistilBERT occasionally emits `I-DRUG` with no
  opening `B-DRUG`; the contract drops that by design. When it happens,
  `FallbackTagger` retries with a rule-based lexicon, and the response reports
  which answered (`tagger: "distilbert+lexicon" | "lexicon"`).
- **Unreadable is a result, not a crash.** An illegible or non-prescription
  image returns HTTP 422 with `status: "unreadable"` and a friendly message —
  never a 500, never a traceback.

Verified end to end on a real printed prescription: 20/20 OCR lines at
0.87–0.99 confidence, 3/3 drugs extracted with dose, frequency, day-grid times
and duration.

## Key Features

1. **Spoken schedule in the patient's own language.** Pick the language, press
   Read aloud. The summary is built from the extracted schedule and spoken with
   the browser's Web Speech API, so no cloud TTS call is needed.
2. **Pictogram day-grid.** Four columns (Morning / Midday / Evening / Night) with
   a pill glyph and a large numeral per dose, so a patient who cannot read a word
   still knows how many pills to take and when.
3. **Works from a phone.** Single-screen, mobile-first PWA: installable,
   `display: standalone`, 108 kB first load, statically prerendered.
4. **Never invents medical advice.** Footer disclaimer plus a scope lock: it only
   reads back the prescription. If the OCR cannot read a line, that line is not
   in the output.
5. **Per-line confidence, shown to the patient.** "What we read from the photo"
   lists every detected line with a certainty percentage, so a low-confidence
   reading is visible rather than hidden.
6. **Bilingual extraction.** Drug names and directions are recognised in English
   and Spanish prescriptions, and the spoken output follows the patient's chosen
    language rather than the document's.

## Follow-up on handwritten input (measured, not integrated)

Zero-shot `LiquidAI/LFM2.5-VL-3B` (6.5 GB bf16 on one RTX 5060 Ti) was tested on
the same visually-verified real photographs. It read the blue-ink form exactly
`Clotrimazol (500mg)`, confirmed against the pixels by eye, where docTR produced
`Mletzina 20` and the synthetic-trained TrOCR scored CER 0.9486. It also
hallucinated on a red-ink form and misread a dose on a third, so it is not
reliable enough to ship and is deliberately excluded from the pipeline. It is the
strongest candidate for a fine-tuned handwriting stage, and it fits the hardware
with roughly 9 GB of headroom.

## Measuring that the model reads rather than memorises

A metric can be high for the wrong reason, so the extraction model's headline
number was checked against a control. The same DistilBERT token classifier was
trained twice on an identical split of 12,000 prescription sentences: once with
the real BIO tags, and once with every gold tag replaced by a random draw from
the corpus label distribution, leaving the text untouched.

| model | entity-level F1 (seqeval, n=960 held out) |
|---|---|
| trained on real labels | **0.9367** |
| trained on shuffled labels (control) | **0.0000** |

The control collapses to exactly zero, so the 0.937 is measuring the
token-to-label mapping rather than an artifact of the data generator. This is
the same measure-the-signal-not-the-artifact check that the previous edition's
first-place entry relied on. Full report in `eval/REPORT.md`; reproduce with
`python3 scripts/eval_final.py`.

## Technologies Used

- **docTR** — text detection and recognition of the prescription image
- **PyTorch / transformers** — DistilBERT token classification, trained by us
- **FastAPI** — extraction service, TDD with 12 API tests
- **Next.js 15 / React 19 / TypeScript** — PWA frontend
- **Pydantic v2** — validated contract with exported JSON Schema, 69 tests
- **pytest** — 81 tests total
- **Playwright** — browser-level end-to-end test and demo recording
- **4× RTX 5060 Ti (64 GB VRAM)** — model training and inference
- **cloudflared** — public tunnel for the deployed backend

## Target Users

**Primary: patients who cannot read their own prescription.**

- Low-literacy adults handed a handwritten form.
- Elderly patients handed a printed prescription in a language they do not read.
- Language-barrier patients who would otherwise need a family member to
  translate dose and timing — the part that matters most and gets garbled most.
- Low-vision patients who can see a pictogram and hear speech but not small
  printed directions.

**Secondary: community pharmacists and clinic staff** verifying that a patient
understood the label before dispensing.

---

## Links

- **Live demo:** https://same-trials-seed-jewellery.trycloudflare.com
- **API:** https://template-pair-microphone-contractor.trycloudflare.com (POST /extract)
- **Repository:** https://github.com/TheOnlyFusionCube/rxbridge-prescription-reader
- **Demo video:** `submission/demo.mp4` (12.5s)

Verified end to end from outside the cluster on the day of submission: the frontend
returns 200 and its JS references the live API, and POSTing a real printed
prescription photo to the public API returns `status: ok` with 3 drugs
(Amoxicillin 500 mg twice daily, Paracetamol 500 mg every 8 hours, Omeprazole
20 mg once daily), tagged `distilbert+lexicon`.

The tunnel URLs are ephemeral (`trycloudflare.com` quick tunnels); if a link
shows a 530 the tunnel process needs restarting rather than the app.
