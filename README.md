# RxBridge

Photograph a prescription. Hear your medicines in your own language.

RxBridge reads a photo of a medical prescription and returns a structured
medication schedule (drug, dose, frequency, duration), then renders it as a
pictogram day-grid and speaks it aloud in the patient's language. Target users
are low-literacy, elderly, or language-barrier patients.

## Architecture

```
photo -> docTR (detect + recognise) -> DistilBERT slot tagger (trained by us)
      -> PrescriptionSchedule (validated contract) -> normalise times
      -> Next.js PWA: pictogram day-grid + Web Speech
```

`src/ocr.py` decodes the upload in memory and hands docTR raw bytes, so no image
is ever written to disk. DocTR reads printed scaffolding (form titles, dates,
numerals) at ~0.99 confidence and fails on cursive handwriting; the slot tagger
recovers structure from the text that survives.

## Status

- `tests/` — 81 tests (69 contract + 12 API). The API tests import docTR, so
  they run under `.venv`, not the system Python. See `CONTRIBUTING.md`.
- `web/` — Next.js PWA, builds clean, 105-108 kB first-load JS.
- `submission/` — Devpost draft, screenshots, and the demo video.

## Measured results

See `submission/EVIDENCE.md`. The headline number is honest rather than
flattering: docTR scores a mean char-F1 of 0.3076 on four visually-verified
real prescription photographs, because they are handwritten. On a printed
prescription the full pipeline extracts 3/3 drugs with dose, frequency, times
and duration.
