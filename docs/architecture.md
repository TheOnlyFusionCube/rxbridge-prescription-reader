# RxBridge architecture

A photograph of a prescription goes in; a spoken, structured schedule comes out.
The pipeline is five stages, each independently testable and independently
replaceable.

```
photo
  │
  ▼
┌─────────────┐   detect text lines, read them
│  OCR        │   src/ocr.py  ·  docTR
│  docTR      │   in-memory only; nothing written to disk
└──────┬──────┘
       │  lines + per-line confidence
       ▼
┌─────────────┐   token → slot labels
│  slot tag   │   src/pipeline.py  ·  DistilBertTagger (trained by us)
│  DistilBERT │   falls back to LexiconTagger when the learned
└──────┬──────┘   model emits an unopened I- span
       │  BIO tags + token confidences
       ▼
┌─────────────┐   tags → validated schedule
│  contract   │   contracts/schema.py  ·  PrescriptionSchedule.from_ner
│  pydantic   │   the single source of truth for what values exist
└──────┬──────┘
       │  structured drugs + confidences
       ▼
┌─────────────┐   "BID" → ["morning","night"], "q8h" → every 8 hours
│  normalise  │   contracts/frequency.py  ·  en / es / fr
└──────┬──────┘
       │
       ▼
┌─────────────┐   pictogram day-grid + Web Speech read-aloud
│  interface  │   web/  ·  Next.js PWA, patient's own language
└─────────────┘
```

## Why the tags are grouped the way they are

The slot tagger labels tokens, not fields — it says `Amoxicillin` is `B-DRUG`
and `500` is `B-DOSE`, not "this prescription has a drug with this dose."
`from_ner` turns those tags into drug records by treating a `B-DRUG` as the start
of a new group and letting every following `I-` extend it. That is the join
between "a model saw words" and "a patient has instructions."

Confidence travels per token and averages up per field, so the weakest field is
always identifiable (`src/explain.py` exposes which one).

## What is learned versus what is read

The learned component is the slot tagger. It was trained on synthetic
prescription text — 12,000 examples across printed, handwritten-corrupted and
natural-language phrasing, with three frequency shards and 100 negatives. The
synthetic-only handwriting recognizer failed on real input (CER 0.9486) and was
removed rather than shipped, which is why docTR still handles the image stage.

The reason the learned tagger can afford to work on text rather than pixels is
that docTR already recovers the printed scaffolding of a real prescription at
~0.99 confidence. The hard problem is cursive handwriting, and that gap is
stated in the write-up instead of being papered over.

## Failure behaviour

Every stage can degrade without the request failing:

| Failure | Behaviour |
|---|---|
| Bytes are not a decodable image | 422 `unreadable`, reason stated |
| Image decodes but contains no text | `unreadable`, drugs `[]`, confidence `None` |
| Learned tagger emits stray `I-` spans | lexicon fallback re-tags the same text |
| Backend unreachable | frontend renders the unreadable state, never a blank screen |

An unreadable prescription is a result, not a crash. There is no path where the
pairing of a stack trace and an image reaches the UI.

## Trust boundaries

The boundary is at the upload. Everything after it — OCR output, tagger rows,
schedule JSON — is internal type-checked data. The only untrusted input is the
image bytes and the declared content type, so those are validated at the edge;
nothing downstream re-validates.

## What to read first

| To understand | Read |
|---|---|
| the data shape both ends agree on | `contracts/README.md`, `contracts/rx_schedule.schema.json` |
| what the model learned and whether it's real | `eval/REPORT.md` |
| honest numbers, including the failure | `submission/EVIDENCE.md` |
| how to run it | `README.md`, `CONTRIBUTING.md` |
