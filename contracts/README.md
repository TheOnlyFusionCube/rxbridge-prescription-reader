# RxBridge prescription contract

This directory is the frozen data contract between the extraction backend and the
Next.js frontend: `contracts/schema.py` defines the pydantic v2 models
`DrugEntry` and `PrescriptionSchedule` (one `DrugEntry` per medication line, one
`PrescriptionSchedule` per photographed prescription), `contracts/frequency.py`
exposes `normalize_frequency(raw)` which turns clinical shorthand such as `BID`
into the canonical phrase `twice daily`, and `contracts/rx_schedule.schema.json`
is the auto-exported JSON Schema of `PrescriptionSchedule` (with `DrugEntry` in
`$defs`) that the frontend generates its TypeScript types from — it is produced
by `PrescriptionSchedule.model_json_schema()` and must stay in sync with the
models. Both models forbid unknown fields, every `span` is a two-integer
`[start, end]` character offset, all probabilities live in `[0, 1]`, and
`PrescriptionSchedule.model_dump_json()` round-trips losslessly through
`model_validate(json.loads(...))`.

## Example (`status: "ok"`)

```json
{
  "status": "ok",
  "drugs": [
    {
      "drug": "Amoxicillin",
      "dose": "500",
      "dose_unit": "mg",
      "route": "oral",
      "frequency": "twice daily",
      "times_of_day": ["morning", "evening"],
      "duration_days": 7,
      "confidence": 0.94,
      "source_text": "Amoxicillin 500 mg BID x 7 days",
      "span": [0, 33]
    }
  ],
  "language": "es",
  "target_language": "en",
  "explanation": "Take one capsule twice a day for seven days.",
  "overall_confidence": 0.94,
  "warnings": []
}
```

## Rules the models enforce

- `times_of_day` entries must be one of `morning`, `noon`, `evening`, `night`
  (compared case-insensitively); they are de-duplicated and emitted in that
  canonical order, and an empty list is allowed.
- `confidence` and `overall_confidence` must be in `[0, 1]`; an out-of-range
  value raises `ValidationError` rather than being silently clamped, because a
  malformed probability is a signal, not noise.
- `status == "unreadable"` implies `drugs == []`. We **raise** `ValidationError`
  on inconsistent input (an `unreadable` schedule carrying drugs) instead of
  silently dropping them; the `from_ner` path is what legitimately produces the
  `unreadable`/empty result.
- `status == "ok"` implies `drugs` is non-empty, every `drug` is non-empty after
  stripping, and `overall_confidence` is not `None`.
- `explanation`, when set, must be non-empty after stripping.

## `PrescriptionSchedule.from_ner(rows, language, target_language="en")`

`rows` is the token-level BIO output of the slot-tagging model (owned by a
separate task): each row is
`{"text": str, "tag": str, "confidence": float, "start": int | None, "end": int | None}`
where `tag` is a `B-`/`I-` label over `DRUG`, `DOSE`, `FREQ`, `DURATION`,
`ROUTE` (anything else, including `O`, is ignored). A `B-DRUG` tag opens a new
`DrugEntry`; `I-*` tags extend the current one. `drug` is the joined `DRUG`
span, `dose`/`dose_unit` are split off the `DOSE` span, `frequency` is the
`FREQ` span passed through `normalize_frequency`, `duration_days` is the first
integer found in the `DURATION` span (else `None`), `route` is the `ROUTE` span
(else `None`), `source_text` is the joined tokens, `span` is
`(min(start), max(end))` when offsets are present, `confidence` is the mean
token confidence across the entry, and `overall_confidence` is the mean of the
entry confidences. If no `B-DRUG` tag appears, the result is
`status="unreadable", drugs=[]`.
