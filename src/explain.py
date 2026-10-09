"""Per-field attribution for an extraction result.

Given the OCR text and the BIO tags the tagger produced, this explains which
source tokens each extracted field came from and how confident the model was
about it. That lets the UI say "we read the dose from here" and lets a human
see at a glance which field is weakest and worth double-checking.

The grouping here mirrors PrescriptionSchedule.from_ner, which stays the single
source of truth for what the values actually are. This module only adds the
"where did it come from" layer, deliberately without changing the contract.
"""

from __future__ import annotations

from contracts.frequency import normalize_frequency

# Group label -> DrugEntry field it feeds.
FIELD_MAP = {
    "DRUG": "drug",
    "DOSE": "dose",
    "UNIT": "dose_unit",
    "ROUTE": "route",
    "FREQ": "frequency",
    "DURATION": "duration_days",
}


def group_by_field(text: str, tags: list[tuple[str, float]]) -> list[dict]:
    """Group BIO tags into one dict of fields per drug.

    A ``B-DRUG`` token starts a new group, matching how ``from_ner`` splits
    drugs. Each field maps to a list of ``(token, confidence)`` pairs so the
    attribution needs no fragile re-matching later. Groups are returned in the
    same order ``from_ner`` emits them, so index i here corresponds to
    ``schedule.drugs[i]``.
    """
    words = text.split()
    groups: list[dict] = []
    current: dict | None = None

    for word, (tag, conf) in zip(words, tags):
        if tag == "O":
            continue
        prefix, _, label = tag.partition("-")
        if label == "DRUG" and prefix == "B":
            current = {}
            groups.append(current)
        if current is None:
            continue
        current.setdefault(label, []).append((word, float(conf)))

    return groups


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def explain_field(pairs: list[tuple[str, float]]) -> dict:
    confidences = [c for _, c in pairs]
    return {
        "source": " ".join(t for t, _ in pairs),
        "confidence": _mean(confidences) or 0.0,
        "token_count": len(pairs),
    }


def explain(text: str, tags: list[tuple[str, float]]) -> list[dict]:
    """Return per-field attribution for every drug found in the text.

    Each drug's entry lists, per field, the source tokens and their mean token
    confidence, plus which field is weakest so a caller can flag it. Fields the
    tagger did not populate are omitted, so an absent field is visible as
    absent rather than silently zero.
    """
    explained: list[dict] = []

    for group in group_by_field(text, tags):
        fields: dict[str, dict] = {}
        for label, pairs in group.items():
            field_name = FIELD_MAP.get(label)
            if field_name is None:
                continue
            fields[field_name] = explain_field(pairs)

        if "frequency" in fields:
            fields["frequency"]["normalized"] = normalize_frequency(
                fields["frequency"]["source"]
            )

        scored = [
            (name, info["confidence"])
            for name, info in fields.items()
        ]
        weakest = min(scored, key=lambda kv: kv[1])[0] if scored else None
        mean_confidence = _mean([c for _, c in scored])
        explained.append(
            {
                "fields": fields,
                "weakest_field": weakest,
                "field_confidence": mean_confidence,
            }
        )
    return explained
