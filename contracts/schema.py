"""Frozen data contract for RxBridge prescription extraction.

Pydantic v2 models shared by the extraction backend and the Next.js frontend.
See ``contracts/README.md`` for the prose contract and a JSON example.

Token row shape accepted by :meth:`PrescriptionSchedule.from_ner`::

    {"text": str, "tag": str, "confidence": float, "start": int | None, "end": int | None}

where ``tag`` is a BIO label over DRUG, DOSE, FREQ, DURATION and ROUTE
(e.g. ``"B-DRUG"``, ``"I-DOSE"``, ``"O"``).
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from contracts.frequency import normalize_frequency

ALLOWED_TIMES_OF_DAY: tuple[str, ...] = ("morning", "noon", "evening", "night")

_FIELD_ORDER: tuple[str, ...] = ("DRUG", "DOSE", "FREQ", "DURATION", "ROUTE")
_FIELD_TAGS = frozenset(_FIELD_ORDER)
_DOSE_RE = re.compile(r"^([0-9]+(?:[./][0-9]+)?)\s*(.*)$")
_DURATION_RE = re.compile(r"(\d+)")


class DrugEntry(BaseModel):
    """One parsed medication line."""

    model_config = ConfigDict(extra="forbid")

    drug: str
    dose: str
    dose_unit: str
    route: str | None = None
    frequency: str
    times_of_day: list[str] = Field(default_factory=list)
    duration_days: int | None = None
    confidence: float
    source_text: str
    span: tuple[int, int] | None = None

    @field_validator("drug")
    @classmethod
    def _drug_nonempty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("drug must be non-empty after strip")
        return stripped

    @field_validator("confidence")
    @classmethod
    def _confidence_in_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be within [0, 1]")
        return value

    @field_validator("times_of_day")
    @classmethod
    def _normalize_times(cls, values: list[str]) -> list[str]:
        ordered: list[str] = []
        for item in values:
            key = item.strip().lower()
            if key not in ALLOWED_TIMES_OF_DAY:
                raise ValueError(
                    f"times_of_day entry {item!r} not in {list(ALLOWED_TIMES_OF_DAY)}"
                )
            if key not in ordered:
                ordered.append(key)
        ordered.sort(key=ALLOWED_TIMES_OF_DAY.index)
        return ordered


class PrescriptionSchedule(BaseModel):
    """Top-level extraction result for a single prescription photo."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "unreadable"] = "unreadable"
    drugs: list[DrugEntry] = Field(default_factory=list)
    language: str = "en"
    target_language: str = "en"
    explanation: str | None = None
    overall_confidence: float | None = None
    warnings: list[str] = Field(default_factory=list)

    @field_validator("overall_confidence")
    @classmethod
    def _overall_in_range(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if not 0.0 <= value <= 1.0:
            raise ValueError("overall_confidence must be within [0, 1]")
        return value

    @field_validator("explanation")
    @classmethod
    def _explanation_nonempty(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.strip():
            raise ValueError("explanation must be non-empty when set")
        return value

    @model_validator(mode="after")
    def _status_consistency(self) -> "PrescriptionSchedule":
        if self.status == "unreadable" and self.drugs:
            raise ValueError("status 'unreadable' requires drugs == []")
        if self.status == "ok":
            if not self.drugs:
                raise ValueError("status 'ok' requires at least one drug")
            for drug in self.drugs:
                if not drug.drug.strip():
                    raise ValueError("status 'ok' forbids empty drug names")
            if self.overall_confidence is None:
                raise ValueError("status 'ok' requires overall_confidence")
        return self

    @classmethod
    def from_ner(
        cls,
        rows: list[dict],
        language: str,
        target_language: str = "en",
    ) -> "PrescriptionSchedule":
        """Group token-level BIO rows into a validated schedule.

        A ``B-DRUG`` tag starts a new drug group; ``I-*`` tags extend the
        current one. Fields are joined from their token spans, the FREQ span is
        canonicalized through :func:`normalize_frequency`, and ``confidence`` is
        the mean token confidence over the group. No ``B-DRUG`` tag anywhere
        yields ``status="unreadable", drugs=[]``.
        """
        groups: list[dict[str, list[dict]]] = []
        current: dict[str, list[dict]] | None = None

        for row in rows:
            tag = str(row.get("tag", "")).strip().upper()
            if not tag or tag == "O":
                continue
            prefix, separator, label = tag.partition("-")
            if not separator or prefix not in ("B", "I") or label not in _FIELD_TAGS:
                continue
            if label == "DRUG" and prefix == "B":
                current = {field: [] for field in _FIELD_ORDER}
                groups.append(current)
            if current is None:
                continue
            current[label].append(row)

        drugs: list[DrugEntry] = []
        for group in groups:
            tokens = [token for field in _FIELD_ORDER for token in group[field]]
            drug_text = _join(group["DRUG"])
            if not drug_text:
                continue
            dose, dose_unit = _split_dose(_join(group["DOSE"]))
            explicit_unit = _join(group["UNIT"]).strip().lower()
            if explicit_unit and not dose_unit:
                dose_unit = explicit_unit
            drugs.append(
                DrugEntry(
                    drug=drug_text,
                    dose=dose,
                    dose_unit=dose_unit,
                    route=_join(group["ROUTE"]) or None,
                    frequency=normalize_frequency(_join(group["FREQ"])),
                    times_of_day=[],
                    duration_days=_parse_duration(_join(group["DURATION"])),
                    confidence=_mean_confidence(tokens),
                    source_text=_join(tokens),
                    span=_span(tokens),
                )
            )

        if not drugs:
            return cls(
                status="unreadable",
                drugs=[],
                language=language,
                target_language=target_language,
            )

        overall = sum(drug.confidence for drug in drugs) / len(drugs)
        return cls(
            status="ok",
            drugs=drugs,
            language=language,
            target_language=target_language,
            overall_confidence=overall,
        )


def _field(row: dict, *names: str, default=None):
    for name in names:
        if name in row and row[name] is not None:
            return row[name]
    return default


def _join(tokens: list[dict]) -> str:
    return " ".join(str(_field(token, "token", "text", default="")) for token in tokens).strip()


_UNIT_WORDS = {"mg", "mcg", "g", "ml", "ui", "iu", "l", "meq", "units", "unit", "mgkg"}
_DOSE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([A-Za-z]{1,5})?\s*$")


def _split_dose(text: str) -> tuple[str, str]:
    text = text.strip()
    if not text:
        return "", ""
    match = _DOSE_RE.match(text)
    if match:
        unit = (match.group(2) or "").lower()
        return match.group(1), unit if unit in _UNIT_WORDS else ""
    head, _, rest = text.partition(" ")
    if rest and rest.lower().strip(".,;") in _UNIT_WORDS:
        return head, rest.lower().strip(".,;")
    return head, ""


def _parse_duration(text: str) -> int | None:
    match = _DURATION_RE.search(text)
    return int(match.group(1)) if match else None


def _mean_confidence(tokens: list[dict]) -> float:
    if not tokens:
        return 0.0
    return sum(float(_field(token, "prob", "confidence", default=0.0)) for token in tokens) / len(tokens)


def _span(tokens: list[dict]) -> tuple[int, int] | None:
    starts: list[int] = []
    ends: list[int] = []
    for token in tokens:
        span = _field(token, "span")
        if isinstance(span, (list, tuple)) and len(span) == 2:
            starts.append(int(span[0]))
            ends.append(int(span[1]))
            continue
        start = _field(token, "start")
        end = _field(token, "end")
        if start is not None and end is not None:
            starts.append(int(start))
            ends.append(int(end))
    if starts and ends:
        return (min(starts), max(ends))
    return None
