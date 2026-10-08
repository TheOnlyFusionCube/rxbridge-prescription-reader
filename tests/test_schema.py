"""Contract tests for the RxBridge frozen data contract.

These tests were written BEFORE the implementation (strict red-green TDD).
Every rule in the task specification has at least one assertion here.

Row shape consumed by ``PrescriptionSchedule.from_ner`` (the token-level BIO
output of the extraction model owned by another task)::

    {
        "text": str,            # token surface form, e.g. "Amoxicillin"
        "tag": str,             # BIO tag, e.g. "B-DRUG", "I-DOSE", "O"
        "confidence": float,    # token softmax for the chosen tag, 0..1
        "start": int | None,    # optional char offset, inclusive
        "end": int | None,      # optional char offset, exclusive
    }
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from contracts.frequency import normalize_frequency
from contracts.schema import DrugEntry, PrescriptionSchedule

REPO_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO_ROOT / "contracts" / "rx_schedule.schema.json"

ALLOWED_TIMES = ["morning", "noon", "evening", "night"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_entry(**overrides) -> DrugEntry:
    base = dict(
        drug="Amoxicillin",
        dose="500",
        dose_unit="mg",
        route="oral",
        frequency="twice daily",
        times_of_day=["morning", "evening"],
        duration_days=7,
        confidence=0.9,
        source_text="Amoxicillin 500 mg twice daily",
        span=(0, 31),
    )
    base.update(overrides)
    return DrugEntry(**base)


def make_ok(**overrides) -> PrescriptionSchedule:
    base = dict(status="ok", drugs=[make_entry()], overall_confidence=0.9)
    base.update(overrides)
    return PrescriptionSchedule(**base)


def tok(text, tag, conf=0.9, start=None, end=None) -> dict:
    return {"text": text, "tag": tag, "confidence": conf, "start": start, "end": end}


# --------------------------------------------------------------------------- #
# times_of_day
# --------------------------------------------------------------------------- #
def test_times_of_day_canonical_order_and_dedup():
    entry = make_entry(times_of_day=["night", "morning", "morning", "evening"])
    assert entry.times_of_day == ["morning", "evening", "night"]


def test_times_of_day_empty_is_allowed():
    assert make_entry(times_of_day=[]).times_of_day == []


def test_times_of_day_full_set_in_canonical_order():
    entry = make_entry(times_of_day=list(reversed(ALLOWED_TIMES)))
    assert entry.times_of_day == ALLOWED_TIMES


def test_times_of_day_rejects_unknown_value():
    with pytest.raises(ValidationError):
        make_entry(times_of_day=["morning", "afternoon"])


def test_times_of_day_rejects_unknown_value_alone():
    with pytest.raises(ValidationError):
        make_entry(times_of_day=["midnight"])


def test_times_of_day_is_case_and_whitespace_normalized():
    entry = make_entry(times_of_day=[" Morning ", "NIGHT"])
    assert entry.times_of_day == ["morning", "night"]


# --------------------------------------------------------------------------- #
# confidence / overall_confidence range
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_confidence_accepts_in_range(value):
    assert make_entry(confidence=value).confidence == value


@pytest.mark.parametrize("value", [1.0001, 1.5, 100.0])
def test_confidence_above_one_raises(value):
    with pytest.raises(ValidationError):
        make_entry(confidence=value)


@pytest.mark.parametrize("value", [-0.0001, -1.0, -50.0])
def test_confidence_below_zero_raises(value):
    with pytest.raises(ValidationError):
        make_entry(confidence=value)


@pytest.mark.parametrize("value", [0.0, 0.42, 1.0])
def test_overall_confidence_accepts_in_range(value):
    assert make_ok(overall_confidence=value).overall_confidence == value


@pytest.mark.parametrize("value", [1.5, -0.5])
def test_overall_confidence_out_of_range_raises(value):
    with pytest.raises(ValidationError):
        make_ok(overall_confidence=value)


def test_overall_confidence_none_is_allowed():
    assert make_ok(overall_confidence=None).overall_confidence is None


# --------------------------------------------------------------------------- #
# status / drugs consistency
# --------------------------------------------------------------------------- #
def test_default_status_is_unreadable_with_no_drugs():
    sched = PrescriptionSchedule()
    assert sched.status == "unreadable"
    assert sched.drugs == []


def test_unreadable_rejects_nonempty_drugs():
    with pytest.raises(ValidationError):
        PrescriptionSchedule(status="unreadable", drugs=[make_entry()])


def test_default_status_with_drugs_raises():
    # status defaults to "unreadable"; supplying drugs without flipping status
    # to "ok" is inconsistent and must be rejected, not silently corrected.
    with pytest.raises(ValidationError):
        PrescriptionSchedule(drugs=[make_entry()])


def test_ok_requires_nonempty_drugs():
    with pytest.raises(ValidationError):
        PrescriptionSchedule(status="ok", drugs=[], overall_confidence=0.5)


def test_ok_requires_overall_confidence():
    with pytest.raises(ValidationError):
        PrescriptionSchedule(status="ok", drugs=[make_entry()], overall_confidence=None)


def test_ok_valid_schedule_passes():
    sched = make_ok()
    assert sched.status == "ok"
    assert len(sched.drugs) == 1
    assert sched.overall_confidence == 0.9


def test_status_rejects_unknown_literal():
    with pytest.raises(ValidationError):
        PrescriptionSchedule(status="maybe")


# --------------------------------------------------------------------------- #
# DrugEntry field rules
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
def test_drug_name_must_be_nonempty_after_strip(bad):
    with pytest.raises(ValidationError):
        make_entry(drug=bad)


def test_drug_name_is_stripped():
    assert make_entry(drug="  Amoxicillin \n").drug == "Amoxicillin"


def test_entry_defaults():
    entry = DrugEntry(
        drug="X", dose="1", dose_unit="mg", frequency="once daily",
        confidence=0.5, source_text="X 1 mg",
    )
    assert entry.route is None
    assert entry.times_of_day == []
    assert entry.duration_days is None
    assert entry.span is None


def test_span_coerces_list_to_tuple():
    assert make_entry(span=[3, 9]).span == (3, 9)


# --------------------------------------------------------------------------- #
# explanation
# --------------------------------------------------------------------------- #
def test_explanation_none_is_allowed():
    assert PrescriptionSchedule().explanation is None


def test_explanation_nonempty_is_allowed():
    assert make_ok(explanation="Take with food").explanation == "Take with food"


@pytest.mark.parametrize("bad", ["", "   ", "\n"])
def test_explanation_empty_when_set_raises(bad):
    with pytest.raises(ValidationError):
        make_ok(explanation=bad)


# --------------------------------------------------------------------------- #
# defaults for language / warnings
# --------------------------------------------------------------------------- #
def test_language_defaults_and_warnings_default():
    sched = PrescriptionSchedule()
    assert sched.language == "en"
    assert sched.target_language == "en"
    assert sched.warnings == []


# --------------------------------------------------------------------------- #
# from_ner
# --------------------------------------------------------------------------- #
def test_from_ner_no_b_drug_is_unreadable():
    rows = [tok("500", "B-DOSE"), tok("mg", "I-DOSE"), tok("BID", "B-FREQ")]
    sched = PrescriptionSchedule.from_ner(rows, language="es")
    assert sched.status == "unreadable"
    assert sched.drugs == []
    assert sched.language == "es"
    assert sched.target_language == "en"


def test_from_ner_empty_rows_is_unreadable():
    assert PrescriptionSchedule.from_ner([], language="en").status == "unreadable"


def test_from_ner_single_drug_fields_and_confidence():
    rows = [
        tok("Amoxicillin", "B-DRUG", 0.9, start=0, end=11),
        tok("500", "B-DOSE", 0.8, start=12, end=15),
        tok("mg", "I-DOSE", 0.6, start=16, end=18),
        tok("BID", "B-FREQ", 1.0, start=19, end=22),
    ]
    sched = PrescriptionSchedule.from_ner(rows, language="en")
    assert sched.status == "ok"
    assert len(sched.drugs) == 1
    entry = sched.drugs[0]
    assert entry.drug == "Amoxicillin"
    assert entry.dose == "500"
    assert entry.dose_unit == "mg"
    assert entry.frequency == "twice daily"          # normalized via normalize_frequency
    assert entry.source_text == "Amoxicillin 500 mg BID"
    assert entry.span == (0, 22)
    expected = (0.9 + 0.8 + 0.6 + 1.0) / 4
    assert entry.confidence == pytest.approx(expected)
    assert sched.overall_confidence == pytest.approx(expected)


def test_from_ner_duration_route_and_multi_drug():
    rows = [
        tok("Amoxicillin", "B-DRUG", 0.9),
        tok("500", "B-DOSE", 0.9),
        tok("mg", "I-DOSE", 0.9),
        tok("BID", "B-FREQ", 0.9),
        tok("7", "B-DURATION", 0.9),
        tok("days", "I-DURATION", 0.9),
        tok("oral", "B-ROUTE", 0.9),
        tok("Ibuprofen", "B-DRUG", 0.5),
        tok("200", "B-DOSE", 0.5),
        tok("mg", "I-DOSE", 0.5),
        tok("TID", "B-FREQ", 0.5),
    ]
    sched = PrescriptionSchedule.from_ner(rows, language="en")
    assert sched.status == "ok"
    assert len(sched.drugs) == 2
    first, second = sched.drugs
    assert first.duration_days == 7
    assert first.route == "oral"
    assert second.drug == "Ibuprofen"
    assert second.frequency == "three times daily"
    assert second.duration_days is None
    assert second.route is None
    expected_overall = (0.9 + 0.5) / 2
    assert sched.overall_confidence == pytest.approx(expected_overall)


def test_from_ner_propagates_languages():
    rows = [tok("Amoxicillin", "B-DRUG"), tok("500", "B-DOSE"), tok("mg", "I-DOSE")]
    sched = PrescriptionSchedule.from_ner(rows, language="es", target_language="fr")
    assert sched.language == "es"
    assert sched.target_language == "fr"


def test_from_ner_ignores_stray_i_tag_before_any_b_drug():
    rows = [tok("orphan", "I-DOSE"), tok("Amoxicillin", "B-DRUG", 0.7), tok("500", "B-DOSE", 0.7)]
    sched = PrescriptionSchedule.from_ner(rows, language="en")
    assert len(sched.drugs) == 1
    assert sched.drugs[0].drug == "Amoxicillin"


# --------------------------------------------------------------------------- #
# JSON round trip
# --------------------------------------------------------------------------- #
def test_json_round_trip_is_lossless_ok():
    sched = make_ok()
    restored = PrescriptionSchedule.model_validate(json.loads(sched.model_dump_json()))
    assert restored == sched


def test_json_round_trip_is_lossless_unreadable():
    sched = PrescriptionSchedule(status="unreadable", language="fr", warnings=["blurry"])
    restored = PrescriptionSchedule.model_validate(json.loads(sched.model_dump_json()))
    assert restored == sched


def test_json_round_trip_preserves_span_tuple():
    sched = make_ok(drugs=[make_entry(span=(4, 17))])
    restored = PrescriptionSchedule.model_validate(json.loads(sched.model_dump_json()))
    assert restored.drugs[0].span == (4, 17)


# --------------------------------------------------------------------------- #
# exported JSON Schema file
# --------------------------------------------------------------------------- #
def test_json_schema_file_is_valid_and_matches_model():
    assert SCHEMA_PATH.exists(), f"missing exported schema: {SCHEMA_PATH}"
    data = json.loads(SCHEMA_PATH.read_text())
    assert data == PrescriptionSchedule.model_json_schema()
    assert "PrescriptionSchedule" in data["properties"]
    assert "DrugEntry" in data["$defs"]


# --------------------------------------------------------------------------- #
# normalize_frequency
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("BID", "twice daily"),
        ("bid", "twice daily"),
        (" TID ", "three times daily"),
        ("tid", "three times daily"),
        ("QID", "four times daily"),
        ("qid", "four times daily"),
        ("PRN", "as needed"),
        ("prn", "as needed"),
        ("qd", "once daily"),
        ("QD", "once daily"),
        ("q8h", "every 8 hours"),
        ("Q8H", "every 8 hours"),
        ("q 8 h", "every 8 hours"),
        ("q6h", "every 6 hours"),
        ("q12h", "every 12 hours"),
        ("twice a day", "twice daily"),
    ],
)
def test_normalize_frequency_known(raw, expected):
    assert normalize_frequency(raw) == expected


def test_normalize_frequency_unknown_is_returned_stripped():
    assert normalize_frequency("  whenever required  ") == "whenever required"


@pytest.mark.parametrize("raw", ["", "   "])
def test_normalize_frequency_blank_returns_empty(raw):
    assert normalize_frequency(raw) == ""
