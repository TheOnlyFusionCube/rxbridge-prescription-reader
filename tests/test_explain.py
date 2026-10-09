"""Tests for src.explain — per-field attribution.

The contract (PrescriptionSchedule) owns what the values are; this module owns
where they came from and how confident the model was. These tests pin the
grouping (it must mirror PrescriptionSchedule.from_ner) and the confidence (it
must be the mean of the actual token probabilities, not invented).

``tags`` is the ``list[tuple[str, float]]`` that ``SlotTagger.tag()`` returns:
one ``(tag, confidence)`` pair per whitespace token, positionally aligned with
``text.split()``.
"""

from contracts.frequency import normalize_frequency
from src.explain import FIELD_MAP, explain, group_by_field

FULL_TEXT = "Amoxicillin 500 mg oral twice daily for 10 days"
FULL_TAGS = [
    ("B-DRUG", 0.99),
    ("B-DOSE", 0.95),
    ("B-UNIT", 0.90),
    ("B-ROUTE", 0.85),
    ("B-FREQ", 0.80),
    ("I-FREQ", 0.78),
    ("O", 0.50),
    ("B-DURATION", 0.70),
    ("I-DURATION", 0.68),
]


# ------------------------------------------------------------------ grouping


def test_groups_one_dict_per_b_drug():
    groups = group_by_field(
        "Amoxicillin 500 Metformin 850",
        [("B-DRUG", 0.9), ("B-DOSE", 0.9), ("B-DRUG", 0.9), ("B-DOSE", 0.9)],
    )
    assert len(groups) == 2
    assert groups[0]["DRUG"] == [("Amoxicillin", 0.9)]
    assert groups[0]["DOSE"] == [("500", 0.9)]
    assert groups[1]["DRUG"] == [("Metformin", 0.9)]


def test_o_tags_do_not_break_a_group():
    groups = group_by_field(
        "Amoxicillin by mouth 500",
        [("B-DRUG", 0.9), ("O", 0.0), ("O", 0.0), ("B-DOSE", 0.9)],
    )
    assert groups[0]["DRUG"] == [("Amoxicillin", 0.9)]
    assert groups[0]["DOSE"] == [("500", 0.9)]


def test_i_tag_before_any_b_drug_is_dropped():
    """from_ner ignores a stray I- with no group open; this must too."""
    assert group_by_field(
        "by mouth Amoxicillin",
        [("O", 0.0), ("O", 0.0), ("I-DRUG", 0.9)],
    ) == []


def test_i_tag_continuation_extends_the_field():
    groups = group_by_field(
        "Alpha Beta 500",
        [("B-DRUG", 0.9), ("I-DRUG", 0.8), ("B-DOSE", 0.9)],
    )
    assert groups[0]["DRUG"] == [("Alpha", 0.9), ("Beta", 0.8)]


# ------------------------------------------------------------------ attribution


def test_explain_reports_source_per_field():
    out = explain(FULL_TEXT, FULL_TAGS)
    assert len(out) == 1
    fields = out[0]["fields"]
    assert set(fields) == set(FIELD_MAP.values())

    assert fields["drug"]["source"] == "Amoxicillin"
    assert fields["dose"]["source"] == "500"
    assert fields["dose_unit"]["source"] == "mg"
    assert fields["route"]["source"] == "oral"
    assert fields["frequency"]["source"] == "twice daily"
    assert fields["duration_days"]["source"] == "10 days"


def test_confidence_is_mean_of_token_probs_not_invented():
    freq = explain(FULL_TEXT, FULL_TAGS)[0]["fields"]["frequency"]
    assert abs(freq["confidence"] - 0.79) < 1e-9
    assert freq["token_count"] == 2


def test_weakest_field_is_the_lowest_confidence_one():
    weakest = explain(FULL_TEXT, FULL_TAGS)[0]["weakest_field"]
    assert weakest == "duration_days"


def test_field_confidence_is_mean_of_field_means():
    means = [0.99, 0.95, 0.90, 0.85, 0.79, 0.69]
    expected = sum(means) / len(means)
    out = explain(FULL_TEXT, FULL_TAGS)[0]["field_confidence"]
    assert abs(out - expected) < 1e-9


def test_multidrug_order_matches_schedule_order():
    out = explain(
        "Amoxicillin 500 BID Metformin 850 TID",
        [
            ("B-DRUG", 0.9), ("B-DOSE", 0.9), ("B-FREQ", 0.9),
            ("B-DRUG", 0.9), ("B-DOSE", 0.9), ("B-FREQ", 0.9),
        ],
    )
    assert [e["fields"]["drug"]["source"] for e in out] == ["Amoxicillin", "Metformin"]


def test_frequency_is_normalized_in_the_attribution():
    out = explain(FULL_TEXT, FULL_TAGS)
    normalized = out[0]["fields"]["frequency"]["normalized"]
    assert normalized == normalize_frequency("twice daily")
    assert normalized == "twice daily"


def test_weak_field_confidence_lower_than_clean_field():
    """A field drawn from low-probability tokens must score below a clean one."""
    out = explain(
        "Amoxicillin 500 mg BID",
        [
            ("B-DRUG", 0.99),
            ("B-DOSE", 0.98),
            ("B-UNIT", 0.97),
            ("B-FREQ", 0.40),
        ],
    )
    drug = out[0]["fields"]["drug"]["confidence"]
    freq = out[0]["fields"]["frequency"]["confidence"]
    assert freq < drug


def test_empty_input_returns_no_attributions():
    assert explain("", []) == []
    assert group_by_field("", []) == []


def test_absent_field_is_absent_not_zero():
    out = explain("Amoxicillin 500", [("B-DRUG", 0.9), ("B-DOSE", 0.9)])
    assert "frequency" not in out[0]["fields"]
    assert out[0]["field_confidence"] == 0.9
