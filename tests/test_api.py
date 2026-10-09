import io
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from app.main import app, ExtractResponse  # noqa: E402
from contracts.schema import PrescriptionSchedule  # noqa: E402

client = TestClient(app)


def jpeg_bytes(color=(255, 255, 255), size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="JPEG")
    return buf.getvalue()


def post_image(data: bytes, content_type: str = "image/jpeg"):
    return client.post("/extract", files={"image": ("rx.jpg", data, content_type)})


def test_health_returns_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert r.json()["tagger"] in {"distilbert", "lexicon", "distilbert+lexicon"}


def test_root_returns_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "RxBridge" in r.text


def test_extract_blank_image_is_unreadable_not_500():
    r = post_image(jpeg_bytes())
    assert r.status_code in (200, 422), f"never a 500: got {r.status_code}"
    body = r.json()
    assert body["status"] == "unreadable"
    assert body["drugs"] == []
    assert body["overall_confidence"] is None
    assert "Traceback" not in r.text


def test_extract_returns_schema_valid_schedule():
    r = post_image(jpeg_bytes())
    assert r.status_code in (200, 422)
    body = r.json()
    ExtractResponse.model_validate(body)
    assert set(body) >= {
        "status", "drugs", "language", "target_language",
        "explanation", "overall_confidence", "warnings",
    }
    assert body["status"] == "unreadable"


def test_extract_rejects_non_image_content_type():
    r = post_image(b"%PDF-1.4 not an image", content_type="application/pdf")
    assert r.status_code == 422


def test_extract_rejects_empty_payload():
    r = client.post("/extract", files={"image": ("rx.jpg", b"", "image/jpeg")})
    assert r.status_code == 422


def test_extract_rejects_corrupt_image():
    r = post_image(b"this is definitely not a jpeg")
    assert r.status_code in (200, 422)
    assert "Traceback" not in r.text


def test_extract_rejects_oversized_payload():
    big = jpeg_bytes() + b"x" * (13 * 1024 * 1024)
    r = post_image(big)
    assert r.status_code == 422


def test_extract_writes_nothing_to_disk(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    post_image(jpeg_bytes())
    assert list(tmp_path.iterdir()) == []


def test_extract_response_reports_tagger():
    r = post_image(jpeg_bytes())
    assert "tagger" in r.json()
    assert r.json()["tagger"] in {"distilbert", "lexicon", "distilbert+lexicon"}


def test_lexicon_extracts_frequency_and_duration():
    from src.pipeline import LexiconTagger, tags_to_schedule

    text = "Amoxicillin 500 mg twice daily for 10 days"
    tags = LexiconTagger().tag(text)
    s = tags_to_schedule(text, tags, "en", "en")
    assert s.status == "ok"
    d = s.drugs[0]
    assert d.drug == "Amoxicillin"
    assert d.dose == "500" and d.dose_unit == "mg"
    assert d.frequency == "twice daily"
    assert d.times_of_day == ["morning", "night"]
    assert d.duration_days == 10


def test_fallback_tagger_prefers_learned_then_lexicon():
    from src.pipeline import FallbackTagger, LexiconTagger, SlotTagger

    class NoDrugLearned(SlotTagger):
        def tag(self, text):
            return [("I-DRUG", 0.9)] * len(text.split())

    class UsefulLexicon(SlotTagger):
        def tag(self, text):
            return [("B-DRUG", 0.8)] * len(text.split())

    fb = FallbackTagger(NoDrugLearned(), UsefulLexicon())
    words = "Amoxicillin 500 mg".split()
    tags = fb.tag("Amoxicillin 500 mg")
    assert len(tags) == len(words)
    assert all(t == "B-DRUG" for t, _ in tags)
    assert fb.last_used == "lexicon"


def test_extract_exposes_per_field_attribution():
    """The response must explain which tokens produced each field."""
    r = post_image(open("fixtures_real/printed_full.jpg", "rb").read())
    assert r.status_code == 200
    fields = r.json()["fields"]
    assert len(fields) == len(r.json()["drugs"])
    for group in fields:
        assert "weakest_field" in group
        for name, info in group["fields"].items():
            assert info["source"]
            assert 0.0 <= info["confidence"] <= 1.0
    names = {f["fields"]["drug"]["source"] for f in fields if "drug" in f["fields"]}
    assert "Amoxicillin" in names


def test_extract_attribution_empty_when_unreadable():
    r = post_image(jpeg_bytes())
    assert r.status_code in (200, 422)
    assert r.json()["fields"] == []
