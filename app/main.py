"""RxBridge API: prescription photo -> structured, spoken schedule.

An uploaded image is recognised (docTR), tagged (a DistilBERT slot tagger
trained by us, or a rule-based lexicon fallback), and shaped into a
PrescriptionSchedule. The image is processed in memory only and is never
written to disk; its contents are never logged.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contracts.schema import PrescriptionSchedule  # noqa: E402
from src.ocr import MAX_IMAGE_BYTES, SUPPORTED_CONTENT_TYPES  # noqa: E402
from src.pipeline import (  # noqa: E402
    DistilBertTagger,
    FallbackTagger,
    LexiconTagger,
    SlotTagger,
    tags_to_schedule,
)

logger = logging.getLogger("rxbridge.api")
logging.basicConfig(level=logging.INFO)

NER_MODEL_DIR = os.environ.get(
    "RXBRIDGE_NER_MODEL", str(REPO_ROOT / "models" / "rxner2")
)

app = FastAPI(
    title="RxBridge",
    description="Reads a prescription photo and returns a structured medication schedule.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_TAGGER: SlotTagger | None = None
_TAGGER_KIND: str = "unloaded"


def get_tagger() -> tuple[SlotTagger, str]:
    """Prefer the learned checkpoint, with the lexicon as a safety net."""
    global _TAGGER, _TAGGER_KIND
    if _TAGGER is not None:
        return _TAGGER, _TAGGER_KIND
    if Path(NER_MODEL_DIR, "config.json").exists():
        try:
            learned = DistilBertTagger(NER_MODEL_DIR, device="cpu")
            _TAGGER = FallbackTagger(learned, LexiconTagger())
            _TAGGER_KIND = "distilbert+lexicon"
            return _TAGGER, _TAGGER_KIND
        except Exception as exc:
            logger.warning("trained tagger unavailable (%s); using lexicon", exc)
    _TAGGER = LexiconTagger()
    _TAGGER_KIND = "lexicon"
    return _TAGGER, _TAGGER_KIND


class ExtractResponse(PrescriptionSchedule):
    lines: list[dict] = []
    tagger: str = "lexicon"


@app.get("/health")
def health() -> dict:
    _, kind = get_tagger()
    return {"ok": True, "tagger": kind}


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>RxBridge</title></head><body style=\"font-family:system-ui;"
        "max-width:34rem;margin:3rem auto;padding:0 1rem\">"
        "<h1>RxBridge</h1><p>Prescription photo &rarr; spoken, structured schedule.</p>"
        "<p>POST an image to <code>/extract</code> as multipart field "
        "<code>image</code>.</p></body></html>"
    )


@app.post("/extract", response_model=ExtractResponse)
async def extract(image: UploadFile = File(...)) -> JSONResponse:
    """Recognise a prescription image and return its structured schedule."""
    started = time.time()
    content_type = (image.content_type or "").lower()
    if content_type and content_type not in SUPPORTED_CONTENT_TYPES:
        raise HTTPException(status_code=422, detail="unsupported image type")

    payload = await image.read()
    if not payload:
        raise HTTPException(status_code=422, detail="empty image payload")
    if len(payload) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=422, detail="image exceeds 12 MB limit")

    # Never log pixels or decoded text: this is patient health information.
    logger.info("extract: bytes=%d content_type=%s", len(payload), content_type or "unknown")

    try:
        from src.ocr import recognize

        page = recognize(payload)
    except ValueError as exc:
        logger.info("extract: unreadable (%s)", exc)
        return JSONResponse(
            status_code=422,
            content=ExtractResponse(
                status="unreadable",
                drugs=[],
                explanation=None,
                overall_confidence=None,
                warnings=[str(exc)],
            ).model_dump(),
        )

    tagger, kind = get_tagger()
    text = page.text
    tags = tagger.tag(text) if text.split() else []
    schedule = tags_to_schedule(text, tags, "en", "en")

    response = ExtractResponse(
        **schedule.model_dump(),
        lines=[
            {"text": ln.text, "confidence": round(ln.confidence, 4)}
            for ln in page.lines
        ],
        tagger=kind,
    )
    logger.info(
        "extract: status=%s drugs=%d lines=%d tagger=%s elapsed=%.2fs",
        schedule.status,
        len(schedule.drugs),
        len(page.lines),
        kind,
        time.time() - started,
    )
    return JSONResponse(content=response.model_dump())
