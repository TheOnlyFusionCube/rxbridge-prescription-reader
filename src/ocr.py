"""DocTR-based line and word recognition for prescription images.

docTR is the measured OCR stage: it locates text lines and reads printed
content well (form scaffolding, numerals, dates) and is weak on cursive
handwriting. This module exposes exactly that, plus per-line confidence, so
the caller can decide what to trust.

Images are processed in memory. Nothing is written to disk.
"""

from __future__ import annotations

import io
import logging
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

logger = logging.getLogger("rxbridge.ocr")

MAX_IMAGE_BYTES = 12 * 1024 * 1024
SUPPORTED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp"}

_PREDICTOR = None


@dataclass
class WordResult:
    text: str
    confidence: float


@dataclass
class LineResult:
    text: str
    confidence: float
    box: list[int] = field(default_factory=list)
    words: list[WordResult] = field(default_factory=list)


@dataclass
class PageResult:
    lines: list[LineResult]
    width: int
    height: int
    text: str


def _get_predictor():
    """Load the docTR model once, lazily, and reuse it."""
    global _PREDICTOR
    if _PREDICTOR is None:
        # Imported lazily so that tests which never touch OCR do not pay the
        # cost of importing torch and downloading weights.
        os.environ.setdefault("USE_TORCH", "1")
        from doctr.models import ocr_predictor

        _PREDICTOR = ocr_predictor(pretrained=True)
    return _PREDICTOR


def _decode_image(image_bytes: bytes) -> Image.Image:
    if not image_bytes:
        raise ValueError("empty image payload")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("image exceeds 12 MB limit")
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except Exception as exc:
        raise ValueError(f"could not decode image: {exc}") from exc
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    return image


def recognize(image_bytes: bytes) -> PageResult:
    """Run OCR over one prescription image.

    Returns a PageResult with one LineResult per detected line, ordered top to
    bottom. Raises ValueError for anything that is not a usable image.

    The payload is validated here and then handed to docTR as raw bytes, so the
    image is decoded in memory only and never touches the filesystem.
    """
    image = _decode_image(image_bytes)
    height, width = np.asarray(image).shape[:2]

    from doctr.io import DocumentFile

    doc = DocumentFile.from_images(image_bytes)
    predictor = _get_predictor()
    result = predictor(doc)

    lines: list[LineResult] = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                words = [
                    WordResult(text=w.value, confidence=float(w.confidence))
                    for w in line.words
                ]
                text = " ".join(w.text for w in words).strip()
                if not text:
                    continue
                geometry = list(line.geometry)
                box = [
                    int(min(p[0] for p in geometry) * width) if geometry else 0,
                    int(min(p[1] for p in geometry) * height) if geometry else 0,
                    int(max(p[0] for p in geometry) * width) if geometry else 0,
                    int(max(p[1] for p in geometry) * height) if geometry else 0,
                ]
                conf = sum(w.confidence for w in words) / len(words) if words else 0.0
                lines.append(
                    LineResult(text=text, confidence=float(conf), box=box, words=words)
                )

    lines.sort(key=lambda ln: (ln.box[1], ln.box[0]))
    return PageResult(
        lines=lines,
        width=width,
        height=height,
        text=" ".join(ln.text for ln in lines),
    )
