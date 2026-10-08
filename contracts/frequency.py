"""Frequency-phrase normalization for the RxBridge prescription contract.

Hand-written, dependency-free. Maps clinical shorthand (BID/TID/QID/PRN/qNh)
and common English phrases to a single canonical phrase so the day-grid and
text-to-speech layers never see raw abbreviations.
"""

from __future__ import annotations

import re

_SIMPLE: dict[str, str] = {
    "qd": "once daily",
    "od": "once daily",
    "daily": "once daily",
    "once daily": "once daily",
    "once a day": "once daily",
    "bid": "twice daily",
    "twice daily": "twice daily",
    "twice a day": "twice daily",
    "tid": "three times daily",
    "three times daily": "three times daily",
    "three times a day": "three times daily",
    "qid": "four times daily",
    "four times daily": "four times daily",
    "four times a day": "four times daily",
    "prn": "as needed",
    "as needed": "as needed",
    "as required": "as needed",
    "qhs": "at bedtime",
    "hs": "at bedtime",
    "at bedtime": "at bedtime",
    "qam": "in the morning",
    "am": "in the morning",
    "in the morning": "in the morning",
    "qpm": "in the evening",
    "pm": "in the evening",
    "in the evening": "in the evening",
}

_COUNT_WORDS = {1: "once daily", 2: "twice daily", 3: "three times daily", 4: "four times daily"}

_QNH = re.compile(r"q\s*(\d+)\s*h")
_NX_PER_DAY = re.compile(r"(\d+)\s*(?:x|times?\s*(?:a|per)\s*day)")


def normalize_frequency(raw: str) -> str:
    """Return the canonical phrase for a raw frequency string.

    Unknown input is returned stripped but otherwise unchanged so no signal is
    silently invented. Blank input returns ``""``.
    """
    text = "" if raw is None else raw.strip()
    if not text:
        return ""

    key = re.sub(r"\s+", " ", text.lower()).strip().rstrip(".")
    if key in _SIMPLE:
        return _SIMPLE[key]

    qnh = _QNH.fullmatch(key)
    if qnh:
        hours = int(qnh.group(1))
        return f"every {hours} hour" if hours == 1 else f"every {hours} hours"

    per_day = _NX_PER_DAY.fullmatch(key)
    if per_day:
        count = int(per_day.group(1))
        return _COUNT_WORDS.get(count, f"{count} times daily")

    return text
