"""Extract a structured prescription schedule from OCR'd text.

Pipeline: OCR text (docTR, printed scaffolding reads well) -> slot tagging
(DistilBERT, trained on synthetic prescription text with OCR-noise augmentation)
-> PrescriptionSchedule.

The tagger is injected so tests run without weights and the trained model drops
in without changing call sites.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from contracts.schema import PrescriptionSchedule

LABELS = [
    "O", "B-DRUG", "I-DRUG", "B-DOSE", "I-DOSE", "B-UNIT", "I-UNIT",
    "B-ROUTE", "I-ROUTE", "B-FREQ", "I-FREQ", "B-DURATION", "I-DURATION",
]
L2I = {label: i for i, label in enumerate(LABELS)}
I2L = {i: label for label, i in L2I.items()}

FREQ_TO_TIMES = {
    "once daily": ["morning"],
    "twice daily": ["morning", "night"],
    "three times daily": ["morning", "noon", "night"],
    "four times daily": ["morning", "noon", "evening", "night"],
    "every 6 hours": ["morning", "noon", "evening", "night"],
    "every 8 hours": ["morning", "evening", "night"],
    "every 12 hours": ["morning", "night"],
    "as needed": [],
    "right eye": [],
    "left eye": [],
    "both eyes": [],
}


@dataclass
class Slot:
    label: str
    text: str
    confidence: float
    span: tuple[int, int]


class SlotTagger:
    def tag(self, text: str) -> list[tuple[str, float]]:
        raise NotImplementedError


class NullTagger(SlotTagger):
    def tag(self, text: str) -> list[tuple[str, float]]:
        return [("O", 0.0)] * len(text.split())


class DistilBertTagger(SlotTagger):
    def __init__(self, model_dir: str, device: str = "cpu"):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self._torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.model.to(device).eval()

    def tag(self, text: str) -> list[tuple[str, float]]:
        torch = self._torch
        words = text.split()
        if not words:
            return []
        enc = self.tokenizer(
            words,
            is_split_into_words=True,
            truncation=True,
            max_length=96,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits[0]
        probs = torch.softmax(logits, dim=-1)
        conf, pred = probs.max(dim=-1)
        out: list[tuple[str, float]] = []
        prev = None
        for wid, p, c in zip(enc.word_ids(), pred.tolist(), conf.tolist()):
            if wid is None or wid == prev:
                continue
            out.append((I2L[int(p)], float(c)))
            prev = wid
        return out[: len(words)]


class LexiconTagger(SlotTagger):
    """Rule-based fallback so the service degrades to something useful."""

    DRUGS = {
        "amoxicillin", "amoxicilina", "amoxicilline", "metformin", "metformina",
        "lisinopril", "lisinoprilo", "atorvastatin", "atorvastatina", "ibuprofen",
        "ibuprofeno", "paracetamol", "omeprazole", "omeprazol", "azithromycin",
        "azitromicina", "amlodipine", "sertraline", "warfarin", "levothyroxine",
        "levotiroxina", "prednisone", "prednisona", "gabapentin", "ciprofloxacin",
        "ciprofloxacino", "ciprofloxacine", "metoprolol", "hydrochlorothiazide",
        "duloxetine", "losartan", "simvastatin", "clotrimazole", "clotrimazol",
        "fenobarbital",
    }
    UNITS = {"mg", "mcg", "g", "ml", "ui", "iu", "l"}
    FREQS = {"bid", "tid", "qid", "qd", "q8h", "q12h", "q6h", "prn", "od", "os", "ou"}
    ROUTES = {"oral", "sublingual", "topical", "iv", "im", "nasal", "rectal"}
    FREQ_PHRASES = (
        "once daily", "once a day", "twice daily", "twice a day",
        "two times a day", "three times daily", "three times a day",
        "four times daily", "four times a day", "every 8 hours",
        "every 12 hours", "every 6 hours", "as needed", "at bedtime",
        "in the morning", "before meals", "after meals",
    )
    DURATION_HINTS = {"x", "for", "por", "durante", "during", "over"}

    def tag(self, text: str) -> list[tuple[str, float]]:
        words = text.split()
        out: list[tuple[str, float]] = []
        i = 0
        while i < len(words):
            span = self._freq_span(words, i)
            if span:
                out.extend([("B-FREQ", 0.75)] * span)
                i += span
                continue
            low = words[i].lower().strip(".,;:")
            prev_low = words[i - 1].lower().strip(".,;:") if i > 0 else ""
            if low in self.DRUGS:
                out.append(("B-DRUG", 0.7))
            elif low in self.FREQS:
                out.append(("B-FREQ", 0.7))
            elif low in self.UNITS:
                out.append(("B-UNIT", 0.6))
            elif low in self.ROUTES:
                out.append(("B-ROUTE", 0.6))
            elif re.fullmatch(r"\d{1,4}", low):
                if prev_low in self.DURATION_HINTS:
                    out.append(("B-DURATION", 0.6))
                elif out and out[-1][0].endswith("DURATION"):
                    out.append(("B-DURATION", 0.6))
                else:
                    out.append(("B-DOSE", 0.5))
            else:
                out.append(("O", 0.1))
            i += 1
        return out

    @classmethod
    def _freq_span(cls, words: list[str], i: int) -> int:
        for n in (4, 3, 2, 1):
            if i + n > len(words):
                continue
            phrase = " ".join(w.lower().strip(".,:;") for w in words[i : i + n])
            if phrase in cls.FREQ_PHRASES:
                return n
        return 0


def group_slots(text: str, tags: list[tuple[str, float]]) -> list[Slot]:
    words = text.split()
    slots: list[Slot] = []
    cur_label = None
    cur_words: list[str] = []
    cur_conf: list[float] = []

    def flush():
        nonlocal cur_label, cur_words, cur_conf
        if cur_label and cur_words:
            slots.append(
                Slot(
                    label=cur_label,
                    text=" ".join(cur_words),
                    confidence=sum(cur_conf) / len(cur_conf),
                    span=(0, 0),
                )
            )
        cur_label, cur_words, cur_conf = None, [], []

    for word, (tag, conf) in zip(words, tags):
        if tag == "O":
            flush()
            continue
        prefix, _, label = tag.partition("-")
        if prefix == "B" or label != cur_label:
            flush()
            cur_label = label
        cur_words.append(word)
        cur_conf.append(conf)
    flush()
    return slots


def tags_to_rows(text: str, tags: list[tuple[str, float]]) -> list[dict]:
    words = text.split()
    rows, offset = [], 0
    for word, (tag, conf) in zip(words, tags):
        rows.append(
            {
                "token": word,
                "tag": tag,
                "prob": float(conf),
                "span": (offset, offset + len(word)),
                "text": word,
                "confidence": float(conf),
                "start": offset,
                "end": offset + len(word),
            }
        )
        offset += len(word) + 1
    return rows


class FallbackTagger(SlotTagger):
    """Prefer the learned tagger, fall back to the lexicon when it finds nothing.

    The learned tagger occasionally emits I- spans with no opening B- tag,
    which the contract drops by design. Rather than returning "unreadable" for
    a prescription the lexicon can read, try the second tagger and report which
    one produced the result.
    """

    def __init__(self, primary: SlotTagger, secondary: SlotTagger):
        self._primary = primary
        self._secondary = secondary
        self.last_used = "primary"

    def tag(self, text: str) -> list[tuple[str, float]]:
        tags = self._primary.tag(text)
        if any(tag.startswith("B-DRUG") for tag, _ in tags):
            self.last_used = "learned"
            return tags
        tags = self._secondary.tag(text)
        self.last_used = "lexicon"
        return tags


def tags_to_schedule(
    text: str, tags: list[tuple[str, float]], language: str, target_language: str
) -> PrescriptionSchedule:
    schedule = PrescriptionSchedule.from_ner(
        tags_to_rows(text, tags), language=language, target_language=target_language
    )
    if schedule.status != "ok":
        return schedule
    drugs = []
    for entry in schedule.drugs:
        times = FREQ_TO_TIMES.get(entry.frequency.lower().strip(), [])
        drugs.append(entry.model_copy(update={"times_of_day": times}))
    return schedule.model_copy(update={"drugs": drugs})
