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

    DURATION_HINTS = {"x", "for", "por", "durante", "during", "over"}

    def tag(self, text: str) -> list[tuple[str, float]]:
        out = []
        prev_low = ""
        for w in text.split():
            low = w.lower().strip(".,;:")
            if low in self.DRUGS:
                out.append(("B-DRUG", 0.7))
            elif low in self.FREQS:
                out.append(("B-FREQ", 0.7))
            elif low in self.UNITS:
                out.append(("B-UNIT", 0.6))
            elif low in self.ROUTES:
                out.append(("B-ROUTE", 0.6))
            elif re.fullmatch(r"\d{1,4}", w) and prev_low not in self.DURATION_HINTS:
                out.append(("B-DOSE", 0.5))
            elif re.fullmatch(r"\d{1,4}", w):
                out.append(("B-DURATION", 0.6))
            else:
                out.append(("O", 0.1))
            prev_low = low
        return out


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
