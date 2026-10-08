# RxBridge
Prescription photo -> spoken + pictogram schedule in the patient's own language.

ML Empowerment Build Challenge 3.0 submission (Devpost deadline Oct 9, 2026 11:45pm PDT).

## Architecture
See `docs/architecture.md`. Pipeline: image -> OCR (learned-tuned) -> token-classification NER -> structured schedule -> multilingual explanation -> TTS + pictogram grid.
