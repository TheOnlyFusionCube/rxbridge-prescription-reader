# demo_v2 narration — source of truth + claim traceability

Voice: macOS `say -v Samantha -r 182`, converted with ffmpeg to 48 kHz stereo WAV.
Files: `audio/line1.wav` … `audio/line6.wav`. Measured durations via ffprobe.

| # | Spoken line | Measured | Traceable to devpost.md |
|---|-------------|----------|--------------------------|
| 1 | "The most important instruction a patient receives can be impossible to read: handwriting, or another language." | 6.12 s | L18-19 ("A prescription is the single most important instruction a patient receives for their own body, and a large share of patients cannot act on it"); L5-8 ("cannot read the writing … illegible … in a language they do not read") |
| 2 | "RxBridge turns a photo of a prescription into a spoken, structured medication schedule." | 5.07 s | L3-6 (tagline, verbatim) |
| 3 | "It extracts the drug, the dose, and the frequency, then lays them out as a simple day grid." | 5.13 s | L48-50 ("a slot-tagger that turns OCR text into drug / dose / frequency / duration"; "pictogram day-grid") |
| 4 | "Press read aloud, and the schedule is spoken in the patient's own language." | 3.78 s | L92-94 ("Pick the language, press Read aloud … spoken with the browser's Web Speech API") |
| 5 | "Recognition needs no cloud round trip. Read aloud uses the browser's own voice." | 4.49 s | L10 ("recognition needs no cloud round trip"); L60-61 ("in-memory only: no disk write, no cloud call"); L93-94 ("so no cloud TTS call is needed") |
| 6 | "RxBridge. Photograph a prescription. Hear your medicines, in your own language." | 5.07 s | L3 (tagline, verbatim) |

Total narration: 29.66 s. Composition plans ~0.25 s lead + ~0.6–0.7 s tail per scene → ~33.8 s total.

## On-screen captions (all quoted/paraphrased from devpost.md)

- S1 problem: "A prescription a patient cannot act on." / "Handwriting. Another language." (L18-19, L5-8)
- S2 app: "Photograph the prescription." / "Single-screen, mobile-first PWA." (L98-99)
- S3 result: "Drug. Dose. Frequency." / "Pictogram day-grid — no reading required." / "Per-line confidence, shown." (L95-97, L103-105)
- S4 read aloud: "Spoken in the patient's own language." / "Web Speech API." (L92-94)
- S5 privacy: "No cloud round trip." / "In-memory only: no disk write, no cloud call." / "Never medical advice — it only reads back the prescription." (L10, L60-61, L11-12, L100-102)
- S6 closing: "Photograph a prescription. Hear your medicines, in your own language." (L3)

## Claims deliberately NOT used (per brief MUST-NOT)

- No accuracy/benchmark numbers of any kind (char-F1, entity F1, CER, test counts, confidence numbers).
- No translation/medical-accuracy claim, no patient-outcome claim, no "multilingual accuracy".
- The phrase "nothing leaves the device" was NOT used: the submission docs only support
  "recognition needs no cloud round trip" / "in-memory only: no disk write, no cloud call"
  (devpost.md L181-182 describes a deployed API), so the exact documented wording is used.
