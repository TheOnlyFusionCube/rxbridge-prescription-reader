# PHI audit — 2026-10-11

Triggered by the campaign constraint that prescription text is PHI and must never
be persisted. This records what was found, what was ruled out, and what still
needs an owner decision. No transcript content is reproduced here.

## Confirmed leak — prescription text in `eval/` artifacts

Ten benchmark artifacts carried per-row `reference` and `prediction` fields
containing real prescription transcription: 374 rows each, references up to
2,767 characters of paragraph-form clinical text.

Ten files: `BENCHMARK_PENAL_1024.json`, `BENCHMARK_V5_PENAL.json`,
`BENCHMARK_V5_PENAL_1024.json`, `BENCHMARK_V6_PENAL.json`,
`BENCHMARK_V7_PENAL.json`, `BENCHMARK_V7_PENAL_1024.json`,
`BENCHMARK_V8_PENAL.json`, `BENCHMARK_V8_PENAL_1024.json`,
`BENCHMARK_V9_PENAL.json`, `BENCHMARK_V9_PENAL_1024.json`.

Working tree scrubbed in `98743a5` — reduced to `{cer, lang}` rows matching the
convention the older artifacts already used. Every `summary` section verified
byte-identical to pre-scrub, so no graded number changed.

**The scrub does not remove PHI from history.** Every commit from `1ff6a35`
onward still serves the original blobs to anyone who runs
`git show <commit>:eval/BENCHMARK_V5_PENAL.json`. Affected commits:

    1ff6a35, baad9cb, 622565d, 43ccdb2, 90cbc24, dee942e, a5bedcd,
    4723a24, 8f971a1, 329caa2, 84edd22

Purging requires rewriting history and force-pushing `main`. `git-filter-repo`
is not installed on this machine; `filter-branch` is available. A force-push to
`main` is outside what an agent may do unilaterally, so this is an owner
decision — see the open questions below.

## Ruled out — not PHI

`data/ner_aug.json`, `data/ner_nl.json`, `data/ner_synth.json` — synthetic.
`gen_ner.py` generates from a closed vocabulary. `ner_nl.json` and
`ner_synth.json` contain 85 unique tokens, all in-vocabulary. `ner_aug.json`
has 2,108 out-of-vocabulary tokens, and every one was verified to be an
OCR-corruption artifact of an in-vocabulary word: `dys` is a character-drop of
"days" (visible in `x 7 dys`, `3 dys` alongside `udrante`, `oias`,
`eiprofloxacine`, `Femobarbital`, `orl`, `PNR`, `fID`); the other 21 are
prefix/suffix truncations (`Amo`←Amoxicillin, `Hydrochl`←Hydrochlorothiazide,
`thiazide`, `xacin`). This is what the augmentation is for.

`models/tokenizer.json`, `models/rxner2/tokenizer.json` — the matched strings
are 30,522-entry vocabulary tokens, not data fields. False positives.

`fixtures_real/printed_full.jpg` + `printed_full_gt.json` — synthetic demo
fixture, not a patient's form. Pixel analysis: 91% adjacent-pixel equality, a
108,274-byte run of identical values, modal value 255 covering ~37% of the
image — a render, not a photograph. Its ground truth opens with a `(555)`
phone prefix (reserved fictional-use range) and lists exactly the drugs and
doses in `gen_ner.py`'s synthetic vocabulary. The "real printed prescription"
phrasing in `devpost.md` refers to this synthetic fixture, which is a wording
problem rather than a PHI problem. Referenced by `tests/test_api.py:134` and
`web/e2e/{daygrid,demo,shots}.spec.ts`.

## Open — four Wikimedia fixtures

`fixtures_real/en_typical.jpg`, `es_clotrimazol.jpg`, `es_fenobarbital.jpg`,
`es_metronidazol.jpg`.

EVIDENCE.md documents these as downloaded from Wikimedia Commons under Public
Domain / CC BY-SA 4.0 — Cuban "RECETA MEDICA" forms plus a Nigerian hospital
form — each visually verified before scoring. Their EXIF blocks are stripped
placeholder data (empty `ImageDescription`, no Make/Model/DateTime); pixel
statistics (18–38% adjacent equality, short flat runs) are consistent with
re-encoded scans of the Commons originals rather than fresh camera captures.

What is known: `EVIDENCE.md:30` records that docTR transcribed a patient's name
from one of these forms (garbled to "larvaus Dauces rauayo"), so a
patient-identifiable detail is visible on the face of at least one image. The
mitigating fact is the documented free licence from Commons. Whether published
under a free licence is acceptable for a Devpost submission is a judgement call
about the uploader's licence, not something an agent can settle from the image.

## Note on remediation limits

Force-pushing removes the blobs from the branch, but GitHub may still serve
cached views for a while and any existing fork preserves the old commits. A
complete purge of public history may additionally require contacting GitHub
support.
