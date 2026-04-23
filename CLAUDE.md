# Golf Shot Sound Classifier

Current status note: `AGENTS.md` is the canonical first-read orientation. This file is useful background, but parts of its YAMNet/ONNX plan predate the deployed browser log-mel Stage 1b verifier.

## What this project is

An app that listens to the sound of a driver hitting a golf ball and classifies the shot as **pure**, **fat**, or **topped**. Mic is an iPhone placed near the ball. Classification happens live.

**Current deployment target is a mobile web app** (iPhone Safari, HTTPS). A native iOS port is Phase 2.4 — out of scope until the web app proves the model works.

### Class definitions

- **pure** — club hits ball before ground. Sharp, broadband "crack" with high-frequency content. The good shot.
- **fat** — club hits ground before ball. Muted low-frequency thud with dirt/grass noise. Bad.
- **topped** — club hits upper half of ball, never hits ground. Thinner, higher-pitched click with less sustain. Bad.
- **fat_borderline** ("1mm fat" / "1mm in front") — barely-fat shots that acoustically sit between pure and fat. Kept as a separate label in the dataset; whether to model them as their own class, fold into fat, or drop is an open experiment.
- **not_a_shot** — anything that isn't a shot: ambient, voices, practice swings, claps, cart noise. Needed to gate the live classifier.

## Tech stack

- **Training**: Python 3.11+. PyTorch / torchaudio. `audiomentations` for augmentation. YAMNet loaded via TensorFlow Hub (or a torch port) as a pretrained audio embedding backbone.
- **Inference (web MVP)**: ONNX Runtime Web in the browser. YAMNet frontend + two small MLP heads (Stage 1b binary, Stage 2 three-class), each a separate ONNX. Loaded once, reused per detection.
- **Onset detection (Stage 1a)**: spectral flux implemented directly in JavaScript with `AnalyserNode` + `AudioWorklet`. Already calibrated.
- **Inference (native iOS, later)**: `SNClassifySoundRequest` + `SNAudioStreamAnalyzer` on a `.mlmodel` re-trained via Apple Create ML from the same labeled data.
- **App shell**: single-page static web app in `frontend/` (no build step, vanilla JS modules). Eventually a SwiftUI + XcodeGen iOS app.

## Audio spec (must match throughout the pipeline)

- Sample rate: **16 kHz**, mono.
- Clip length: **500 ms** (100 ms pre-peak + 400 ms post-peak).
- Centering: each clip is re-centered on the **peak |amplitude| sample** in a 120 ms forward search window after the spectral-flux onset. This aligns training clips with what humans hear as the impact moment (labels lag the amplitude peak by ~28 ms of click reaction time — re-centering cancels it out).
- Normalization: peak-normalize to **-3 dBFS** per clip.
- Format on disk: 16-bit PCM WAV.

## Where we are right now

**Phase 0 is done.** A web app at `frontend/` captures live mic audio, runs spectral-flux onset detection, labels ground-truth onset times, sweeps thresholds against those labels, and exports 500 ms clips. See `SUMMARY_1.md` for the full report and `frontend/README.md` for how to run it.

Key Phase 0 results:
- 28 recordings labeled in `data/labels.json`: 11 pure, 11 fat, 2 `fat_borderline`, 4 topped. If borderlines are folded into fat, Stage 2 sees 11 pure / 13 fat / 4 topped.
- Detector calibrated to `threshold=0.8, min_gap=200 ms, recenter=on, FFT=1024, hop=256` → **100% precision, 100% recall** on the labeled set, median detection offset 27 ms.
- Pipeline (ring buffer → worklet → resample → WAV) produces correctly timed 16 kHz clips. Training/inference code must still enforce the shared normalization spec (`-3 dBFS` peak) before feeding models.

**Phase 0 did NOT validate:** anything ML (no classifier exists yet), range-ambient behavior, or iPhone Safari live mic (HTTPS not yet deployed).

## What's next

Stages 1b (shot verifier, binary) and 2 (pure/fat/topped) — both YAMNet-embedding-plus-MLP-head. Shared backbone, two cheap heads. See `PLAN_STAGE1B_STAGE2.md` for the detailed plan and `IDEAS.md` for sequencing, risks, and alternatives.

Fastest path from here:
- Build the offline Python pipeline first: extract clips, cut seed negatives, write `clip_manifest.csv`.
- Cache embeddings once, then train Stage 1b and Stage 2 heads in the same experiment runner.
- Decide Stage 2 v0 scope from the unaugmented confusion matrix before spending time on augmentation. Likely v0 is pure-vs-fat, with topped collected in parallel.
- Add hard negatives before judging Stage 1b. Edge audio from current recordings is useful but too easy.
- Export to ONNX and wire the web app only after offline metrics show a real signal.

## Directory layout (actual, as of now)

```
samples/
├── CLAUDE.md                   # this file
├── PROJECT.md                  # full roadmap (Phase 0 → shipping)
├── STAGE1.md                   # original Stage 1 plan (superseded by PLAN_STAGE1B_STAGE2 for 1b)
├── PLAN_STAGE1B_STAGE2.md      # current focus — ML heads over YAMNet embeddings
├── SUMMARY_1.md                # end-of-Phase-0 report (detector calibration, metrics)
├── IDEAS.md                    # risks, gotchas, alternatives, sequencing
├── data/
│   └── labels.json             # canonical ground truth (28 recordings)
├── labels_2026-04-23T08-36-48-110Z.json   # provenance copy of original export
├── frontend/                   # Phase 0 web app + offline diagnostics
│   ├── index.html  app.js  fft.js  wav.js  onset-worklet.js  style.css
│   ├── README.md
│   ├── smoke_test.mjs  diag.mjs  eval_labels.mjs     # offline analysis tools
│   └── smoke_report.json  diag_report.json  eval_*.json   # their outputs
└── <folder per shot>/          # original .MOV + .m4a, one shot per folder, 28 folders
    e.g. "14 - very pure/", "22 - topped/", "7- 1mm in front/"
```

Not yet created (planned by `PLAN_STAGE1B_STAGE2.md`):
- `scripts/` — Python training pipeline. Prefer the fast-path shape: `extract_clips.py`, `embed_clips.py`, `train_heads.py`, then `augment.py` and `export_onnx.py`.
- `data/clips/<class>/` — extracted 500 ms training clips.
- `data/clips_aug/<class>/` — augmented variants.
- `models/stage1b_*.onnx`, `models/stage2_*.onnx` — trained models for the web app.
- Future: `backend/` (FastAPI + SQLite for the data-collection loop, Phase 2.3).
- Much further out: `app/` (XcodeGen + SwiftUI, Phase 2.4).

**Raw folder naming**: the top-level `N - <label>` folders are the source of truth for both the recording (an `.m4a` + matching `.MOV` per folder) and the folder-label-to-class mapping (see `PLAN_STAGE1B_STAGE2.md` §2.0.1). Never rename these folders.

## Current dataset reality check

- **28** labeled recordings total: **11 pure, 11 fat, 2 fat_borderline, 4 topped**. Treat the two "1mm fat" edge cases as fat for Stage 1b; for Stage 2 v0, fold into fat but mark `confidence=low`.
- **Zero** `not_a_shot` clips extracted yet. The existing recordings total ~216 seconds; excluding the shot windows leaves roughly 3 minutes of edge audio for seed negatives. This should produce hundreds of overlapping windows, but most will be easy quiet negatives.
- **One location** (Rancho Park + a couple labeled "Fareways"). Any model trained on this alone is a Rancho-Park-specific classifier. Holdout from a different course before trusting numbers.
- **Topped is the bottleneck**: 4 samples cannot produce honest 5-fold CV and cannot train a 3-class classifier that won't collapse topped into another class. Either collect more topped *before* Stage 2 or ship pure-vs-fat first.

## Architecture: two stages, shared backbone

```
mic → AudioWorklet ring buffer
     → Stage 1a: spectral-flux onset detector  (JS, already built, 100% P/R calibrated)
     → 500 ms window re-centered on peak amplitude
     → YAMNet (1024-d embedding, ~15–30 ms on iPhone Safari via ORT-web)
     → Stage 1b head: shot / not_shot      → reject if below threshold
     → Stage 2 head: pure / fat / topped / (fat_borderline?) / unsure
     → UI event { label, confidence, timestamp, clip }
```

**Why two heads on one embedding**: YAMNet is the expensive part. Running it once per detection and attaching two tiny (<50 KB) MLP heads is cheap. Both heads train from the same cached embedding tensors — iterate heads without re-extracting features.

**Why Stage 1b at all**: the onset detector over-triggers on non-shot transients (claps, bag drops, voices). Without Stage 1b the UI fills with junk the moment you leave lab conditions.

## Conventions and rules

- **iOS work (when we get there)**: never modify `project.pbxproj`. Use XcodeGen. Small scopes. Test after every change.
- **Raw data is sacred**: never overwrite files in the source shot folders or `data/labels.json`. All transformations produce new directories.
- **Reproducibility**: every Python script uses a fixed random seed and writes a sidecar JSON or CSV recording what it did. Preprocessed and augmented data must be fully regenerable from raw + scripts.
- **No leakage across CV folds**: augmented variants of a source clip stay grouped with the source. When we have multi-session data, also group by recording session.
- **Validation gates are binding**: every phase has them. Failing a gate is information, not embarrassment. Do not skip them.
- **Honest metrics**: k-fold CV (k=5) at the source-clip level, per-class precision/recall + confusion matrix, plus a frozen holdout. Not overall accuracy alone.
- **Label provenance**: `data/labels.json` is the single source of truth. The folder name is the class label; keep the folder name verbatim in the `folderLabel` field.
- **Train and deploy from the same audio spec**: 500 ms, 16 kHz mono, peak-centered, -3 dBFS. A mismatch silently wrecks everything downstream.

## Git and data versioning

Normal Git tracks the reproducible project state, not large raw or generated audio/video payloads. `git-lfs` is not assumed to be installed.

Commit these:

- Code under `frontend/` and `scripts/`.
- Docs and handoff notes.
- Small canonical metadata such as `data/labels.json`, `data/external/manifest.*`, and `data/stage1b_prepared/manifest.jsonl`.
- Small generated model/report JSON: `frontend/models/*.json` and `data/stage1b_*_report.json`.
- Git policy files such as `.gitignore` and `.gitattributes`.

Do not commit these in normal Git:

- Raw top-level `.m4a` and `.MOV` shot recordings.
- Generated clip WAVs in `data/stage1b_prepared/shot/` and `data/stage1b_prepared/not_shot/`.
- External downloaded audio, future holdout/session audio, browser exports, local settings, caches, or `.DS_Store`.

Workflow:

- Check `git status --short --ignored` before staging data-related changes.
- Prefer explicit path staging or `git add -p`.
- After training, commit report/model JSON only when the metrics are intentionally accepted.
- If raw media needs versioning later, install Git LFS or use a dedicated data-versioning store before adding those files.
- Never use destructive reset/checkout commands unless the user explicitly asks for that exact operation.

## Non-goals (for now)

- Android. Native iOS (until web MVP validates the model).
- Cloud inference, user accounts, multi-device sync.
- Detecting swing tempo, ball flight, or any non-impact metric.
- Real-time feedback during the swing — we classify the impact, not the motion.
- Polishing the UI before the classifier is trustworthy.

## Where to look for the plan

- `SUMMARY_1.md` — what was built and validated through Phase 0.
- `PLAN_STAGE1B_STAGE2.md` — **current focus**. Training, export, and web integration of both ML heads.
- `STAGE1.md` — original Stage 1 plan (pre-calibration). Stage 1a is done; Stage 1b is now covered by `PLAN_STAGE1B_STAGE2.md`.
- `PROJECT.md` — long-term roadmap from data audit through in-the-wild validation and native iOS.
- `IDEAS.md` — sequencing recommendation, risks, gotchas, and alternatives raised during planning.

Read these before starting work. Every plan document has explicit validation gates — treat them as binding.
