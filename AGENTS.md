# AGENTS.md

First-read orientation for agents working in this repo.

## Project Goal

Build an iPhone-ready golf-shot audio app.

Target behavior:

1. Listen through an iPhone mic near the ball.
2. Detect candidate golf impacts with onset detection.
3. Reject non-shot transients with a binary Stage 1b verifier.
4. Classify accepted shots as `pure`, `fat`, or `unsure` for the current v0; later expand to `topped`.
5. Run first as a mobile web app in iPhone Safari; native iOS is later.

Current work is live validation and data collection. The hybrid detector is deployed in the web app. A pure-vs-fat Stage 2 classifier is also deployed for live comparison, but the full pure/fat/topped classifier is not implemented yet.

Current product milestone: **mobile web v1 data-collection app**. The code is functionally set up for range testing; the next UI pass should wait for the planned external design/mockup input rather than doing more ad hoc styling.

## Current Truth

The app currently has a working hybrid detector:

- Stage 1a: spectral-flux onset detector in the browser.
- Stage 1b: deployed log-mel logistic verifier loaded from `frontend/models/stage1b_detector.json`.
- Stage 2 v0: deployed experimental log-mel logistic pure-vs-fat classifier loaded from `frontend/models/stage2_pure_fat.json`.
- Live mic mode runs onset detection, extracts a 500 ms clip, then runs Stage 1b.
- Accepted live/file detections run the Stage 2 pure-vs-fat model for comparison.
- Live/file candidates are stored locally in IndexedDB with a 500 ms model clip and configurable review clip.
- The review clip defaults to `5` seconds so the user can speak shot notes before or after impact.
- Live audio uses a `12` second AudioWorklet ring buffer so the default 5 second review clip can be extracted after waiting for post-shot context.
- File-analysis mode runs the same detector/verifier path on uploaded files.
- File mode uses the calibrated labeled-sample onset threshold, not the live slider.
- Live mode starts with one-shot calibration: user starts recording, hits one calibration shot, confirms/retries it, and the app maps measured shot strength to onset threshold before entering live mode.

Current deployed Stage 1b model:

- Model file: `frontend/models/stage1b_detector.json`
- Feature extractor: `logmel_summary`
- Threshold: `0.80`
- Feature count: `206`
- Training report: `data/stage1b_detector_report.json`

Current prepared training set:

- `28` shot clips.
- `1082` not-shot clips.
- `1110` total prepared clips.
- Prepared manifest: `data/stage1b_prepared/manifest.jsonl`

Current deployed log-mel CV metrics:

- TP `28`
- FP `0`
- TN `1082`
- FN `0`
- Precision `1.000`
- Recall `1.000`
- Specificity `1.000`
- Worst-fold specificity `1.000`
- OOF separation margin `0.076` (`minPositiveP=0.801`, `maxNegativeP=0.725`)

Important interpretation: the perfect CV is promising but not production proof. Positives are still only 28 local recordings, so live iPhone/range holdout testing is the next required validation.

Current deployed Stage 2 pure/fat model:

- Model file: `frontend/models/stage2_pure_fat.json`
- Feature extractor: `logmel_summary`
- Confidence threshold: `0.60`
- Feature count: `206`
- Training report: `data/stage2_pure_fat_report.json`
- Repeated-CV report: `data/stage2_pure_fat_repeated_cv_report.json`
- Curation policy: `data/stage2_pure_fat_exclusions.json`

Current Stage 2 v0 metrics after visual bad-data exclusion:

- Included examples: `19` total = `10` pure + `9` fat.
- Excluded examples: `9` total = `4` topped, `2` borderline/1mm fat, `3` bad-data visual-review exclusions (`49`, `67`, `83`).
- Single 5-fold CV: accuracy `0.895`, pure recall `1.000`, fat recall `0.778`.
- At confidence >= `0.60`: coverage `0.947`, kept accuracy `0.944`, pure recall `1.000`, fat recall `0.875`.
- Repeated randomized 5-fold CV, 200 repeats: mean accuracy `0.882`, median `0.895`, min `0.842`, p95 `0.947`, max `1.000`.
- Repeated-CV confidence >= `0.60`: mean coverage `0.924`, mean kept accuracy `0.926`.
- OOF separation margin `0.058` (`minPureP=0.698`, `maxFatP=0.639`).

Important interpretation: Stage 2 now has a real local pure/fat signal and is worth running in the app. It is still not production-proof because it has only 19 included examples from one local recording domain.

## Critical Data Rules

Raw local positive recordings include a person saying the shot name before impact. Do not train on full local `.m4a` files as positive examples.

Correct handling:

- Use `data/labels.json` as the canonical source of shot timestamps.
- Crop positives around labeled impact time.
- Recenter near impact using peak amplitude.
- Use 500 ms clips at 16 kHz mono.
- Peak-normalize to `-3 dBFS`.
- Treat pre-shot local audio, including spoken labels, as negative or ignore it.

The current trainer already adds local pre-shot hard negatives:

- Category: `local_preshot_voice_ambient`
- Count: `56`
- Purpose: prevent the model from learning "local phone recording" versus "external library clip."

External positives are not currently useful. The sourced positives appear to be copies of local recordings, not independent golf-shot sources. Do not count them as external diversity.

Raw top-level shot folders are source material. Do not rename or overwrite them.

## Repo Map

Top-level docs:

- `AGENTS.md` - this orientation file.
- `CLAUDE.md` - shorter synced handoff for Claude-style agents.
- `PROJECT.md` - current v1 product status and next-step roadmap.
- `MANUAL.md` - manual GitHub Pages setup and iPhone deployment test steps.
- `ML_2.md` - current Stage 2 pure/fat validation report.
- `SOURCING_GUIDE.md` - guidance for sourcing external positive/negative audio.
- `archive/` - older plans and progress docs that are useful history but not current guidance.

Raw shot folders:

- Top-level folders such as `14 - very pure/`, `22 - topped/`, `58 - fat for sure/`.
- Each usually contains an `.m4a` and matching `.MOV`.
- Folder name encodes the human label/notes.
- These are raw source files and should not be modified in-place.

`data/`:

- `data/labels.json` - canonical shot labels and shot times for the 28 local recordings.
- `data/external/` - sourced sound-effect data organized by polarity/license/status.
- `data/external/manifest.jsonl` - manifest consumed by the Stage 1b trainer.
- `data/stage1b_prepared/` - generated 500 ms training WAV clips.
- `data/stage1b_prepared/shot/` - generated positive clips.
- `data/stage1b_prepared/not_shot/` - generated negative clips.
- `data/stage1b_detector_report.json` - report for deployed model.
- `data/stage1b_logmel_report.json` - report for log-mel model.
- `data/stage1b_handcrafted_report.json` - report for handcrafted baseline.
- `data/stage2_pure_fat_report.json` - current pure/fat training and single-CV report.
- `data/stage2_pure_fat_repeated_cv_report.json` - repeated randomized CV report.
- `data/stage2_pure_fat_exclusions.json` - auditable Stage 2 v0 exclusion policy.

`frontend/`:

- `frontend/index.html` - single-page web UI.
- `frontend/app.js` - main live/file detection app logic, v1 live session state, calibration confirmation, review/export flow.
- `frontend/audio_features.js` - shared 16 kHz clip prep, handcrafted features, log-mel features.
- `frontend/stage1b.js` - model loading and Stage 1b inference.
- `frontend/fft.js` - FFT and Hann window helpers used by detectors/features.
- `frontend/wav.js` - WAV encoding/decoding utilities.
- `frontend/onset-worklet.js` - audio worklet support for live audio buffering.
- `frontend/style.css` - app styles.
- `frontend/README.md` - how to run/test the web app.
- `frontend/eval_labels.mjs` - offline onset detector evaluation against `data/labels.json`.
- `frontend/diag.mjs`, `frontend/smoke_test.mjs` - diagnostics/smoke tools.
- `frontend/models/stage1b_detector.json` - deployed browser model.
- `frontend/models/stage1b_logmel.json` - log-mel model copy.
- `frontend/models/stage1b_handcrafted.json` - handcrafted baseline model copy.
- `frontend/models/stage2_pure_fat.json` - deployed experimental pure-vs-fat model.

`scripts/`:

- `scripts/train_stage1b_detector.mjs` - prepares Stage 1b dataset and trains handcrafted baseline.
- `scripts/train_stage1b_logmel.mjs` - trains/deploys log-mel verifier from prepared clips.
- `scripts/train_stage2_pure_fat.mjs` - trains the pure-vs-fat classifier from prepared positive clips.
- `scripts/stage2_pure_fat_policy.mjs` - shared Stage 2 label/exclusion policy helper.
- `scripts/validate_stage2_repeated_cv.mjs` - repeated randomized Stage 2 pure/fat CV.
- `scripts/source_golf_detector_data.py` - sourcing helper from prior data-collection work.

`package.json`:

- Defines Node scripts for Stage 1b training, Stage 2 training, and repeated Stage 2 validation.
- No frontend build step.

## How The App Works Now

V1 app states:

- `idle`: ready to start recording.
- `calibrating`: microphone is running and calibration is armed with the temporary low onset gate.
- `calibration-confirm`: one calibration shot was heard; detection is paused until the user confirms or retries.
- `live`: calibrated shot detection is running and recent accepted shots show quality estimates.

Live path:

1. User opens `frontend/index.html` in browser.
2. User taps `Start recording`.
3. The full screen enters a visible recording state and asks for a calibration shot.
4. User hits one calibration shot.
5. The app pauses detection after hearing it and asks the user to confirm or retry.
6. After confirmation, live mode uses the calibrated onset threshold.
7. `frontend/app.js` reads audio from Web Audio.
8. Spectral flux is computed over live frames.
9. Candidate onsets are picked when flux crosses threshold and min-gap rules.
10. A 500 ms clip is extracted around the candidate and resampled to 16 kHz.
11. `frontend/stage1b.js` runs the deployed verifier.
12. Accepted candidates run the experimental Stage 2 pure-vs-fat classifier.
13. Recent shot result and quality are shown in the practice-session panel.
14. Accepted and rejected candidates are stored locally for review/export; rejected candidates can be hidden or shown in the UI.
15. Export writes a ZIP containing `manifest.json`, `clips/context/` review WAVs, and `clips/model_500ms/` canonical clips.

File-analysis path:

1. User uploads an audio file.
2. Browser decodes it.
3. App resamples to detector sample rate.
4. Spectral-flux onset detector runs offline.
5. Each candidate clip is passed through Stage 1b.
6. Detections are rendered with waveform/flux visualization and playable clips.

Calibration:

- Live calibration is one-dimensional.
- User starts recording and the app arms calibration automatically.
- App temporarily lowers the onset gate to catch the next real shot.
- It measures spectral-flux shot strength, pauses detection, and asks the user to confirm or retry.
- On confirmation, live onset threshold becomes a fraction of measured strength.
- This calibrates onset threshold only; it does not calibrate model probability or mic EQ.

Stage 1b inference:

- `frontend/stage1b.js` loads `frontend/models/stage1b_detector.json`.
- Model declares `featureExtractor`.
- Supported extractors:
  - `stage1_handcrafted`
  - `logmel_summary`
- Frontend checks the model feature list against the corresponding extractor.
- Features are standardized with model `mean` and `std`.
- Logistic regression score is converted with sigmoid.
- Candidate is accepted if `pShot >= model.threshold`.

## Audio Spec

Keep this consistent across training and inference:

- Sample rate: `16000 Hz`
- Channels: mono
- Clip length: `500 ms`
- Clip samples: `8000`
- Review clip default: `5` seconds, configurable in the app before capture/export.
- Live ring buffer: `12` seconds. Keep it longer than the max review clip length plus post-shot wait time.
- Positive crop shape: roughly `100 ms` pre-impact and `400 ms` post-impact
- Normalization: peak-normalize to `-3 dBFS`
- Generated training format: 16-bit PCM WAV

Spec mismatch will silently break model quality.

## Training Flow

Full current training command:

```sh
npm run train:stage1b
```

This runs:

```sh
node scripts/train_stage1b_detector.mjs && node scripts/train_stage1b_logmel.mjs
```

What happens:

1. `train_stage1b_detector.mjs` rebuilds `data/stage1b_prepared/`.
2. It crops local positives from `data/labels.json`.
3. It adds local pre-shot negatives from the same local recordings.
4. It pulls trainable external negatives from `data/external/manifest.jsonl`.
5. It writes prepared 500 ms WAVs and manifest.
6. It trains the handcrafted baseline.
7. It writes only the `stage1b_handcrafted` model/report.
8. `train_stage1b_logmel.mjs` reads only the prepared clips.
9. It extracts log-mel summary features.
10. It trains the log-mel logistic verifier.
11. It writes `stage1b_logmel` model/report.
12. It writes `frontend/models/stage1b_detector.json` and `data/stage1b_detector_report.json` with the deployed log-mel model/report.

Individual commands:

```sh
npm run train:stage1b:handcrafted
npm run train:stage1b:logmel
npm run train:stage2:pure-fat
npm run validate:stage2:pure-fat
```

Use `train:stage1b:logmel` only if `data/stage1b_prepared/` is already current.
Use `train:stage1b:handcrafted` when you need to regenerate prepared clips or refresh the handcrafted baseline; it must not change the deployed detector.
Use `train:stage2:pure-fat` only after `data/stage1b_prepared/` exists and is current.
Use `validate:stage2:pure-fat` after Stage 2 training to test split stability over 200 randomized stratified 5-fold repeats.

## Threshold Selection

Do not choose deployment threshold from full-training predictions.

Current correct behavior:

- Cross-validation predictions choose deployment threshold.
- Train-only metrics are reported but not used for deployment.
- CV grouping uses source/group IDs so crops from the same source file stay in the same fold.
- Fold assignment is grouped and class-stratified: groups containing positives are balanced across folds before negative-only groups are filled in.
- Threshold selection considers worst-fold recall, not only pooled OOF metrics.
- The selected threshold is the highest threshold preserving worst-fold recall >= `0.95`.
- Reports include OOF score-separation diagnostics: `minPositiveP`, `maxNegativeP`, and `separationMargin`.

This fixed a prior bug where a train-picked threshold looked better than honest CV.

## Validation Commands

Syntax checks:

```sh
node --check frontend/audio_features.js
node --check frontend/stage1b.js
node --check frontend/stage2.js
node --check frontend/app.js
node --check scripts/train_stage1b_detector.mjs
node --check scripts/train_stage1b_logmel.mjs
node --check scripts/stage2_pure_fat_policy.mjs
node --check scripts/train_stage2_pure_fat.mjs
node --check scripts/validate_stage2_repeated_cv.mjs
```

Full retrain:

```sh
npm run train:stage1b
```

Onset detector eval:

```sh
node frontend/eval_labels.mjs recentered
```

Known good onset result at threshold `0.65`:

- TP `28`
- FP `0`
- FN `0`
- Mean signed offset about `-27.9 ms`
- Median absolute offset about `26.9 ms`

Run frontend locally:

```sh
cd frontend
python3 -m http.server 8000
```

For iPhone mic testing, Safari needs HTTPS. Use a tunnel such as ngrok/Cloudflare/Tailscale or deploy static frontend to HTTPS hosting.

## Current Metrics Snapshot

Deployed log-mel verifier:

- Report: `data/stage1b_detector_report.json`
- Model: `frontend/models/stage1b_detector.json`
- Threshold: `0.80`
- TP `28`
- FP `0`
- TN `1082`
- FN `0`
- Precision `1.000`
- Recall `1.000`
- Specificity `1.000`
- Worst-fold specificity `1.000`
- OOF separation margin `0.076`

Handcrafted baseline:

- Report: `data/stage1b_handcrafted_report.json`
- Model: `frontend/models/stage1b_handcrafted.json`
- Threshold: `0.63`
- TP `28`
- FP `97`
- TN `985`
- FN `0`
- Precision `0.224`
- Recall `1.000`
- Specificity `0.910`
- Worst-fold specificity `0.871`
- OOF separation margin `-0.338`

Prepared negative categories:

- `local_preshot_voice_ambient`: 56
- `club_bag_equipment`: 118
- `human_percussive`: 256
- `non_golf_impacts`: 296
- `outdoor_ambient`: 201
- `phone_handling`: 96
- `whoosh_swing`: 59

Stage 2 pure/fat v0:

- Report: `data/stage2_pure_fat_report.json`
- Repeated CV: `data/stage2_pure_fat_repeated_cv_report.json`
- Model: `frontend/models/stage2_pure_fat.json`
- Included examples: `19` = `10` pure / `9` fat
- Excluded examples: `9` = `4` topped / `2` borderline / `3` bad-data visual-review exclusions
- Single 5-fold CV accuracy: `0.895`
- Confidence >= `0.60` kept accuracy: `0.944` at `0.947` coverage
- Repeated CV mean accuracy: `0.882`
- Repeated CV min accuracy: `0.842`
- Repeated CV mean kept accuracy at confidence >= `0.60`: `0.926`
- OOF separation margin: `0.058`

## Current Limitations

- Only 28 local positive shot recordings.
- Topped class has very few examples.
- Stage 2 pure/fat exists and runs in the app, but it is still an early local-domain classifier.
- Stage 2 topped classifier does not exist yet.
- No frozen live iPhone holdout yet.
- Perfect log-mel CV may be inflated by small positive count and source/domain differences.
- External positives are not independent.
- The app is a web MVP, not native iOS.
- There is no backend or persistent labeling system yet.

## Recommended Next Steps

Immediate non-model next steps:

1. Provide the design prompt/output for the v1 mobile app UI.
2. Apply that design to the existing functional flow in `frontend/`.
3. Write a deployment guide for static HTTPS hosting.
4. Plan the longer path from web v1 validation to a real native iPhone app.

Model/data next steps:

1. Live iPhone/range test the deployed hybrid detector plus pure/fat quality column.

Test real conditions:

- Real shots.
- Practice swings.
- Club taps.
- Bag drops.
- Cart noise.
- Footsteps.
- Voice/conversation.
- Phone handling.
- Wind.

2. Export live detections and rejected clips.

Use exported false positives as high-value hard negatives. These are more useful than generic sound-library negatives.

3. Build a frozen live holdout.

Minimum useful holdout:

- 20-30 new real shots.
- 100-300 real local negatives.
- Different session from the current 28 positives.
- Do not train on it until after evaluation.

4. Re-run Stage 1b training after adding live hard negatives.

Command:

```sh
npm run train:stage1b
```

5. Use the live collection/export loop to grow Stage 2 data while treating current pure-vs-fat predictions as experimental comparison data.

Stage 2 should use trusted detector crops, not full `.m4a` recordings. More positives per class are needed before expecting a useful classifier, especially for topped.

## Stage 2 Classifier Direction

Goal:

- Classify accepted golf shots as `pure`, `fat`, `topped`, or `unsure`.

Current implementation:

- `pure` vs `fat` only.
- Excludes topped, 1mm/borderline examples, and three visually reviewed bad-data examples from v0 training.
- Exclusion policy lives in `data/stage2_pure_fat_exclusions.json`; do not add hard-coded exclusions in trainer scripts.
- CV accuracy `0.895` on 19 included clear examples.
- At confidence >= `0.60`: coverage `0.947`, kept accuracy `0.944`, unsure `1/19`.
- Repeated randomized CV mean accuracy `0.882`; all 200 repeats were >= `0.80`.
- Intended for live comparison and data collection, not final trusted feedback.

Current data reality:

- 28 total local positives.
- Approximate labels from folder names:
  - Pure: about 11.
  - Fat: about 11 plus 2 borderline/1mm fat.
  - Topped: about 4.
- Topped is the bottleneck.

Pragmatic classifier sequence:

1. Collect more positives, especially topped and borderline cases.
2. Use the trusted detector to produce consistent 500 ms crops.
3. Start with pure-vs-fat if topped count remains too low.
4. Add augmentation only after baseline confusion matrix shows real signal.
5. Keep source/session grouping for CV.
6. Use a frozen holdout before trusting metrics.

Do not spend time on UI polish or native iOS before the classifier has a real signal.

## Engineering Rules

- Prefer `rg` for file/text search.
- Use `apply_patch` for edits.
- Do not overwrite raw data.
- Do not rename top-level raw shot folders.
- Do not train positives from full local `.m4a` files.
- Keep train/inference audio specs aligned.
- Keep source/group IDs intact in manifests and CV.
- Treat train-only metrics as diagnostic only.
- Record new generated metrics in report JSON.
- If adding dependencies, justify them; current frontend intentionally has no build step.
- If native iOS work starts later, do not hand-edit `project.pbxproj`; use XcodeGen.

## Git/Workspace Notes

This repo contains large raw and generated audio/video assets. Normal Git should track reproducible source and metadata, not raw media payloads.

Track in Git:

- Source code under `frontend/` and `scripts/`.
- Docs such as `AGENTS.md`, `CLAUDE.md`, `PROJECT.md`, and progress reports.
- Small canonical metadata: `data/labels.json`, `data/external/manifest.*`, `data/stage1b_prepared/manifest.jsonl`.
- Small model/report artifacts: `frontend/models/*.json`, `data/stage1b_*_report.json`, and `data/stage2_*_report.json`.
- Git policy files: `.gitignore` and `.gitattributes`.

Do not track in normal Git:

- Top-level raw shot `.m4a` / `.MOV` files.
- Generated prepared WAV clips under `data/stage1b_prepared/shot/` and `data/stage1b_prepared/not_shot/`.
- External downloaded audio files, future holdout audio, local browser exports, or session dumps.
- `.DS_Store`, local settings, dependencies, caches, or temporary files.

Workflow rules:

- Run `git status --short --ignored` before staging if data files changed.
- Use explicit path staging or `git add -p`; avoid broad `git add .` until `.gitignore` has been checked.
- After `npm run train:stage1b`, commit changed model/report JSON and manifests when the metrics are intentional.
- If raw/holdout media must be versioned later, install Git LFS or use a data store/DVC-style workflow before adding media.
- Do not use destructive commands such as `git reset --hard` or `git checkout --` unless the user explicitly asks for that exact operation.

## Best Resume Point

Resume by testing the deployed detector on a real iPhone at a range.

Concrete sequence:

1. Serve `frontend/` over HTTPS.
2. Open on iPhone Safari.
3. Start recording and confirm the calibration shot.
4. Hit shots and intentionally create non-shot transients.
5. Export detections.
6. Add false positives as hard negatives.
7. Re-run `npm run train:stage1b`.
8. Evaluate against a frozen live holdout.
9. Retrain/replace the Stage 2 pure/fat model as live labeled data grows, keeping a frozen holdout before adding new clips to training.
