# AGENTS.md

First-read orientation for agents working in this repo.

## Project Goal

Build an iPhone-ready golf-shot audio app.

Target behavior:

1. Listen through an iPhone mic near the ball.
2. Detect candidate golf impacts with onset detection.
3. Reject non-shot transients with a binary Stage 1b verifier.
4. Later classify accepted shots as `pure`, `fat`, `topped`, or `unsure`.
5. Run first as a mobile web app in iPhone Safari; native iOS is later.

Current work is still detection/verifier quality and live data collection. A very early pure-vs-fat Stage 2 classifier exists for live comparison, but the full pure/fat/topped classifier is not implemented yet.

## Current Truth

The app currently has a working hybrid detector:

- Stage 1a: spectral-flux onset detector in the browser.
- Stage 1b: deployed log-mel logistic verifier loaded from `frontend/models/stage1b_detector.json`.
- Stage 2 v0: experimental log-mel logistic pure-vs-fat classifier loaded from `frontend/models/stage2_pure_fat.json`.
- Live mic mode runs onset detection, extracts a 500 ms clip, then runs Stage 1b.
- Accepted live/file detections run the Stage 2 pure-vs-fat model for comparison.
- Live/file candidates are stored locally in IndexedDB with a 500 ms model clip and 2 second review clip.
- File-analysis mode runs the same detector/verifier path on uploaded files.
- File mode uses the calibrated labeled-sample onset threshold, not the live slider.
- Live mode has one-shot calibration: user taps calibrate, hits a ball, and the app maps measured shot strength to onset threshold.

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
- `CLAUDE.md` - previous agent orientation. Useful, but parts are now stale because it predates the current log-mel deployed verifier.
- `PROJECT.md` - long-term roadmap from data audit through in-the-wild validation and native iOS.
- `SUMMARY_1.md` - Phase 0 detector calibration report.
- `STAGE1.md` - original Stage 1 plan.
- `PLAN_STAGE1B_STAGE2.md` - plan for Stage 1b and Stage 2 model work.
- `IDEAS.md` - gotchas, risks, sequencing notes.
- `SOURCING_GUIDE.md` - guidance for sourcing external positive/negative audio.
- `PROGRESS_fc6c97af.md` - latest detailed handoff before this file.

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

`frontend/`:

- `frontend/index.html` - single-page web UI.
- `frontend/app.js` - main live/file detection app logic.
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
- `frontend/models/stage2_pure_fat.json` - experimental pure-vs-fat model.

`scripts/`:

- `scripts/train_stage1b_detector.mjs` - prepares Stage 1b dataset and trains handcrafted baseline.
- `scripts/train_stage1b_logmel.mjs` - trains/deploys log-mel verifier from prepared clips.
- `scripts/train_stage2_pure_fat.mjs` - trains experimental pure-vs-fat classifier from prepared positive clips.
- `scripts/source_golf_detector_data.py` - sourcing helper from prior data-collection work.

`package.json`:

- Defines Node scripts for Stage 1b training.
- No frontend build step.

## How The App Works Now

Live path:

1. User opens `frontend/index.html` in browser.
2. User starts mic capture.
3. `frontend/app.js` reads audio from Web Audio.
4. Spectral flux is computed over live frames.
5. Candidate onsets are picked when flux crosses threshold and min-gap rules.
6. A 500 ms clip is extracted around the candidate and resampled to 16 kHz.
7. `frontend/stage1b.js` runs the deployed verifier.
8. Accepted candidates run the experimental Stage 2 pure-vs-fat classifier.
9. Accepted and rejected candidates are stored locally for review/export; rejected candidates can be hidden or shown in the UI.

File-analysis path:

1. User uploads an audio file.
2. Browser decodes it.
3. App resamples to detector sample rate.
4. Spectral-flux onset detector runs offline.
5. Each candidate clip is passed through Stage 1b.
6. Detections are rendered with waveform/flux visualization and playable clips.

Calibration:

- Live calibration is one-dimensional.
- User clicks `Calibrate next shot`.
- App temporarily lowers the onset gate to catch the next real shot.
- It measures spectral-flux shot strength.
- Live onset threshold becomes a fraction of measured strength.
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
```

Use `train:stage1b:logmel` only if `data/stage1b_prepared/` is already current.
Use `train:stage1b:handcrafted` when you need to regenerate prepared clips or refresh the handcrafted baseline; it must not change the deployed detector.
Use `train:stage2:pure-fat` only after `data/stage1b_prepared/` exists and is current.

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
node --check scripts/train_stage1b_detector.mjs
node --check scripts/train_stage1b_logmel.mjs
node --check scripts/train_stage2_pure_fat.mjs
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

## Current Limitations

- Only 28 local positive shot recordings.
- Topped class has very few examples.
- Stage 2 pure/fat exists only as an early experimental classifier.
- Stage 2 topped classifier does not exist yet.
- No frozen live iPhone holdout yet.
- Perfect log-mel CV may be inflated by small positive count and source/domain differences.
- External positives are not independent.
- The app is a web MVP, not native iOS.
- There is no backend or persistent labeling system yet.

## Recommended Next Steps

1. Live iPhone/range test the deployed log-mel verifier.

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

5. Use the live collection/export loop to grow Stage 2 data while treating current pure-vs-fat predictions as experimental.

Stage 2 should use trusted detector crops, not full `.m4a` recordings. More positives per class are needed before expecting a useful classifier, especially for topped.

## Stage 2 Classifier Direction

Goal:

- Classify accepted golf shots as `pure`, `fat`, `topped`, or `unsure`.

Current implementation:

- `pure` vs `fat` only.
- Excludes topped and 1mm/borderline examples from v0 training.
- CV accuracy `0.727` on 22 clear examples.
- At confidence >= `0.60`: coverage `0.864`, kept accuracy `0.842`, unsure `3/22`.
- Intended for live comparison and data collection, not trusted feedback.

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
3. Run one-shot calibration.
4. Hit shots and intentionally create non-shot transients.
5. Export detections.
6. Add false positives as hard negatives.
7. Re-run `npm run train:stage1b`.
8. Evaluate against a frozen live holdout.
9. Retrain/replace the experimental Stage 2 model as live labeled data grows.
