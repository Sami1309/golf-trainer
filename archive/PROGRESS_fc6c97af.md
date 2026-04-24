# Progress Handoff fc6c97af

Generated: 2026-04-23

## Current Objective

Build an iPhone-ready hybrid golf shot detector:

1. Spectral-flux onset detector finds candidate impact moments.
2. Stage 1b verifier model rejects non-shot onsets.
3. Later classifier model labels shot quality/classes such as pure/fat/topped.
4. App should work live from mic and in file-analysis mode.

Current focus is still Stage 1 / Stage 1b detection, not the shot-quality classifier yet.

## High-Level State

The hybrid detector app is working in the browser:

- Live mic path runs onset detection and then Stage 1b model verification.
- File-analysis path also runs onset detection and Stage 1b verification.
- File mode uses the calibrated labeled-sample onset threshold, independent of the live slider.
- Live mode includes a visible one-shot calibration flow where the user hits one ball and the app maps measured shot strength to the onset threshold.
- Stage 1b is now deployed as a log-mel logistic verifier, not the earlier handcrafted-feature model.

## Important Data Notes

The local positive recordings contain spoken shot-name pre-roll. Do not train on full local `.m4a` files as positives.

Correct positive handling:

- Positives are cropped from `data/labels.json` around the labeled shot time.
- Clips are 500 ms at 16 kHz.
- Shot impact is recentered near the label by searching for the local peak.
- Full local recordings are not used as positive training clips.

Current external data status:

- External negatives are available and used from `data/external/manifest.jsonl`.
- External positives are not meaningfully available. The sourced positives appear to be copies of local recordings, so they are not treated as external diversity.
- Negative set includes birds, whooshes, human percussive sounds, impacts, outdoor ambient, phone handling, and equipment-like noises.

New hard-negative handling added:

- The trainer now extracts local pre-shot clips from the same labeled local recordings.
- These are labeled `not_shot`.
- Purpose: prevent the model from learning "local iPhone recording" versus "external library clip" and ensure spoken pre-roll is explicitly treated as negative.

## Core Artifacts

Frontend:

- `frontend/index.html`
- `frontend/app.js`
- `frontend/style.css`
- `frontend/audio_features.js`
- `frontend/stage1b.js`
- `frontend/models/stage1b_detector.json`
- `frontend/models/stage1b_logmel.json`
- `frontend/models/stage1b_handcrafted.json`

Training/eval:

- `scripts/train_stage1b_detector.mjs`
- `scripts/train_stage1b_logmel.mjs`
- `frontend/eval_labels.mjs`

Reports/data:

- `data/labels.json`
- `data/stage1b_prepared/manifest.jsonl`
- `data/stage1b_detector_report.json`
- `data/stage1b_logmel_report.json`
- `data/stage1b_handcrafted_report.json`

Docs:

- `frontend/README.md`
- `SOURCING_GUIDE.md`
- `CLAUDE.md`
- `IDEAS.md`

## Commands

Run full Stage 1b training flow:

```sh
npm run train:stage1b
```

This does:

```sh
node scripts/train_stage1b_detector.mjs && node scripts/train_stage1b_logmel.mjs
```

Individual commands:

```sh
npm run train:stage1b:handcrafted
npm run train:stage1b:logmel
```

Evaluate onset detection against labeled local files:

```sh
node frontend/eval_labels.mjs recentered
```

Serve frontend locally:

```sh
cd frontend
python3 -m http.server 8000
```

## Training Pipeline Details

### Handcrafted Baseline Trainer

File: `scripts/train_stage1b_detector.mjs`

Responsibilities:

- Reads local shot labels from `data/labels.json`.
- Crops positives around labeled impact time.
- Adds local pre-shot hard negatives from the same source recordings.
- Reads external trainable negatives from `data/external/manifest.jsonl`.
- Crops external negatives around spectral-flux onset candidates.
- Writes prepared WAV clips into `data/stage1b_prepared/`.
- Trains a standardized logistic regression with handcrafted Stage 1 features.
- Writes:
  - `frontend/models/stage1b_detector.json`
  - `frontend/models/stage1b_handcrafted.json`
  - `data/stage1b_detector_report.json`
  - `data/stage1b_handcrafted_report.json`

Important: when the full `npm run train:stage1b` command runs, the handcrafted script temporarily writes `stage1b_detector.json`, but then the log-mel script overwrites it with the deployed log-mel model.

### Log-Mel Trainer

File: `scripts/train_stage1b_logmel.mjs`

Responsibilities:

- Reads only `data/stage1b_prepared/manifest.jsonl`.
- Does not decode full source recordings.
- Extracts 206 log-mel summary features from the prepared 500 ms clips.
- Trains a standardized logistic regression.
- Uses grouped cross-validation and the same threshold-selection logic.
- Writes:
  - `frontend/models/stage1b_logmel.json`
  - `frontend/models/stage1b_detector.json`
  - `data/stage1b_logmel_report.json`
  - `data/stage1b_detector_report.json`

Current deployed model:

- `frontend/models/stage1b_detector.json`
- `featureExtractor: "logmel_summary"`
- `threshold: 0.79`
- `features: 206`

## Threshold Bug Evaluation

A prior recommendation identified that the trainer was selecting a deployment threshold from predictions on the full training set. That was wrong because it produced an optimistic train-picked threshold.

Fix implemented:

- Deployment threshold now comes from cross-validation threshold metrics.
- Train-only metrics remain in reports but are explicitly labeled as train-only and not used for deployment.
- Grouped CV is used so crops from the same source file stay in the same fold.
- Threshold selection now considers worst-fold recall instead of only pooled OOF metrics.

Current threshold-selection behavior:

- Candidate thresholds are swept from 0.05 to 0.95.
- Selects the highest threshold that preserves worst-fold recall >= 0.95.
- Reports worst-fold specificity.
- This intentionally avoids hiding a weak validation fold behind pooled metrics.

Note: the prior suggestion said "smallest threshold" satisfying recall, but highest threshold is the correct operating choice when the goal is fewer false positives while preserving recall.

## Current Dataset Counts

Prepared dataset:

- `28` shot clips.
- `1082` not-shot clips.
- Total: `1110` prepared clips.

Negative categories:

- `local_preshot_voice_ambient`: 56
- `club_bag_equipment`: 118
- `human_percussive`: 256
- `non_golf_impacts`: 296
- `outdoor_ambient`: 201
- `phone_handling`: 96
- `whoosh_swing`: 59

Skipped external negatives:

- `648` external negative files did not produce usable spectral-flux onset candidates or failed thresholds.

## Current Metrics

### Deployed Log-Mel Stage 1b Verifier

Report: `data/stage1b_detector_report.json`

Model: `frontend/models/stage1b_detector.json`

- Feature extractor: `logmel_summary`
- Threshold: `0.79`
- Feature count: `206`
- CV TP: `28`
- CV FP: `0`
- CV TN: `1082`
- CV FN: `0`
- Precision: `1.000`
- Recall: `1.000`
- Specificity: `1.000`
- Worst-fold recall: `1.000`
- Worst-fold specificity: `1.000`

Interpretation:

- This is strong but should not be considered final proof.
- The small positive set and source-domain differences can still make CV too optimistic.
- Real iPhone/range holdout testing is required.

### Handcrafted Baseline

Report: `data/stage1b_handcrafted_report.json`

Model: `frontend/models/stage1b_handcrafted.json`

- Feature extractor: `stage1_handcrafted`
- Threshold: `0.67`
- CV TP: `28`
- CV FP: `89`
- CV TN: `993`
- CV FN: `0`
- Precision: `0.239`
- Recall: `1.000`
- Specificity: `0.918`
- Worst-fold specificity: `0.901`

Interpretation:

- Handcrafted model catches all positives but allows many onset-triggered negatives through.
- Log-mel is currently much stronger on the prepared dataset.

### Onset Detector Eval

Command:

```sh
node frontend/eval_labels.mjs recentered
```

At threshold `0.65`:

- TP: `28`
- FP: `0`
- FN: `0`
- Precision: `1.000`
- Recall: `1.000`
- F1: `1.000`
- Mean signed timing offset: about `-27.9 ms`
- Median absolute offset: about `26.9 ms`
- Max absolute offset: about `68.1 ms`

Important checked example:

- `Rancho Park Golf Course 50.m4a`
- Label: `3.851s`
- Detection: `3.813s`
- Counts as TP.

## Frontend Behavior

File: `frontend/stage1b.js`

The app can now load either model type:

- `stage1_handcrafted`
- `logmel_summary`

The model JSON declares its `featureExtractor`, and the frontend checks the feature list against the matching extractor before running inference.

File: `frontend/audio_features.js`

Contains:

- Existing handcrafted Stage 1 feature extraction.
- New log-mel feature extraction.
- `LOG_MEL_BANDS = 40`
- `LOG_MEL_FEATURE_NAMES`
- `extractLogMelFeatures(...)`

The log-mel extractor is browser-compatible and does not require a build step or external dependency.

## Important Gotchas

1. Perfect log-mel CV is suspicious until validated live.

The current positives are only 28 local files. Even with local pre-shot hard negatives, the model may still overfit to recording conditions or crop patterns. Treat the perfect CV as "good candidate for live testing," not as final production quality.

2. External positives are not currently useful.

The sourced positives appear to be local copies rather than genuinely external golf shots. Do not count them as independent source diversity.

3. The classifier should not start until detector holdout is tested.

The classifier depends on reliable shot detection/cropping. If Stage 1b has hidden live false positives or missed shots, classifier labels and training data will degrade.

4. The onset threshold and verifier threshold are separate.

Onset threshold controls candidate generation. Stage 1b threshold controls acceptance/rejection of candidate clips.

5. Live calibration is only one-dimensional right now.

It maps measured shot strength/spectral flux to onset threshold. It does not calibrate model probability, mic EQ, or environment noise.

6. Git state is effectively untracked.

`git status --short` shows the repo contents as untracked. Do not assume git can distinguish prior user changes from agent changes without care.

## Validation Already Run

Syntax checks passed:

```sh
node --check frontend/audio_features.js
node --check frontend/stage1b.js
node --check scripts/train_stage1b_detector.mjs
node --check scripts/train_stage1b_logmel.mjs
```

Full training flow completed:

```sh
npm run train:stage1b
```

Onset eval completed:

```sh
node frontend/eval_labels.mjs recentered
```

## Recommended Next Steps

1. Live iPhone/range test the deployed log-mel verifier.

Use the current app with:

- One-shot live calibration.
- Real shots.
- Practice swings.
- Club taps.
- Bag drops.
- Cart/footstep/voice noise.
- Phone handling and pocket movement.

Record accepted/rejected detections and export clips.

2. Create a frozen holdout from live iPhone data.

Minimum useful holdout:

- 20-30 new real shots.
- 100-300 real local negatives.
- Different session from the current 28 positives.
- Do not train on it until after evaluation.

3. Convert live false positives into hard negatives.

Any false positive from live testing should become a high-priority negative. These are more valuable than generic external negatives.

4. Add more positives before classifier training.

For shot-quality classifier, current count is too small. Need more labeled shots per class:

- pure
- fat
- very fat
- topped
- maybe thin/toe/heel later

Target at least dozens per class before expecting a useful classifier.

5. Start classifier only after detector crops are trusted.

The classifier should train on the same 500 ms detected/cropped representation or an intentionally chosen classifier crop. Avoid training classifier directly on full `.m4a` recordings.

6. Consider frozen split support in the trainer.

The earlier reviewer suggested carving out 6 positives and about 200 negatives before CV. This is still useful once more local positives exist. With only 28 positives, it is possible but expensive.

## Best Resume Point

Resume from live validation:

1. Serve the frontend.
2. Test on iPhone at the range.
3. Export all detections.
4. Add exported false positives as hard negatives.
5. Re-run `npm run train:stage1b`.
6. Only then move toward classifier training.

The current deployed detector is ready for that test.
