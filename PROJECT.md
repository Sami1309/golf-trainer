# Project Status And Next Steps

This repo is now set up around the mobile web v1 data-collection app.

## Current V1 Product Shape

The active app is `frontend/index.html`, a static browser app intended to run on iPhone Safari over HTTPS.

User flow:

1. User places the phone in front of or behind the club path.
2. User starts recording.
3. The whole screen enters a recording state.
4. The app asks for one calibration shot.
5. When it hears that shot, it pauses detection and asks the user to confirm or retry calibration.
6. After confirmation, live mode runs the hybrid shot detector and shows the most recent detected shot plus pure/fat/unsure quality.
7. After the session, the user reviews locally stored detections, replays the review audio, labels each row, and exports a ZIP.

Export contents:

- `manifest.json` with model outputs, labels, timestamps, tester name, calibration metadata, and file paths.
- `clips/context/` review WAVs. Default review duration is 5 seconds and can be changed in the UI.
- `clips/model_500ms/` canonical 500 ms model clips.

## Current Models

Stage 1a:

- Browser spectral-flux onset detector.
- File-mode calibrated threshold: `0.65`.
- Known local labeled-set result: TP `42`, FP `0`, FN `0`.

Stage 1b:

- Browser model: `frontend/models/stage1b_detector.json`.
- Type: log-mel summary standardized logistic regression.
- Threshold: `0.79`.
- CV: TP `42`, FP `0`, TN `1110`, FN `0`.
- OOF margin: `0.072`.

Stage 2:

- Browser model: `frontend/models/stage2_pure_fat.json`.
- Type: log-mel summary standardized logistic regression.
- Output: `pure`, `fat`, or `unsure`.
- Included examples: `33` = `15` pure + `18` fat.
- Exclusion policy: `data/stage2_pure_fat_exclusions.json`.
- Single 5-fold CV accuracy: `0.939`.
- Repeated 200x 5-fold CV mean accuracy: `0.933`.
- Confidence >= `0.60` repeated-CV mean kept accuracy: `0.949`.

Interpretation: Stage 2 is good enough for live comparison and data collection. It is not yet production-proof coaching feedback.

## Validation Commands

```sh
npm run import:new-data -- --dry-run
npm run import:new-data
node --check frontend/audio_features.js
node --check frontend/stage1b.js
node --check frontend/stage2.js
node --check frontend/app.js
node --check scripts/train_stage1b_detector.mjs
node --check scripts/train_stage1b_logmel.mjs
node --check scripts/stage2_pure_fat_policy.mjs
node --check scripts/train_stage2_pure_fat.mjs
node --check scripts/validate_stage2_repeated_cv.mjs
npm run train:stage2:pure-fat
npm run validate:stage2:pure-fat
node frontend/eval_labels.mjs recentered
```

## Next Work

1. Apply the intended visual design to the current v1 flow.
2. Deploy `frontend/` to GitHub Pages for HTTPS iPhone testing. Manual setup is in `MANUAL.md`.
3. Run live range sessions and export all accepted/rejected candidates.
4. Create a frozen holdout from the first live data before adding new clips to training.
5. Add live false positives as Stage 1b hard negatives.
6. Retrain Stage 1b and Stage 2 against the committed baselines.
7. Collect enough topped examples to promote Stage 2 from pure/fat to pure/fat/topped.
8. After the web app proves the loop, plan the native iPhone path.

## Guardrails

- Keep raw `.m4a` and `.MOV` files out of normal Git.
- Stage new one-shot recordings under `new_data/`, run `npm run import:new-data -- --dry-run`, and manually review any file with multiple plausible onset candidates.
- Do not train positives from full local `.m4a` recordings.
- Keep 500 ms, 16 kHz mono, peak-normalized clips aligned across training and inference.
- Keep manual Stage 2 exclusions in `data/stage2_pure_fat_exclusions.json`, not in trainer code.
- Use frozen holdouts before claiming model improvement.
