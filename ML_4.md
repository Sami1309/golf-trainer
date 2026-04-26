# ML 4 - Label review pass + confidence threshold lift

Generated: 2026-04-24

## What Changed

Two things changed since ML 3:

1. **Label audit.** The local labels review tool (`npm run review`) was used to visually re-time impact onsets for the 28 originally `sam`-labeled recordings. The 14 auto-imported recordings were left as-is.
2. **Stage 2 confidence threshold raised from `0.60` to `0.78`.** Driven by a 200-repeat × 5-fold confidence sweep against the cleaner labels.

The Stage 1b verifier and the Stage 2 logistic regression weights are unchanged from ML 3 because the ±100 ms recenter window in `prepare_stage1b_data.mjs` already absorbs sub-100 ms label drift, so the prepared 500 ms positive crops snap to the same peaks before and after the audit. The honest gain from the audit is at the *label* layer (cleaner ground-truth times), not the model layer.

## Label Audit Summary

```
Total entries:                 42
shotTimes edited (>0.5 ms):    28 (all sam-labeled originals; auto-imports untouched)
Mean |delta| ms:               30.4
Max  |delta| ms:               68.0
Sign of delta:                 every edit moved impact earlier
```

User observation: the auto-detector's recentered peak times are tighter than the older `sam` hand-labels. This is now reflected in `data/labels.json` with `reviewedBy: sam-visual` and `lastEditedAt` audit fields.

Onset detector evaluation against the new labels:

| Metric | ML 3 (old labels) | ML 4 (audited labels) |
|---|---:|---:|
| TP at threshold `0.65` | 42 | 42 |
| FP at threshold `0.65` | 0 | 0 |
| FN at threshold `0.65` | 0 | 0 |
| Mean signed offset | `-18.6 ms` | **`+1.66 ms`** |
| Median absolute offset | `21.4 ms` | **`<1 ms`** for most |
| Max absolute offset | `68.1 ms` | `22.1 ms` |

The mean offset went from `-18.6 ms` to `+1.66 ms`, showing the audited labels now sit on top of the actual impact peaks instead of trailing them.

## Stage 1b

Unchanged from ML 3.

- Deployed model: `frontend/models/stage1b_detector.json` (`logmel_summary`, 206 features, threshold `0.79`).
- CV: TP `42`, FP `0`, TN `1110`, FN `0`. Worst-fold recall `1.000`, worst-fold specificity `1.000`.
- OOF margin `0.072`.

The handcrafted baseline is also unchanged (precision `0.339`, OOF margin `-0.302`). Deployed log-mel verifier wins clearly.

## Stage 2 Pure/Fat

The model weights are identical to ML 3 because the prepared positives are byte-identical. What changed is the deployed `confidenceThreshold`.

### Confidence threshold sweep (200×5-fold repeated CV)

| threshold | coverage | kept accuracy | kept fat recall | kept pure recall |
|---:|---:|---:|---:|---:|
| 0.50 | 1.000 | 0.933 | 0.890 | 0.985 |
| 0.55 | 0.977 | 0.942 | 0.900 | 0.993 |
| **0.60 (ML 3)** | **0.952** | **0.949** | **0.909** | **0.997** |
| 0.65 | 0.921 | 0.956 | 0.920 | 1.000 |
| 0.70 | 0.879 | 0.965 | 0.936 | 1.000 |
| 0.72 | 0.861 | 0.968 | 0.942 | 1.000 |
| 0.75 | 0.832 | 0.977 | 0.957 | 1.000 |
| **0.78 (ML 4)** | **0.807** | **0.985** | **0.972** | **1.000** |
| 0.80 | 0.790 | 0.991 | 0.983 | 1.000 |
| 0.82 | 0.776 | 0.995 | 0.990 | 1.000 |
| 0.85 | 0.755 | 0.997 | 0.994 | 1.000 |

`0.78` was chosen because it is the highest threshold where:
- kept pure recall is `1.000`,
- kept fat recall jumps above `0.97`,
- coverage stays above `0.80` (less than 1 in 5 clips lands in `unsure`).

Pushing further (`0.82`, `0.85`) buys very little extra accuracy at meaningful coverage cost.

### Single 5-fold CV at the new threshold

- Included examples: `33` = `15` pure + `18` fat.
- Accuracy at `0.50`: `0.939` (unchanged).
- OOF margin: `-0.209` (`minPureP=0.561`, `maxFatP=0.770`) — unchanged.
- At confidence `>= 0.78`:
  - Coverage `0.879` (`4/33` unsure).
  - Kept accuracy `1.000`.
  - Kept pure recall `1.000`.
  - Kept fat recall `1.000`.

The 4 unsure examples in this single split:

| folder | actual | pPure | reason |
|---|---|---:|---|
| `31 - for sure and very fat` | fat | `0.455` | borderline — model leans fat but weakly |
| `39 - very pure` | pure | `0.561` | borderline — model leans pure but weakly |
| `101 - slightly fat` | fat | `0.707` | confidently *wrong* in old gate; now unsure |
| `53 - fat for sure 1 inch` | fat | `0.770` | confidently *wrong* in old gate; now unsure |

`53` and `101` are the same examples that the user previously confirmed are correctly labeled fat. Putting them into `unsure` instead of confidently calling them pure is the actual win.

## What This Validates

- The audited label corpus aligns onset times to true peaks within ~1 ms (median).
- At the deployed Stage 2 threshold `0.78`, the model is right `100%` of the time it makes a single-CV confident call, and right `~98.5%` of the time across 200 randomized 5-fold repeats.
- The two known confidently-wrong fat clips (`53`, `101`) move from confident-pure to `unsure` rather than misleading the user.
- Stage 1b stays at perfect local CV with margin `0.072`.

## What This Does Not Validate

- Still 33 included examples from one phone / one player / two locations. No external-domain CV.
- OOF margin is still `-0.209`. The model has real overlap between classes and the threshold lift is masking that overlap, not curing it. A new clip that lands in `[0.561, 0.770]` is a coin flip — the gate just refuses to call it.
- Coverage drops from `0.952` to `0.807`: about one in five live clips will display `unsure`. This is intended for v0 data collection. If live use turns "unsure" into a usability problem, revisit after more fat data is collected.
- Cross-CV thresholding measures stability of the confidence rule on shuffled splits, not generalization to a new phone, turf, or player. Frozen live holdout still required.

## Implementation Notes

- Confidence threshold lives in `frontend/models/stage2_pure_fat.json` (`confidenceThreshold: 0.78`).
- `frontend/app.js` defaults `params.stage2ConfidenceThreshold` to `null`, so the live app honors whatever the deployed model declares (mirrors the Stage 1b threshold fix). Calibration records save the threshold actually used at inference.
- `scripts/validate_stage2_repeated_cv.mjs` writes a `confidenceSweep` array in `data/stage2_pure_fat_repeated_cv_report.json` for future threshold decisions.
- `data/labels.json` carries `reviewedAt`, `reviewedBy`, and `lastEditedAt` audit fields per entry.

## Verification

```sh
node --check frontend/audio_features.js
node --check frontend/stage1b.js
node --check frontend/stage2.js
node --check frontend/app.js
node --check scripts/prepare_stage1b_data.mjs
node --check scripts/train_stage1b_detector.mjs
node --check scripts/train_stage1b_logmel.mjs
node --check scripts/train_stage2_pure_fat.mjs
node --check scripts/validate_stage2_repeated_cv.mjs
npm run check
npm run prepare:stage1b
npm run train:stage1b
npm run train:stage2:pure-fat
npm run validate:stage2:pure-fat
node frontend/eval_labels.mjs recentered
```

## Immediate Next Steps

1. Live iPhone/range session with the new threshold. Watch how often `unsure` shows up; collect fat-heavy data so future Stage 2 retrains can lower the unsure rate.
2. Set aside a frozen holdout from the next session before adding any new clips to training.
3. After the next session, retrain and re-sweep the confidence threshold against the bigger corpus.
4. Consider class-asymmetric thresholds if `unsure` rate is too high but only on the pure side (per repeated-CV, kept pure recall already saturates at `1.000` for any threshold `>= 0.65`).
