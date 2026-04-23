# ML 2 - Pure/Fat Classifier v0 after bad-data exclusion

Generated: 2026-04-23

## What Changed

Three recordings identified as dominating the residual error in ML 1 were excluded from the classifier dataset after explicit user visual review:

- `49 - for sure very fat` (labeled fat; model consistently called pure with mean `pPure` = 0.818)
- `67 - fat for sure` (labeled fat; mean `pPure` = 0.905)
- `83 - pure very pure` (labeled pure; mean `pPure` = 0.189)

User judgment: these were bad/unusable data for the pure-vs-fat v0 training target. The important point is that the model surfaced suspicious examples, and independent review confirmed they should not define the class boundary.

These clips are excluded with reason `bad_data_excluded_by_visual_review`. They are not relabeled or deleted, so the decision remains auditable.

`7- 1mm in front` remains excluded by the existing `borderline_fat_excluded` rule.

Implementation:

- Manual exclusions now live in `data/stage2_pure_fat_exclusions.json`.
- `scripts/stage2_pure_fat_policy.mjs` is the shared policy helper.
- `scripts/train_stage2_pure_fat.mjs` and `scripts/validate_stage2_repeated_cv.mjs` both read the same policy.
- Shot-number parsing now trims folder labels before matching, so leading-space folders are handled correctly.
- Validator reads `folderLabel` from `data/labels.json` via `source_path`, matching the trainer.

No model architecture or feature changes. Same log-mel summary features, same standardized logistic regression, same confidence threshold of `0.60`.

## Dataset After Exclusion

Included: **19 examples = 10 pure + 9 fat**

Excluded by reason, 9 total:

- `topped_not_in_pure_vs_fat_v0`: 4 (`21`, `22`, `24`, `26`)
- `bad_data_excluded_by_visual_review`: 3 (`49`, `67`, `83`)
- `borderline_fat_excluded`: 2 (`7- 1mm in front`, `20 - great example of 1mm fat`)

## Single 5-Fold CV

Command:

```sh
npm run train:stage2:pure-fat
```

Result:

- Accuracy: `0.895`
- Pure recall: `1.000`
- Fat recall: `0.778`
- Pure precision: `0.833`
- Fat precision: `1.000`

With confidence >= `0.60`:

- Coverage: `0.947`
- Unsure: `1/19`
- Kept accuracy: `0.944`
- Pure recall: `1.000`
- Fat recall: `0.875`

OOF score separation:

- Min pure `pPure`: `0.698`
- Max fat `pPure`: `0.639`
- Margin: `+0.058`

The margin is positive but thin. A new clip landing in the 0.64-0.70 band should be treated as uncertain.

## Repeated Randomized 5-Fold CV

Command:

```sh
npm run validate:stage2:pure-fat
```

Design: 200 repeats, 5 folds, stratified by class, deterministic seeds from `20260423`.

Summary over 200 repeats:

| Metric | ML 1 (22 samples) | ML 2 (19 samples) | Delta |
|---|---:|---:|---:|
| Mean accuracy | 0.807 | **0.882** | +7.5pp |
| Median accuracy | 0.818 | **0.895** | +7.7pp |
| Min accuracy | 0.682 | **0.842** | +16.0pp |
| 5th percentile accuracy | 0.727 | **0.842** | +11.5pp |
| 95th percentile accuracy | 0.864 | **0.947** | +8.3pp |
| Max accuracy | 0.864 | **1.000** | +13.6pp |
| Mean pure recall | 0.858 | **0.950** | +9.2pp |
| Mean fat recall | 0.755 | **0.807** | +5.2pp |
| Repeats >= 0.70 acc | 199/200 | **200/200** | +1 |
| Repeats >= 0.80 acc | 123/200 | **200/200** | +77 |
| Repeats >= 0.90 acc | 0/200 | **42/200** | +42 |
| Mean coverage @ conf 0.60 | 0.932 | 0.924 | -0.8pp |
| Mean kept acc @ conf 0.60 | 0.829 | **0.926** | +9.7pp |
| Median kept acc @ conf 0.60 | 0.842 | **0.941** | +9.9pp |
| 95th pct kept acc @ conf 0.60 | 0.857 | **1.000** | +14.3pp |

## Hypothesis Verdict

The ML 1 error analysis was useful: it highlighted three examples that were wrong in essentially every repeated-CV split, and user visual review confirmed those examples were bad data for this v0 target.

This supports the following narrower conclusion:

- The current log-mel-summary + logistic-regression architecture has real pure/fat signal on the clean local corpus.
- The prior 81% mean accuracy was partly capped by bad labels/data, not only by model capacity.
- Fat remains the harder class.

It does not prove production accuracy. It proves the current curated local corpus is separable enough to justify running the model live and collecting labels.

## New Weakest Examples

Repeated CV on the cleaned set shows the remaining tail:

- `31 - for sure and very fat/...` - labeled fat; mean `pPure` = `0.627`; 200-repeat accuracy `0.060`; confident-prediction accuracy `0.000` (n=129).
- `53 - fat for sure 1 inch/...` - labeled fat; mean `pPure` = `0.596`; 200-repeat accuracy `0.200`; confident-prediction accuracy `0.056` (n=124).
- `39 - very pure/...` - labeled pure; mean `pPure` = `0.514`; 200-repeat accuracy `0.500`; confident-prediction accuracy `0.717` (n=60).

The two remaining worst examples are both labeled fat and are often called pure. They should be reviewed next, but they are not being excluded automatically.

## What This Validates

- The pure/fat model is stable across randomized folds inside this curated local corpus.
- The app's `quality` column is using a real Stage 2 model, not a heuristic.
- Confidence-gated display is useful: mean kept accuracy is `0.926` at about `0.924` coverage.
- The model architecture is good enough for live comparison and data collection.

## What This Does Not Validate

- Still only 19 included examples from one phone/player/location domain.
- The OOF margin is thin at `0.058`.
- Topped is not modeled.
- Cross-validation cannot test new phone placement, turf, wind, player, club, or course.
- A frozen live holdout is still required before trusting the numbers as product accuracy.

## Verification

Commands rerun after the shared exclusion-policy fix:

```sh
node --check scripts/stage2_pure_fat_policy.mjs
node --check scripts/train_stage2_pure_fat.mjs
node --check scripts/validate_stage2_repeated_cv.mjs
npm run train:stage2:pure-fat
npm run validate:stage2:pure-fat
```

Frontend path sanity check:

- `frontend/stage1b.js` loaded `frontend/models/stage1b_detector.json`.
- `frontend/stage2.js` loaded `frontend/models/stage2_pure_fat.json`.
- Known pure prepared clip: Stage 1b accepted as shot, Stage 2 returned pure.
- Known fat prepared clip: Stage 1b accepted as shot, Stage 2 returned fat.

Onset detector check:

```sh
node frontend/eval_labels.mjs recentered
```

Known good result reproduced at threshold `0.65`: TP `28`, FP `0`, FN `0`.

## Immediate Next Steps

1. Commit this cleaned Stage 2 baseline and tag it `stage2-purefat-v0.2`.
2. Live iPhone/range test the full hybrid detector plus quality column.
3. Export accepted and rejected clips after each session.
4. Set aside a frozen holdout before adding new clips to training.
5. Add live false positives as Stage 1b hard negatives.
6. Add clear live pure/fat examples to Stage 2 only after the holdout split is defined.
