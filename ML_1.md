# ML 1 - Pure/Fat Classifier v0

Generated: 2026-04-23

## What The Quality Column Is

The `quality` column in the app is the Stage 2 pure-vs-fat classifier.

Pipeline:

1. Stage 1a spectral-flux onset detector finds a candidate impact.
2. Stage 1b log-mel verifier accepts or rejects it as `shot` / `not_shot`.
3. If Stage 1b accepts it, Stage 2 runs on the same canonical 500 ms, 16 kHz, peak-normalized clip.
4. The app shows `pure`, `fat`, or `unsure` in the `quality` column.

Current Stage 2 model:

- Model: `frontend/models/stage2_pure_fat.json`
- Trainer: `scripts/train_stage2_pure_fat.mjs`
- Report: `data/stage2_pure_fat_report.json`
- Feature extractor: `logmel_summary`
- Model type: standardized logistic regression
- Classes: `pure` vs `fat`
- Confidence threshold for surfacing a label: `0.60`

This is not the future full `pure/fat/topped` classifier. It is a deliberately narrow pure-vs-fat v0 for live comparison while collecting better data.

## Label Source

Labels are parsed from the shot folder/title in the prepared manifest source path.

Examples:

- `14 - very pure/...` -> `pure`
- `49 - for sure very fat/...` -> `fat`
- `21 - topped/...` -> excluded from this v0
- `7- 1mm in front/...` -> excluded as borderline

This matches the repo rule that the raw top-level shot folders encode the human label/notes.

## Training Set

Included:

- `11` clear pure clips
- `11` clear fat clips
- `22` total training/eval clips

Excluded:

- `4` topped clips
- `2` 1mm/borderline fat clips
- `6` total excluded

Reason: topped and borderline contact need more examples before they should be modeled. For v0, the useful question is whether clear pure and clear fat have an acoustic signal.

## Single 5-Fold CV

Command:

```sh
npm run train:stage2:pure-fat
```

Result:

- Accuracy: `0.727`
- Pure precision: `0.727`
- Pure recall: `0.727`
- Fat precision: `0.727`
- Fat recall: `0.727`

With confidence >= `0.60`:

- Coverage: `0.864`
- Unsure: `3/22`
- Kept accuracy: `0.842`
- Pure precision: `0.800`
- Pure recall: `0.889`
- Fat precision: `0.889`
- Fat recall: `0.800`

Interpretation: the classifier has a real signal, but the dataset is small enough that a single split is not enough evidence.

## Repeated Randomized 5-Fold CV

Command:

```sh
npm run validate:stage2:pure-fat
```

Validation design:

- `200` repeats
- `5` folds per repeat
- Stratified by class each repeat
- Deterministic rotating random seeds starting at `20260423`
- Every repeat evaluates all `22` examples out-of-fold
- Labels are parsed from the folder/title in each source path

Summary over 200 repeats:

- Mean accuracy: `0.807`
- Median accuracy: `0.818`
- Minimum accuracy: `0.682`
- 5th percentile accuracy: `0.727`
- 95th percentile accuracy: `0.864`
- Maximum accuracy: `0.864`
- Mean pure recall: `0.858`
- Mean fat recall: `0.755`

Repeat stability:

- Repeats with accuracy >= `0.70`: `199/200`
- Repeats with accuracy >= `0.80`: `123/200`
- Repeats with accuracy >= `0.90`: `0/200`

With confidence >= `0.60`:

- Mean coverage: `0.932`
- Mean kept accuracy: `0.829`
- Median kept accuracy: `0.842`
- 5th percentile kept accuracy: `0.762`
- 95th percentile kept accuracy: `0.857`
- Maximum kept accuracy: `0.905`

## What This Validates

The pure-vs-fat model is not just getting lucky on one split. Across randomized fold assignments, it usually lands around `80%` accuracy, and almost every repeat is above `70%`.

That supports:

- The log-mel summary features contain useful pure-vs-fat signal.
- A very small logistic model can extract some of that signal.
- Running this model live in the app for comparison is worthwhile.

## What This Does Not Validate

This does not prove production accuracy or eliminate overfitting risk.

Reasons:

- There are only `22` clear pure/fat examples.
- All positives are from the same small local recording domain.
- Cross-validation can only test variation inside this corpus.
- It cannot test a new range, phone placement, phone case, player, turf, wind, or club.
- Some examples are consistently unstable or consistently misclassified.

The right interpretation is: pure-vs-fat is promising enough to run live and collect labels, but not ready to trust as final feedback.

## Weakest Examples

Repeated CV found a few samples that dominate the remaining error:

- `49 - for sure very fat/Rancho Park Golf Club 5.m4a`
  - Labeled `fat`
  - Accuracy across repeats: `0.000`
  - Mean `pPure`: `0.818`

- `67 - fat for sure/Rancho Park Golf Course 57.m4a`
  - Labeled `fat`
  - Accuracy across repeats: `0.005`
  - Mean `pPure`: `0.905`

- `83 - pure very pure/Rancho Park Golf Club 21.m4a`
  - Labeled `pure`
  - Accuracy across repeats: `0.005`
  - Mean `pPure`: `0.189`

- `53 - fat for sure 1 inch/Rancho Park Golf Course 46.m4a`
  - Labeled `fat`
  - Accuracy across repeats: `0.450`
  - Mean `pPure`: `0.524`

- `6- pure (very)/Rancho Park Golf Course 6.m4a`
  - Labeled `pure`
  - Accuracy across repeats: `0.530`
  - Mean `pPure`: `0.551`

These should be listened to again. They might be genuinely acoustically ambiguous, mislabeled, cropped oddly, or examples where the current features miss the distinction.

## Next ML Step

Use the app to collect live labeled clips:

1. Save every accepted and rejected candidate.
2. Label live shots as `pure`, `fat`, `topped`, `no_shot`, or `unsure`.
3. Export the ZIP after each session.
4. Add clear pure/fat examples to Stage 2 training only after setting aside a frozen holdout.
5. Keep collecting topped separately until there are enough examples for a real topped model.

The most useful immediate additions are more clear fat and clear pure shots from a different session/location/phone placement.
