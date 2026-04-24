# ML 3 - New Sample Import And Scaled Training Pipeline

Generated: 2026-04-24

## What Changed

Imported 14 new one-shot recordings from `new_data/` into the normal top-level raw sample layout, added them to `data/labels.json`, rebuilt the Stage 1b prepared dataset, retrained the deployed shot verifier, retrained the Stage 2 pure/fat classifier, and reran 5-fold plus repeated randomized 5-fold validation.

The new reusable import command is:

```sh
npm run import:new-data
```

It expects `new_data/<folder label>/` folders with exactly one `.m4a` each. It moves those folders to the repo root, keeps the folder names as the human labels, and writes impact timing metadata into `data/labels.json`.

## Imported Samples

New positive recordings:

- `87 - fat 1 inch/Fareways Grab & Golf 3.m4a`
- `90 - fat/Rancho Park Golf Course 64.m4a`
- `92 - fat/Rancho Park Golf Club 25.m4a`
- `94 - pure/Rancho Park Golf Club 27.m4a`
- `95 - pure 100%/Rancho Park Golf Course 65.m4a`
- `99 - fat/Rancho Park Golf Club 30.m4a`
- `101 - slightly fat/Rancho Park Golf Club 32.m4a`
- `102 - for sure pure/Rancho Park Golf Course 67.m4a`
- `103 - for sure pure/Rancho Park Golf Club 33.m4a`
- `109 - for sure fat/Rancho Park Golf Course 70.m4a`
- `114 - pure/Rancho Park Golf Club 39.m4a`
- `118 - fat/Rancho Park Golf Club 43.m4a`
- `119 - fat/Fareways Grab & Golf 4.m4a`
- `120 - fat /Fareways Grab & Golf 5.m4a`

The importer found exactly one strong recentered spectral-flux onset in every new file. These labels are marked with `labeledBy: codex-auto-import` and `labelingMethod: spectral_flux_recentered_peak_v1` so they remain auditable.

## Data Processing That Works

Canonical source:

- `data/labels.json` is the only source of local positive impact times and folder labels.
- Raw `.m4a`/`.MOV` folders stay as source material and are not tracked in normal Git.
- Stage 2 labels are inferred from `folderLabel`, not from full audio contents.

Positive crop sanitation:

- Decode source audio with `ffmpeg` to mono `16000 Hz` float samples.
- Use the labeled impact time as the starting target.
- Recenter within a small neighborhood to the peak absolute-amplitude sample.
- Crop exactly `500 ms` as `100 ms` pre-impact and `400 ms` post-impact.
- Peak-normalize the model clip to `-3 dBFS` through `prepareModelClip`.
- Write generated training clips as 16-bit PCM WAV under `data/stage1b_prepared/`.

Leakage prevention:

- Do not train positives on full local `.m4a` files because they include spoken shot labels before impact.
- Add pre-shot crops from local recordings as hard negatives under `local_preshot_voice_ambient`.
- Keep all crops from a source recording in the same CV group.

Stage 1b negatives:

- Local pre-shot negatives: up to 2 crops per labeled local recording.
- External negatives: only trainable rows from `data/external/manifest.jsonl` with AI training permission.
- Negative categories retained in this run: `club_bag_equipment`, `human_percussive`, `non_golf_impacts`, `outdoor_ambient`, `phone_handling`, `whoosh_swing`.
- External negative crops use onset candidates when possible and a peak fallback only when useful.
- Very quiet negative clips are skipped.

Modeling approach:

- Stage 1b deployed verifier uses `logmel_summary` features with standardized logistic regression.
- The handcrafted Stage 1b model is kept as a baseline only.
- Stage 2 pure/fat uses the same `logmel_summary` feature family and standardized logistic regression.
- No augmentation was added in this run.

Cross-validation policy:

- Stage 1b uses grouped 5-fold CV by `group_id`/source so related crops cannot split across train and validation.
- Stage 1b deployment threshold is selected from CV predictions, not train predictions.
- Stage 1b threshold selection preserves worst-fold recall first, then pushes the threshold high to reduce false positives.
- Stage 2 single run uses 5-fold CV over included pure/fat examples.
- Stage 2 stability uses 200 repeated randomized stratified 5-fold splits.
- Train-only metrics are diagnostic only.

## Current Dataset

Labeled local positives: **42** total.

Prepared Stage 1b clips:

- `42` shot clips.
- `1110` not-shot clips.
- `1152` total prepared clips.

Prepared negative categories:

- `local_preshot_voice_ambient`: `84`
- `club_bag_equipment`: `118`
- `human_percussive`: `256`
- `non_golf_impacts`: `296`
- `outdoor_ambient`: `201`
- `phone_handling`: `96`
- `whoosh_swing`: `59`

Stage 2 pure/fat rows:

- Included: `33` total = `15` pure + `18` fat.
- Excluded: `9` total = `4` topped, `2` borderline/1mm fat, `3` manual bad-data exclusions (`49`, `67`, `83`).

## Stage 1b Results

Command:

```sh
npm run train:stage1b
```

Deployed log-mel verifier:

- Model: `frontend/models/stage1b_detector.json`
- Report: `data/stage1b_detector_report.json`
- Feature extractor: `logmel_summary`
- Feature count: `206`
- Threshold: `0.79`
- CV: TP `42`, FP `0`, TN `1110`, FN `0`
- Precision `1.000`
- Recall `1.000`
- Specificity `1.000`
- Worst-fold recall `1.000`
- Worst-fold specificity `1.000`
- OOF margin `0.072` (`minPositiveP=0.790`, `maxNegativeP=0.719`)

Handcrafted baseline:

- Threshold: `0.67`
- CV: TP `42`, FP `82`, TN `1028`, FN `0`
- Precision `0.339`
- Recall `1.000`
- Specificity `0.926`
- Worst-fold specificity `0.905`
- OOF margin `-0.302`

Interpretation: the log-mel verifier still cleanly separates this local corpus plus current negatives. The margin stayed positive after adding 14 new positives, but this is still not a production claim because there is no frozen live iPhone holdout yet.

## Onset Validation

Command:

```sh
node frontend/eval_labels.mjs recentered
```

At threshold `0.65` on all 42 labeled files:

- TP `42`
- FP `0`
- FN `0`
- Precision `1.000`
- Recall `1.000`
- F1 `1.000`
- Mean signed offset `-18.6 ms`
- Median absolute offset `21.4 ms`
- Max absolute offset `68.1 ms`

The highest threshold with recall 1.000 in the sweep was `0.90`; file mode still uses `0.65`, which remains a conservative validated setting.

## Stage 2 Pure/Fat Results

Command:

```sh
npm run train:stage2:pure-fat
```

Single 5-fold CV:

- Included examples: `33` = `15` pure + `18` fat.
- Accuracy `0.939`
- Pure precision `0.882`
- Pure recall `1.000`
- Fat precision `1.000`
- Fat recall `0.889`
- OOF margin `-0.209` (`minPureP=0.561`, `maxFatP=0.770`)

With confidence >= `0.60`:

- Coverage `0.939`
- Unsure `2/33`
- Kept accuracy `0.935`
- Kept pure recall `1.000`
- Kept fat recall `0.882`

Repeated randomized 5-fold CV:

```sh
npm run validate:stage2:pure-fat
```

200 repeats:

- Mean accuracy `0.933`
- Median accuracy `0.939`
- Min accuracy `0.848`
- 5th percentile accuracy `0.879`
- 95th percentile accuracy `0.970`
- Max accuracy `0.970`
- Mean pure recall `0.985`
- Mean fat recall `0.890`
- Repeats >= `0.70` accuracy: `200/200`
- Repeats >= `0.80` accuracy: `200/200`
- Repeats >= `0.90` accuracy: `185/200`
- Mean confidence >= `0.60` coverage `0.952`
- Mean kept accuracy at confidence >= `0.60`: `0.949`

Interpretation: Stage 2 improved after adding the new pure/fat examples, especially in repeated-CV stability. The negative OOF separation margin means at least one pure/fat pair still overlaps, so the `unsure` path and per-example review remain important.

## Weakest Stage 2 Examples

Repeated CV weakest examples:

- `53 - fat for sure 1 inch/...`: actual fat, accuracy `0.005`, mean `pPure=0.727`.
- `101 - slightly fat/...`: actual fat, accuracy `0.255`, mean `pPure=0.605`.
- `39 - very pure /...`: actual pure, accuracy `0.785`, mean `pPure=0.580`.
- `90 - fat/...`: actual fat, accuracy `0.870`, mean `pPure=0.336`.

User follow-up confirmed `53` and `101` are definitely fat, so treat those labels as correct. The model calling them pure is a real classifier error on fat examples, not a label issue.

Do not auto-exclude weak examples. Use them as review targets. If visual review says a recording is bad for the pure/fat target, add the exclusion to `data/stage2_pure_fat_exclusions.json` so the decision remains auditable.

## Scale-Up Rules

Use this sequence when adding more samples:

1. Put raw sample folders under `new_data/`.
2. Run `npm run import:new-data -- --dry-run`.
3. If each clip has one strong candidate and the reported times look plausible, run `npm run import:new-data`.
4. If a future clip has multiple plausible candidates, label it manually in the app instead of trusting auto-import.
5. Run `npm run train:stage1b`.
6. Run `npm run train:stage2:pure-fat`.
7. Run `npm run validate:stage2:pure-fat`.
8. Run `node frontend/eval_labels.mjs recentered`.
9. Record the new metrics in a new `ML_*.md` note.
10. Keep a frozen holdout before using metrics as product accuracy.

The current techniques worth keeping are: label-driven positive crops, peak recentering, strict 500 ms clips, peak normalization, local pre-shot hard negatives, external negative onset mining, source-grouped CV, CV-selected Stage 1b thresholds, shared Stage 2 exclusion policy, confidence-gated Stage 2 display, and repeated randomized CV for small-data stability checks.
