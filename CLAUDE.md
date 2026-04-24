# Golf Shot Audio App

Current status note: `AGENTS.md` is the canonical first-read orientation. This file is kept in sync as a shorter handoff for Claude-style agents.

## Goal

Build an iPhone-ready golf-shot audio app:

1. Listen through an iPhone mic near the ball.
2. Detect candidate impact onsets in the browser.
3. Verify shot vs not-shot with a lightweight Stage 1b model.
4. Classify accepted shots as pure/fat/unsure now, and later add topped.
5. Run first as a mobile web app in iPhone Safari. Native iOS is later.

## Current App State

The web app in `frontend/` is the active product surface. It is vanilla JS with no build step.

Current product milestone: **mobile web v1 data-collection app**. The functional flow is in place. Do not spend more time hand-polishing the UI before the planned design/mockup input is available.

Live/file pipeline:

1. Stage 1a spectral-flux onset detector finds candidate transients.
2. A 500 ms clip is extracted, recentered around impact, resampled to 16 kHz mono, and peak-normalized.
3. Stage 1b loads `frontend/models/stage1b_detector.json` and rejects non-shot candidates.
4. Accepted candidates run Stage 2 pure/fat classification from `frontend/models/stage2_pure_fat.json`.
5. Accepted candidates are stored locally in IndexedDB with both the model clip and a configurable review clip; Stage 1b rejected onsets are also stored by default as `no_shot` negatives when the data-collection option is enabled.
6. Users can label clips in the app and export a ZIP with WAVs plus manifest JSON.

Live session flow:

- Start recording arms calibration automatically.
- The first heard shot becomes a pending calibration shot.
- The user confirms or retries that calibration shot.
- Confirmed calibration puts the app into live mode.
- Live mode shows the most recent accepted shot and the pure/fat/unsure quality estimate.
- Review clips default to 5 seconds so the user can speak labels around the shot.
- Live audio uses a 12 second worklet ring buffer so review clips can include post-shot context.
- Export writes `manifest.json`, context WAVs, and canonical 500 ms model WAVs.

The `quality` column in the app is the Stage 2 pure/fat classifier result. It is a useful live comparison signal, not final trusted coaching feedback.

## Current Models

Stage 1b deployed detector:

- Model: `frontend/models/stage1b_detector.json`
- Report: `data/stage1b_detector_report.json`
- Feature extractor: `logmel_summary`
- Feature count: `206`
- Threshold: `0.79`
- CV: TP `42`, FP `0`, TN `1110`, FN `0`
- OOF margin: `0.072` (`minPositiveP=0.790`, `maxNegativeP=0.719`)

Stage 2 pure/fat v0:

- Model: `frontend/models/stage2_pure_fat.json`
- Report: `data/stage2_pure_fat_report.json`
- Repeated CV report: `data/stage2_pure_fat_repeated_cv_report.json`
- Curation policy: `data/stage2_pure_fat_exclusions.json`
- Feature extractor: `logmel_summary`
- Feature count: `206`
- Confidence threshold: `0.60`
- Included examples: `33` = `15` pure + `18` fat
- Excluded examples: `9` = `4` topped, `2` borderline/1mm fat, `3` bad-data visual-review exclusions
- Single 5-fold CV accuracy: `0.939`
- Confidence >= `0.60`: coverage `0.939`, kept accuracy `0.935`
- Repeated randomized 5-fold CV over 200 repeats: mean accuracy `0.933`, min `0.848`, max `0.970`
- Repeated-CV confidence >= `0.60`: mean coverage `0.952`, mean kept accuracy `0.949`
- OOF margin: `-0.209`

Interpretation: Stage 2 has stronger signal in the current local corpus. It does not prove production generalization because there are only 33 included examples from local recording domains and the OOF margin is negative.

## Data Rules

Do not train on full local `.m4a` positives. The raw recordings include spoken shot names before impact.

Correct handling:

- Stage new one-shot raw folders under `new_data/` and import with `npm run import:new-data`.
- Use `data/labels.json` as canonical shot timing and folder-label source.
- Crop positives around labeled impact time.
- Recenter near impact using peak amplitude.
- Use 500 ms clips at 16 kHz mono.
- Peak-normalize to `-3 dBFS`.
- Treat pre-shot local audio, including spoken labels, as negative or ignore it.
- Never rename or overwrite top-level raw shot folders.

Stage 2 curation:

- Topped examples are excluded from pure/fat v0.
- `1mm`/borderline fat examples are excluded from pure/fat v0.
- Manual bad-data exclusions live only in `data/stage2_pure_fat_exclusions.json`.
- Do not hard-code manual exclusions in trainer scripts.

## Commands

Syntax checks:

```sh
node --check frontend/audio_features.js
node --check frontend/stage1b.js
node --check frontend/stage2.js
node --check frontend/app.js
node --check scripts/import_new_data.mjs
node --check scripts/train_stage1b_detector.mjs
node --check scripts/train_stage1b_logmel.mjs
node --check scripts/stage2_pure_fat_policy.mjs
node --check scripts/train_stage2_pure_fat.mjs
node --check scripts/validate_stage2_repeated_cv.mjs
```

Training:

```sh
npm run import:new-data -- --dry-run
npm run import:new-data
npm run train:stage1b
npm run train:stage2:pure-fat
npm run validate:stage2:pure-fat
```

Onset validation:

```sh
node frontend/eval_labels.mjs recentered
```

Known good onset result: at threshold `0.65`, TP `42`, FP `0`, FN `0`, mean signed offset about `-18.6 ms`, median absolute offset about `21.4 ms`.

Run frontend locally:

```sh
cd frontend
python3 -m http.server 8000
```

For iPhone mic testing, Safari needs HTTPS. Use ngrok, Cloudflare Tunnel, Tailscale Funnel, or static HTTPS hosting.

## Git And Data Policy

Track in Git:

- Source under `frontend/` and `scripts/`.
- Docs and progress reports.
- Canonical metadata such as `data/labels.json`, `data/external/manifest.*`, `data/stage1b_prepared/manifest.jsonl`, and `data/stage2_pure_fat_exclusions.json`.
- Small model/report JSON under `frontend/models/` and `data/*_report.json`.
- `.gitignore` and `.gitattributes`.

Do not track in normal Git:

- Raw top-level `.m4a` and `.MOV` shot recordings.
- Generated prepared WAV clips under `data/stage1b_prepared/shot/` and `data/stage1b_prepared/not_shot/`.
- External downloaded audio, future holdout/session audio, browser exports, local settings, caches, or `.DS_Store`.

Workflow:

- Check `git status --short --ignored` before staging data changes.
- Prefer explicit path staging or `git add -p`.
- Commit report/model JSON only when the metrics are intentionally accepted.
- Never use destructive reset/checkout commands unless the user explicitly asks for that exact operation.

## Best Resume Point

Next useful product work:

1. Take the user's design/mockup input for the v1 mobile UI.
2. Apply it to the existing functional flow in `frontend/`.
3. Write a static HTTPS deployment guide.
4. Plan the web-v1-to-native-iPhone path.

Next useful model/data work is live iPhone/range validation:

1. Serve `frontend/` over HTTPS.
2. Open on iPhone Safari.
3. Start recording and confirm the calibration shot.
4. Hit real shots and deliberately create non-shot transients.
5. Export accepted and rejected clips.
6. Label pure/fat/topped/no-shot in the app.
7. Set aside a frozen holdout before adding new clips to training.
8. Add live false positives as Stage 1b hard negatives.
9. Retrain Stage 1b and Stage 2, then compare against the current committed baseline.
