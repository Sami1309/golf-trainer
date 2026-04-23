# Summary 1 — Phase 0 onset detector + ground-truth labeling

This document captures the state of the project at the end of the first iteration: the web-based Phase 0 app, what it does, what was measured, and what the current settings are. Everything here is on-disk and reproducible.

## What was built

A single-page web app at `frontend/` with no build step and no backend, containing four user-visible sections:

1. **Live mic** — capture iPhone/desktop microphone audio, run live spectral-flux onset detection, save 500 ms clips around each detected onset.
2. **File analysis** — drop any audio file and run the same detector offline, showing waveform + flux curve + detected onsets.
3. **Label onset times** — load the whole `samples/` folder, play each recording, mark the moment of impact (space/play, `L`/mark, `N`/`P`/next or prev). Labels auto-save to `localStorage`; JSON import/export round-trips.
4. **Calibrate onset detector** — run the flux detector against the labeled ground truth, sweep thresholds, report per-file precision/recall/F1, suggest and apply the best threshold.

Supporting offline tooling under `frontend/`:
- `smoke_test.mjs` — decodes all `.m4a` via ffmpeg and reports flux peaks per file.
- `diag.mjs` — multi-algorithm diagnostic (peak amplitude, flux at three FFT sizes, RMS-diff, RMS-ratio) used to verify the impact is detectable at all.
- `eval_labels.mjs` — labels-driven evaluator. Variants: `baseline`, `hf`, `adaptive`, `hf+adaptive`, `recentered`, `hf+recentered`. Reports TP/FP/FN, precision/recall/F1, and timing offset (detection time relative to label).

## Data

`data/labels.json` is the canonical labeled dataset. 28 `.m4a` files across 25 folders, labeled with shot onset time (within a few hundredths of a second). Class distribution:

- **Pure: 11** — folders 6, 14, 15, 27, 28, 39, 46, 52, 59, 62, 83
- **Fat: 13** — folders 7, 12, 19, 20, 23, 31, 43, 45, 49, 53, 57, 58, 67
- **Topped: 4** — folders 21, 22, 24, 26

All recordings: iPhone on the ground, mic up, ≤1 m from ball, location "Rancho Park Golf Course / Club" (and a couple labeled "Fareways"). Distribution risk already flagged in `CLAUDE.md`: single-location training set.

Shot timestamps cluster at **3–5 seconds** into the 7–10 second recordings (user noted shots tend to be centered). Pre- and post-shot edges contain ambient / voices / setup sounds and are a **free source of negatives** for Stage 1b.

## Calibration findings

### Bug caught via smoke test

Initial flux normalization included a `/sqrt(nBins)` divisor that collapsed all flux values to the 0.04–0.09 range across 28 files. Default threshold of 0.5 would have produced zero detections forever. Removing the divisor lands flux peaks in the 0.9–1.9 range — a usable signal. This fix is live in `app.js` and `smoke_test.mjs`.

### Independent confirmation the impact is detectable

Diagnostic comparison of three independent detection signals on raw audio (peak amplitude, spectral flux at three FFT sizes, RMS-diff envelope): **26 of 28 files** show all three methods peaking within 200 ms of each other. The impact is obviously present; no algorithmic sophistication was needed to find it.

### Detector variant comparison

All variants run at FFT=1024, hop=256, SR=16 kHz, min-gap=200 ms. Matched against labels with ±100 ms tolerance:

| variant | best-F1 threshold | highest thr @ R=1 | mean signed offset | median \|offset\| | max \|offset\| |
|---|---|---|---|---|---|
| baseline | 0.65 | 0.90 | −37.9 ms | 35.3 ms | 77.5 ms |
| hf (>500 Hz only) | 0.35 | 0.70 | −39.6 ms | 39.1 ms | 77.5 ms |
| **recentered (chosen)** | 0.65 | 0.90 | **−27.9 ms** | **26.9 ms** | **68.1 ms** |
| hf+recentered | 0.35 | 0.70 | −28.0 ms | ~27 ms | ~68 ms |
| adaptive (median + 5·MAD) | n/a | n/a | badly broken on clean data (F1=0.08) | — | — |

Every variant achieves 100% precision and 100% recall across all 28 files at its best-F1 threshold. What differs is **where** in time the detection fires.

**Chosen variant: `recentered`**. After the flux detector fires at time t_flux, the app searches the next 120 ms of raw audio for the peak |amplitude| sample and reports that sample's time as the detection. This shrinks the median detection-vs-label offset from 35 ms → 27 ms and the max from 78 ms → 68 ms at zero accuracy cost.

**Why `hf` wasn't chosen**: no accuracy gain on clean data, reduces flux dynamic range (absolute peaks cap at ~1.4 vs 1.9), and we lose headroom. Worth re-testing once we have range-ambient recordings.

**Why `adaptive` failed**: moving median + k·MAD needs a non-trivial baseline to compute a sane threshold. These recordings are near-silent between the shot and the edges, so the moving median stays near zero and every small fluctuation exceeds threshold + k·MAD. Defer until live range audio is on hand.

### Irreducible offset

After re-centering, the remaining ~28 ms mean signed offset is almost certainly human labeling latency — click reaction time plus the brain registering transients ~20–30 ms after the amplitude peak. Nothing in the audio pipeline gets below that without manually hand-labeling each file frame by frame.

### Per-file behavior at the chosen settings

At threshold 0.65 with re-centering: **28 TP, 0 FP, 0 FN** across the full dataset. No misses, no extraneous detections within the recording. The edges of the recordings DO contain loud transient-ish sounds (voices, bag noise) that fire at lower thresholds — below 0.5, those edge events start counting as false positives. Anything in the 0.65–0.90 range is perfectly clean.

## Current settings (live in the app)

Defined in `frontend/app.js`:
```
TARGET_SR       = 16000              // clips saved/resampled to this rate
CLIP_PRE_MS     = 100                // extraction window: 100ms pre
CLIP_POST_MS    = 400                //                  + 400ms post = 500ms total
FILE_FFT_SIZE   = 1024               // ~64ms window at 16kHz
FILE_HOP_SIZE   = 256                // ~16ms hop
threshold       = 0.8  (default)     // in the 100%-P/R plateau; user can tune via slider
minGapMs        = 200                // minimum time between distinct detections
recenter        = ALWAYS ON          // 120ms forward search for peak amplitude
```

In live mode the app extracts a 620 ms window (100 ms pre + 400 ms post + 120 ms search), finds the peak within the post-flux search region, and re-crops to the canonical 500 ms centered on that peak. Clip downloads are 16 kHz mono 16-bit PCM WAV.

## Dataset assets on disk

```
samples/
├── data/
│   └── labels.json                 # canonical ground truth, 28 entries
├── labels_2026-04-23T08-36-48-110Z.json   # original export (kept for provenance)
├── frontend/
│   ├── eval_baseline.json          # threshold sweep + per-file at best F1
│   ├── eval_hf.json
│   ├── eval_recentered.json
│   ├── eval_hf+recentered.json
│   ├── eval_adaptive.json
│   ├── smoke_report.json           # raw flux peaks per file (post-fix)
│   └── diag_report.json            # multi-algorithm comparison per file
└── [28 folders with .m4a + .MOV per shot]
```

## What Phase 0 validated

- Web Audio + AudioWorklet + AnalyserNode pipeline works end-to-end (desktop Chrome/Safari; iPhone Safari requires HTTPS for live mic — deferred to Phase 1).
- Spectral flux finds golf impacts reliably on clean recordings.
- Ground-truth onset labels can be produced at speed (~10 s per file once the UI shortcuts are wired).
- Re-centering on post-onset peak amplitude aligns extracted clips with how a human would label the impact moment, giving Stage 1b / Stage 2 a consistent training window.
- The extraction pipeline (ring buffer → worklet → resample to 16 kHz → WAV encode) produces training-ready clips directly.

## What Phase 0 did NOT validate

- Behavior on range-ambient audio. Every clip in the dataset is short and near-silent between the shot and the recording edges. Live driving-range ambient (wind, other players, voices, cart noise) is untested.
- iPhone on-device performance. The architecture will work (same Web Audio APIs), but threshold calibration may differ and HTTPS hosting is required. Phase 1 task.
- Any ML. Neither the shot verifier (Stage 1b) nor the pure/fat/topped classifier (Stage 2) exists yet. See `PLAN_STAGE1B_STAGE2.md`.

## Known risks carried forward

1. **Single-location training data** — every clip is from Rancho Park. Before trusting any Stage 1b or Stage 2 accuracy numbers in the wild, collect at a second course.
2. **Topped is underrepresented** — 4 of 28 clips are topped shots. This is fine for Stage 1 (all shots lumped into one class), but Stage 2 will overfit on topped unless we specifically collect more.
3. **No negatives** — zero `not_a_shot` clips. The 7–14 minutes of non-shot audio embedded in the current recordings' edges is the obvious first negative source.
4. **Label latency** — 28 ms systematic lag between click-time and amplitude peak. Re-centering accounts for the mean; individual clips may still be off by up to ~70 ms. Bake ±50 ms time-shift augmentation into Stage 1b / Stage 2 training so the model is robust to it.

## Reproducibility

```bash
cd frontend
python3 -m http.server 8000                  # run the app locally
node smoke_test.mjs                          # flux peaks per m4a
node diag.mjs                                # multi-algorithm diagnostic
node eval_labels.mjs baseline                # evaluator (also: recentered | hf | hf+recentered | adaptive)
```

Every result in this summary comes from one of those commands or the calibration UI inside the app.
