# Hybrid Golf Shot Detector (web)

Single-page web app for the Stage 1 hybrid detector:
- Captures mic audio via Web Audio API
- Runs spectral-flux onset detection live (via `AnalyserNode` polled at ~60 Hz)
- Runs a Stage 1b shot verifier model on each detected 500 ms clip
- Runs an experimental Stage 2 pure-vs-fat classifier on accepted shots
- Runs the same algorithm offline against uploaded audio files
- Extracts a canonical 500 ms model clip plus a 2 second review clip around each detection
- Stores detections locally in IndexedDB for review after reloads
- Plays / labels / exports detections as WAVs plus a JSON manifest

No backend and no build step. ZIP export uses JSZip from jsDelivr in the browser.

The default verifier is a small JSON logistic model loaded from
`frontend/models/stage1b_detector.json`. The full Stage 1b training flow is:

```sh
npm run train:stage1b
```

That rebuilds the prepared clips, trains the handcrafted baseline, then trains
and deploys the log-mel verifier. Individual commands are also available:

```sh
node scripts/train_stage1b_detector.mjs
node scripts/train_stage1b_logmel.mjs
```

It writes:

- `frontend/models/stage1b_detector.json` — browser-loaded verifier model.
- `frontend/models/stage1b_logmel.json` — log-mel deployed model copy.
- `frontend/models/stage1b_handcrafted.json` — handcrafted baseline copy.
- `data/stage1b_detector_report.json` — metrics and false-positive details.
- `data/stage1b_logmel_report.json` — log-mel metrics and false-positive details.
- `data/stage1b_handcrafted_report.json` — handcrafted baseline metrics.
- `data/stage1b_prepared/` — regenerated 500 ms training clips used for Stage 1b.

The pure-vs-fat Stage 2 model is also a small JSON logistic model loaded from
`frontend/models/stage2_pure_fat.json`. Train it after the Stage 1b prepared clips exist:

```sh
npm run train:stage2:pure-fat
```

Stage 2 v0 intentionally excludes topped examples, 1mm/borderline fat examples, and three visually reviewed bad-data examples recorded in `data/stage2_pure_fat_exclusions.json`. Current local CV is promising: 19 included examples, single 5-fold CV accuracy 0.895, and 0.944 kept accuracy at confidence >= 0.60. It is useful for live comparison and data collection, not yet a production swing-quality classifier.

Important: positives are cropped from `data/labels.json` around the labeled impact time. The script intentionally does **not** train on the full local `.m4a` positives because those recordings contain spoken shot names before impact. It now adds pre-impact crops from those same local files as hard negatives, so spoken pre-roll is treated as `not_shot` instead of leaking into positives. External positives are not used yet; current sourced positives are only copies of the same local recordings.

## Run locally

From the `frontend/` directory:

```sh
python3 -m http.server 8000
```

Then open http://localhost:8000/?tester=YOURNAME in a desktop browser (Chrome or Safari both work). Mic access works on `http://localhost` without HTTPS.

## Testing on iPhone

Safari requires HTTPS for `getUserMedia`. Options, easiest first:

1. **ngrok** (fastest for a one-off test):
   ```sh
   # terminal 1
   python3 -m http.server 8000
   # terminal 2
   ngrok http 8000
   ```
   Open the `https://*.ngrok.io` URL on your iPhone. Grant mic permission when prompted.

2. **Tailscale Funnel** or **Cloudflare Tunnel** — same idea, different provider.

3. **Deploy to Vercel / Cloudflare Pages** — static hosting, free tier, auto-HTTPS. Drag `frontend/` into a Vercel project. This is where Phase 1 will end up anyway.

## What to test

1. **Sanity check on desktop**: open, click Start, clap. You should see the green flux curve spike. If the verifier rejects it, the accepted/total counter will show something like `0/1`; enable **show rejected** to inspect it.
2. **One-shot live calibration**: click **Calibrate next shot**, then hit one real ball. The app temporarily lowers the onset gate to catch that shot, measures one dimension (`shot strength` / spectral flux), and sets the live onset threshold to 65% of that measured value.
3. **File mode on known recordings**: upload any `.m4a` from the sample folders. File mode uses the calibrated labeled-sample threshold (`0.65`) instead of the live slider. The waveform should render with pink vertical lines at each detected onset.
4. **Threshold tuning**: if live mode misses shots, use one-shot calibration rather than hand-tuning first. If everything fires, raise the threshold slightly.
5. **iPhone live at range**: open URL, phone on ground, calibrate with one real shot, then hit a few more. Check the detections table for real shots vs false positives and compare the pure/fat quality guess against your label.
6. **Export**: hit Export All after a session. The app writes a ZIP with `manifest.json`, `clips/context/` 2 second review WAVs, and `clips/model_500ms/` canonical model clips.

## Known gotchas

- **iOS requires a user gesture** before starting audio. That's what the Start button is for; don't auto-start on page load.
- **AudioContext runs at 48 kHz on most iPhones**, not 16 kHz. The app uses the native rate internally and resamples only when saving clips. This is intentional — forcing a context rate is historically flaky on Safari.
- **Mic feedback**: the app does NOT route mic audio to the speakers. If you hear your own voice, something's wrong.
- **Tab must stay foregrounded**. Safari suspends AudioContext in background tabs; Phase 1 (with a backend and background sync) will handle this better.
- **Threshold units are arbitrary**. The spectral flux magnitude depends on FFT size, mic gain, and the AnalyserNode's dBFS scale. Treat the threshold as "knob I turn until detections line up with real events."

## What validates Phase 0

- [ ] On desktop: clap fires a detection, silence doesn't.
- [ ] On desktop file-mode: every one of the 28 sample `.m4a` files produces ≥1 detection, and at least one of those detections, played back, clearly contains the impact.
- [ ] On iPhone: mic permission granted, live detection works at a driving range.
- [ ] Exported ZIP contains valid 16 kHz mono 16-bit WAV files that play everywhere.

Failing any of these before adding a backend means the pipeline has a problem we need to solve first, not paper over with more layers.

## What this does NOT do yet

- Pure/fat classifier is experimental and trained on only 19 included local examples.
- No topped classifier yet.
- No multi-device sync / backend.
- No spectrogram (yet).
- The verifier was trained only on current shots plus edge negatives. It still needs deliberate hard negatives: practice swings, claps, bag drops, club taps, phone bumps.
