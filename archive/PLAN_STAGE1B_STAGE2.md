# Plan — Stage 1b (shot verifier) and Stage 2 (pure/fat/topped classifier)

This plan picks up from `SUMMARY_1.md`. The onset detector is calibrated and produces well-aligned 500 ms 16 kHz mono WAV clips. We now need:

- **Stage 1b**: a small ML model that filters the onset detector's candidates, answering "is this 500 ms window a golf shot, yes/no?"
- **Stage 2**: a small ML model that takes a confirmed shot and labels it pure / fat / topped.

Both models need to run live in the iPhone Safari web app. Both need to be trainable in Python from the labeled dataset we've built. Both need to be replaceable later by native Core ML when we build the iOS app.

---

## The hybrid inference pipeline (live, web)

```
mic stream
  │
  ▼
AudioWorklet ring buffer  ◄──── always running
  │
  ▼
Spectral-flux onset detector (existing, calibrated)        ◄── Stage 1a
  │  fires on candidate onsets
  ▼
Re-crop 500 ms window centered on peak amplitude
  │
  ▼
YAMNet embedding (1024-d vector)                           ◄── shared feature extractor
  │
  ├──► Stage 1b head: shot / not_shot   ── reject if below confidence ──► drop
  │
  ▼
Stage 2 head: pure / fat / topped / unsure
  │
  ▼
UI event: { label, confidence, timestamp, clip }
```

**Why two heads on one embedding extractor**: YAMNet is the expensive part (~0.7 M params, ~5 MB ONNX, runs in ~15–30 ms on iPhone Safari via ONNX Runtime Web). Running it once per detection and attaching two tiny MLP heads is far cheaper than two separate models. Both heads are <50 KB each and train in seconds.

**Why YAMNet specifically**: pretrained on Google's AudioSet (2 M+ human-labeled sounds). It already knows what "impact sounds" / "wood on ball" / "thud on turf" look like in feature space. With our tiny dataset, that prior is worth more than any model we could train end-to-end. Apple's Create ML Sound Classifier uses the same approach with a different pretrained backbone — YAMNet is the closest web-usable analogue, and when we port to native Core ML later we'll swap it for Apple's embedding.

---

## Phase 2.0 — Data pipeline

**Goal**: go from 28 labeled raw recordings to a clean, reproducible training set of 500 ms clips with class labels.

### 2.0.1 Extract training clips

Script `scripts/extract_training_clips.py`:
- Read `data/labels.json`.
- For each entry: ffmpeg-decode the original `.m4a` to 16 kHz mono float32.
- Extract a 500 ms window **centered on `shotTimes[0]`** (the label — ground truth).
- Peak-normalize to −3 dBFS.
- Derive the class from the `folderLabel` field using a mapping table:
  - `pure`, `very pure`, `pure for sure`, `pure 99% sure`, `pure (very)`, `pure for sure small divot`, `pure very pure` → **pure**
  - `fat`, `very fat`, `medium fat`, `for sure fat`, `fat for sure`, `for sure very fat`, `for sure and very fat`, `fat for sure 1 inch` → **fat**
  - `1mm in front`, `great example of 1mm fat` → **fat_borderline** (kept separate; include or exclude per experiment)
  - `topped` → **topped**
- Write WAVs to `data/clips/<class>/<sourcefile>.wav` and append a `clip_manifest.csv` mapping every clip → source file → class → label time → augmentation notes.

**Validation gate**:
- [ ] Every entry in `data/labels.json` produces exactly one clip.
- [ ] Listen to 5 random clips per class — impact clearly audible and roughly centered.
- [ ] `clip_manifest.csv` is complete and regenerable.

### 2.0.2 Cut seed negatives from existing edges

The 28 recordings contain 7–14 minutes of non-shot audio in the pre-shot and post-shot edges. This is our first `not_a_shot` dataset and it's distribution-matched (same mic, same location, same placement).

Script `scripts/extract_negatives.py`:
- For each recording, exclude the labeled shot window ± 600 ms.
- Slide a 500 ms window with 250 ms hop across the remaining audio.
- Reject windows whose peak amplitude is below −35 dBFS (pure silence — not interesting).
- Save to `data/clips/not_a_shot/<sourcefile>_seg<i>.wav`.

This should yield ~400–800 seed negatives from the existing data.

**Validation gate**:
- [ ] Listen to 20 random negatives — none contain an actual shot. Any that do indicate the onset labels missed a shot or the recording contains multiple shots (none should).
- [ ] Count distribution: target 300+ negatives before starting Stage 1b training.

### 2.0.3 Augmentation

Use [`audiomentations`](https://github.com/iver56/audiomentations). For each real clip, generate 10 augmented variants:
- **Time shift**: ±50 ms (critical — makes the model robust to Stage 1a's timing variance).
- **Gain**: ±6 dB.
- **Pitch shift**: ±2 semitones.
- **Time stretch**: ±5%.
- **Additive noise** at SNR {5, 10, 20} dB using randomly selected `not_a_shot` clips as noise beds.
- (Optional) short room reverb.

All augmentations use a fixed random seed. Output to `data/clips_aug/<class>/...` with sidecar JSON recording the augmentation chain per clip.

**Validation gate**:
- [ ] Each real clip produced exactly 10 augmented variants.
- [ ] Listen to 5 random augmented clips per class — plausible, not distorted.
- [ ] Augmentation chain is recorded for every augmented clip.
- [ ] **No leakage across folds**: augmented variants must stay grouped with their source clip for cross-validation splits.

---

## Phase 2.1 — Stage 1b: shot verifier

**Goal**: a YAMNet-embedding + small MLP head that takes a 500 ms clip and outputs P(shot) vs P(not_shot).

### 2.1.1 Training

Script `scripts/train_stage1b.py`:
- Load YAMNet via TensorFlow Hub or the `torchaudio` port. YAMNet expects 16 kHz mono; chunk the 500 ms clip into YAMNet's native ~0.96 s windows (pad/repeat) OR use the patch-level embeddings YAMNet produces internally.
- For each clip: extract embedding (1024-d vector).
- Head: 1024 → 128 (ReLU + dropout 0.3) → 2 logits (shot / not_shot). Softmax output.
- Dataset: all `shot` class clips (augmented) as positives, all `not_a_shot` clips as negatives. Class weights to handle imbalance.
- Training: Adam, lr=1e-3, 50 epochs, early stopping on validation F1.
- Cross-validation: 5-fold at the **source-clip** level (augmented variants stay grouped). Report mean ± stdev per-class recall and F1.

Output: `models/stage1b_vN_<date>.{pt, onnx}`.

**Validation gate**:
- [ ] 5-fold CV recall on `shot` class ≥ **95%** (missing 1 in 20 shots is the ceiling).
- [ ] 5-fold CV precision on `shot` class ≥ **90%** (FP rate will be tightened by the confidence threshold in the app).
- [ ] Per-fold variance is small (stdev < 5 percentage points). Large variance means the model is sensitive to which clips landed in validation — probably means we need more data.
- [ ] Confusion on the borderline "1mm fat" clips: inspect manually. They SHOULD count as shots for Stage 1b (positive class).

### 2.1.2 Export to ONNX

- Trace the YAMNet-backbone + head as a single model.
- Export to ONNX with dynamic batch axis, fixed 500 ms input (8000 samples @ 16 kHz).
- Quantize weights to INT8 to shrink file size and speed up inference. Target: ≤8 MB.
- Verify numerical parity with the Python forward pass on a test batch.

**Validation gate**:
- [ ] ONNX output matches PyTorch output within 1e-4 on 20 test clips.
- [ ] ONNX file size ≤ 8 MB.
- [ ] Model runs in ≤50 ms per inference on iPhone Safari (test via ONNX Runtime Web).

### 2.1.3 Wire into the web app

New file `frontend/stage1b.js`:
- On page load: `ort.InferenceSession.create('/models/stage1b.onnx', { executionProviders: ['webgl', 'wasm'] })`.
- Export a function `async verifyShot(samples: Float32Array) -> { label, confidence }`.

In `app.js` after `onLiveDetected` extracts the 500 ms window, call `verifyShot()` before adding the detection to the UI list. Reject anything with P(shot) < threshold (configurable, default 0.7). Add a "verified by Stage 1b" visual tag.

**Validation gate**:
- [ ] App loads the ONNX model without error on iPhone Safari.
- [ ] Clap test still produces detections (Stage 1b should classify claps as `not_a_shot`, so they now get rejected — this is desirable behavior; add a "show rejected" toggle for debugging).
- [ ] Play back the 28 known shot clips through the mic → every one is verified as a shot.
- [ ] 2 minutes of range ambient through the mic → ≤1 verified "shot" per minute (false positive rate).

---

## Phase 2.2 — Stage 2: pure/fat/topped classifier

**Goal**: same backbone, different head. Takes a verified shot, outputs P(pure), P(fat), P(topped).

### 2.2.1 Training

Script `scripts/train_stage2.py`:
- Dataset: only the `shot` class clips (not `not_a_shot`). Labels from the folder-name parse. Augmented variants of each real clip grouped together.
- Decide early: include `fat_borderline` as its own class OR fold it into `fat`. Try both, report both. Our hypothesis is folding reduces effective class count but adds label noise; keeping separate may work if we can collect more borderline samples.
- Head: 1024 → 128 (ReLU + dropout 0.5 — more regularization since classes are harder to separate than shot/not-shot) → 3 logits. Softmax.
- 5-fold CV at source-clip level.
- Class weights: the current 11:13:4 pure/fat/topped imbalance means topped recall will suffer without weighting. Use `class_weight='balanced'` or loss-function weights inversely proportional to class counts.

Output: `models/stage2_vN_<date>.{pt, onnx}`.

**Validation gate**:
- [ ] 5-fold CV overall accuracy ≥ **70%** (chance on 3 classes = 33%). Prototype target.
- [ ] Per-class recall ≥ **50%** for each of the three classes. If topped is below 50% even with class weights, you need more topped recordings, full stop.
- [ ] Confusion matrix is saved to `models/stage2_vN_confusion.png`. Expected pattern: topped confuses with pure (both "clean" hits, different contact points); fat rarely confuses with topped. If actual confusion doesn't look like this, either the labels are noisy or the model is underfit.
- [ ] Training accuracy vs validation accuracy gap: if >20 pp, note overfitting. Mitigation: more dropout, more augmentation, or (likelier) more data.

### 2.2.2 Dedicated held-out evaluation

Before shipping, set aside ~20% of the real clips (stratified by class) as a **frozen** holdout. These never get augmented; they never touch training. Report final metrics on this set only. This is the one number worth quoting in a product context.

**Validation gate**:
- [ ] Holdout accuracy within 10 pp of 5-fold CV accuracy. Larger gap suggests CV was optimistic (usually: data leakage, over-strong augmentation, or an over-uniform training set).

### 2.2.3 Confidence calibration

- Bucket predictions by top-class confidence in 10% buckets.
- Plot confidence vs actual accuracy. A well-calibrated model: 0.8 confidence predictions are right ~80% of the time.
- Pick a confidence threshold for the app. Below it, show "unsure" instead of a class label.

**Validation gate**:
- [ ] Calibration plot is monotonic (no buckets where higher confidence means lower accuracy).
- [ ] Confidence threshold chosen such that predictions above it are right ≥85% of the time.

### 2.2.4 Wire into the web app

Extend `frontend/stage1b.js` (rename to `stage2.js` or add a second model loader):
- Load `/models/stage2.onnx`.
- Export `async classifyShot(samples) -> { label, confidence, perClassProb }`.

Wire the pipeline: `onset → Stage 1b verify → if shot, Stage 2 classify → emit labeled event`.

UI:
- Big label ("PURE" / "FAT" / "TOPPED" / "—").
- Confidence bar.
- Recent-shots list with label + confidence + play button.
- "Correct?" thumbs up/down per shot for in-app feedback labeling.

**Validation gate**:
- [ ] Full pipeline runs on iPhone Safari at ≥5 fps (onset→verify→classify latency <200 ms).
- [ ] Known clips played back → classified correctly ≥80% of the time.
- [ ] Correction UI logs every prediction + user judgment locally for the next training round.

---

## Phase 2.3 — Cloud data-collection loop

**Goal**: every practice session grows the dataset. Once deployed, users testing the app produce labeled data that feeds the next model.

### Architecture (matches the original roadmap)

- **Frontend**: static deploy (Vercel / Cloudflare Pages). HTTPS required for mic access.
- **Backend**: FastAPI + SQLite + local disk on Fly.io / Railway. Endpoints:
  - `POST /clips` — upload a detection clip with metadata (tester, timestamp, predicted label, confidence, user-corrected label).
  - `GET /clips` — browse/filter clips for the labeling queue.
  - `POST /verify` (Phase 2.4+) — server-side re-inference during model iteration.

### Phasing inside Phase 2.3

- **2.3.a — MVP backend**: just `POST /clips` + local storage. Deploy. Frontend uploads every detection plus user-corrected label.
- **2.3.b — Labeling queue UI**: a `/review` page showing uploaded clips sorted by (confidence desc, user_corrected != predicted). You review, confirm/correct, write back.
- **2.3.c — Retraining on demand**: a script that pulls clips from the backend, bundles them with the original 28, retrains Stage 1b and Stage 2, evaluates on the frozen holdout, and ships a new ONNX if and only if per-class metrics are at least as good.

**Validation gate — overall Phase 2.3**:
- [ ] Every detection in a practice session is persisted with its user correction.
- [ ] Retraining from the accumulated dataset produces a model at least as good per-class as the previous one.
- [ ] Distribution drift gets logged (e.g. accuracy dips on iPhone 17 Pro, cold weather, a new turf type) so we know where the next data-collection effort should focus.

---

## Phase 2.4 — Native iOS port (optional, later)

When the web app proves the concept and users want better latency / background processing / no Safari quirks:

1. Export the labeled dataset from the backend (JSONL + WAVs).
2. Retrain the same heads with Apple's Create ML Sound Classifier template (its embedding backbone differs from YAMNet but serves the same role — we throw away the ONNX models and start from the same labeled data).
3. Ship a native SwiftUI app using `SNClassifySoundRequest` + `SNAudioStreamAnalyzer` for live inference. See `PROJECT.md` Phase 6.
4. Keep the web app running as a data-collection tool even after the native app ships.

---

## What success looks like, concretely

### End of Phase 2.1 (Stage 1b done)
- Web app detects shots live on an iPhone at a driving range.
- ≥95% of hit balls produce a verified "shot" event within 200 ms.
- Non-shot sounds (voices, claps, cart noise) are rejected ≥95% of the time.
- Click-to-confirm UI captures the true label for every detection.

### End of Phase 2.2 (Stage 2 done)
- Every verified shot gets a pure / fat / topped label with calibrated confidence.
- 80% per-class accuracy on the held-out dataset. Good enough to be useful, not good enough to trust blindly — gate low-confidence predictions as "unsure."
- Confusion matrix looks the way we expect (topped ↔ pure is the main confusion; fat is typically distinct).

### End of Phase 2.3 (flywheel running)
- New practice sessions grow the labeled dataset automatically.
- Periodic retraining closes gaps surfaced by user corrections.
- We know what distribution-drift failures exist because we're watching for them, not because a user filed a complaint.

---

## What to resist

- **Shipping a Stage 2 model trained only on Rancho Park data and calling it a classifier.** It's a Rancho-Park-specific classifier until proven otherwise. Collect at ≥2 courses before trusting the numbers.
- **Training Stage 2 before Stage 1b is solid.** Without a good filter, Stage 2 wastes compute and context space on non-shot inputs. Stage 1b first.
- **Overfitting on 4 topped samples.** Until you have ≥20 topped recordings, Stage 2's topped class is a lucky guess, not a signal. Consider explicitly dropping topped and shipping pure-vs-fat until the topped count catches up.
- **Ignoring the feedback UI.** Every shipped model should be evaluable from day one via the in-app correction buttons. Shipping without that is shipping blind.
- **Adding on-device iOS work before the web MVP teaches us what the model can actually do.** The web path exists specifically to surface classification reliability problems cheaply.

## Tooling commitments

- **Training**: Python 3.11+, PyTorch + torchaudio, `audiomentations`, TensorFlow Hub (for YAMNet loader; export via `torch.onnx.export` after porting).
- **Inference**: ONNX Runtime Web in the frontend, Apple Neural Engine via Core ML in the eventual iOS app.
- **Data**: SQLite for metadata, filesystem for audio.
- **Reproducibility**: fixed random seeds in every training run; ONNX model export includes a hash of the training-manifest CSV so each shipped model is traceable to exactly the data it saw.
