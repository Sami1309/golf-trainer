# Project Plan: Golf Shot Sound Classifier

This document is the step-by-step plan. Each phase has a **Validation gate** — a concrete, measurable check that must pass before moving on. Failing a gate is information, not an embarrassment: it tells you where to invest next.

---

## Phase 0 — Data audit and reorganization

**Goal**: know exactly what data we have, move it into a clean structure, and surface labeling ambiguities.

**Steps**:
1. Inventory every folder in `samples/`. For each, note: current label (from folder name), audio file, video file, duration, and any qualitative notes (e.g. "1mm fat — borderline").
2. Create `data/raw/<class>/` directories: `pure`, `fat`, `topped`, `not_a_shot`.
3. Copy (don't move — keep originals) audio files into `data/raw/<class>/` with descriptive filenames: `pure_001_rancho_20260422.m4a`.
4. Build `data/labels.csv` with columns: `processed_path, original_path, class, confidence (high/medium/low), notes`. Mark the "1mm fat" samples as `confidence=low`.
5. Decide, explicitly and in writing, how to handle the 1mm-fat edge cases: (a) include as `fat`, (b) exclude from training, or (c) add a fifth class `borderline_fat`. Pick one and record the choice here.

**Validation gate**:
- [ ] Every audio file in `samples/` appears in `data/labels.csv`, and every row in the CSV points to a file that exists.
- [ ] Counts per class printed and recorded here: `pure=__, fat=__, topped=__, not_a_shot=0 (expected, collect in Phase 1)`.
- [ ] A second person (or a second listening session 24h later) spot-checks ≥20% of the labels. Disagreements are resolved or the sample is marked `confidence=low`.
- [ ] Decision on 1mm-fat samples is written into `data/labels.csv` notes column AND into this file (below).

**1mm-fat handling decision** (fill in when made): _TBD_

---

## Phase 1 — Collect more data

**Goal**: get to a dataset you can actually train on. Current ~9 samples will not produce a trustworthy model no matter how good the pipeline is.

**Targets**:
- **Prototype threshold**: 100 clips per class (~400 total including `not_a_shot`). Enough to see if the approach works at all.
- **Trust threshold**: 500+ clips per class. Required before believing the accuracy numbers.

**Steps**:
1. **Record new sessions**. Vary deliberately: multiple clubs (driver, 3-wood, iron), multiple players if possible, multiple turf types (range mat, real grass, indoor mat), multiple mic distances (0.3m, 1m, 2m), multiple days/weather/time-of-day.
2. **Collect `not_a_shot` samples**: 100+ clips of range ambient (wind, distant voices, cart noise, practice swings without a ball, club waggles). These will be the negative class in the live app.
3. **Search public sources** for additional labeled golf impact sounds (YouTube slow-mo golf compilations, Freesound.org, any published golf audio datasets). Any externally-sourced clip must be **listened to and labeled by you personally** before joining the dataset — YouTube comment labels are noise.
4. **Label new clips** into `data/raw/<class>/` and append to `data/labels.csv`.

**Validation gate**:
- [ ] ≥100 clips per class in `data/raw/` (prototype threshold). If short, document the gap and proceed only with eyes open.
- [ ] Each class has clips from ≥3 distinct recording sessions (different day, location, or mic setup).
- [ ] `not_a_shot` class has ≥100 clips covering ambient, voices, practice swings.
- [ ] Label audit: random-sample 20 clips per class, re-listen, agreement rate ≥95%.
- [ ] One recording session is fully set aside as `data/holdout/` and **never touched** until Phase 5 evaluation. Different day from everything else in training.

---

## Phase 2 — Preprocessing pipeline

**Goal**: turn raw, variable-length recordings into consistent 500ms 16kHz mono WAV clips centered on the impact.

**Steps**:
1. `scripts/extract_audio.py` — convert `.MOV` and `.m4a` in `data/raw/` to 16 kHz mono WAV in a temp directory. Use `ffmpeg` under the hood.
2. `scripts/detect_onset.py` — for each clip, detect the impact using spectral flux onset detection (`librosa.onset.onset_detect`). Extract a 500ms window: 100ms before, 400ms after. Fall back to energy-based detection for clips where spectral flux finds nothing.
3. `scripts/normalize.py` — peak-normalize each clip to -3 dBFS.
4. Write final clips to `data/processed/<class>/<original_name>.wav`.
5. Update `data/labels.csv` with a `processed_path` column.

**Validation gate** (this one is critical — a bad onset detector will silently wreck everything downstream):
- [ ] All clips in `data/processed/` are exactly 500ms, 16 kHz, mono, 16-bit PCM. Script asserts this.
- [ ] Peak amplitude of every clip is within ±1 dB of -3 dBFS.
- [ ] **Manual listening check**: open 10 random clips per class in Audacity (or equivalent). Confirm the impact is clearly audible and roughly centered at the 100ms mark. If onset detection mis-fires on >10% of clips, fix the detector before proceeding.
- [ ] Visual check: plot log-mel spectrograms of 5 random clips per class side-by-side. The three classes should look visually distinguishable. If they don't, either the detector is broken, the labels are wrong, or the classes genuinely overlap too much for this approach.

---

## Phase 3 — Data augmentation

**Goal**: turn N real clips into roughly 10–20N effective training examples by simulating the variation a real deployment will see.

**Augmentations to apply** (use `audiomentations`):
- Additive background noise at SNR 5, 10, 20 dB (use your collected `not_a_shot` clips as noise beds — this is why you collected them).
- Pitch shift: ±2 semitones.
- Time stretch: ±10%.
- Random gain: ±6 dB.
- Small time shift within the 500ms window: ±50ms.
- (Optional) short room reverb, low-pass filter to simulate mic-in-pocket.

**Steps**:
1. `scripts/augment.py` — for each clip in `data/processed/`, generate 10 augmented variants in `data/augmented/<class>/`. Record the augmentation chain used for each output in a sidecar JSON or in `data/labels.csv`.
2. Use a fixed random seed. Make the script regenerable.

**Validation gate**:
- [ ] Each real clip produced the expected number of augmented variants.
- [ ] **Listen to 5 random augmented clips per class**. They should sound like plausibly real shots recorded in different conditions — not glitchy, not over-processed, not unrecognizable.
- [ ] Augmented clips are **only** added to training folds, **never** to validation or test folds. A clip's augmented versions must not leak across the fold boundary, or your cross-validation accuracy will be optimistic. Group augmented variants by their source clip and split at the source-clip level.

---

## Phase 4 — Train the baseline model

**Goal**: a Create ML Sound Classifier that runs on iPhone and does better than chance.

**Steps**:
1. Organize training data into Create ML's expected folder structure: one folder per class inside a parent directory, pointing at `data/processed/` + `data/augmented/` (with the fold-level separation from Phase 3 respected).
2. Train via the Create ML app (simplest) or via the `CreateML` framework in a Swift script (scriptable, better for cross-validation). Save as `models/v0_<date>.mlmodel`.
3. Run **5-fold cross-validation** at the source-clip level. Record mean ± stdev accuracy across folds.
4. Generate a **confusion matrix** and per-class precision/recall.
5. Listen to every misclassified clip in the validation folds. Write down patterns.

**Validation gate**:
- [ ] Mean cross-validation accuracy ≥60% (chance on 4 classes = 25%, so 60% is a real signal).
- [ ] Per-class recall ≥50% for each class. If one class is near 0% recall, you have a data imbalance or labeling problem — fix before tuning.
- [ ] Confusion matrix is saved to `models/v0_<date>_confusion.png` and reviewed.
- [ ] Training accuracy is not wildly higher than validation accuracy. If training = 100% and validation = 65%, note overfitting and plan more data / more regularization rather than celebrating.
- [ ] Failure analysis notes written: top 3 patterns in misclassified clips (e.g. "mic close to impact → fat sounds too loud, classified as pure").

---

## Phase 5 — Held-out evaluation

**Goal**: an honest estimate of how the model will perform in the wild.

**Steps**:
1. Preprocess (but do **not** augment) the Phase 1 holdout set through `scripts/detect_onset.py` + `scripts/normalize.py`.
2. Run the trained model against every holdout clip. Record predictions and confidence scores.
3. Build a holdout confusion matrix and per-class precision/recall.
4. **Confidence calibration check**: bucket predictions by confidence (0.5–0.6, 0.6–0.7, ..., 0.9–1.0). Plot confidence vs. actual accuracy. In a well-calibrated model, a 0.8-confidence prediction is correct ~80% of the time.
5. Decide a **confidence threshold** for the live app: below it, show "unsure" instead of a label. Pick it from the calibration plot (e.g. the lowest bucket where accuracy ≥85%).

**Validation gate**:
- [ ] Holdout accuracy is within ~10 percentage points of cross-validation accuracy. A much larger gap means cross-val was overoptimistic (usually data leakage or an overly uniform training set).
- [ ] Per-class recall on holdout ≥70% (prototype target) or ≥85% (trust target).
- [ ] Confidence threshold is chosen and recorded here: _TBD_.
- [ ] Top failure modes are documented and converted into data-collection tasks for the next iteration.

---

## Phase 6 — iOS app integration

**Goal**: the `.mlmodel` runs on a real iPhone with live mic input, correctly handles the streaming / gating / dedup logic, and produces one labeled result per real shot.

**Steps**:
1. `xcodegen generate` from `app/project.yml` (never hand-edit `.pbxproj`).
2. Drop `models/vN_<date>.mlmodel` into the Xcode project.
3. Build the audio pipeline:
   - `AVAudioEngine` input tap, 16 kHz mono.
   - `SNAudioStreamAnalyzer` with `SNClassifySoundRequest(mlModel:)`.
   - Analyze 500ms windows with 100ms hop (5× redundancy per impact).
4. Build the gating logic:
   - Energy/onset gate: ignore windows below an RMS threshold (silence).
   - Confidence gate: only surface predictions above the Phase 5 threshold.
   - Dedup: within any 1-second window, take the single highest-confidence classification as the "shot result".
   - Reject `not_a_shot` predictions silently.
5. SwiftUI screen: big label ("PURE" / "FAT" / "TOPPED" / "…"), confidence bar, recent-shots list.

**Validation gate**:
- [ ] Build runs on a real iPhone (not just simulator — simulator mic behavior is not representative).
- [ ] Test matrix: play back 5 known clips per class through a speaker near the phone. ≥80% correctly labeled.
- [ ] False-positive test: let the app listen to 2 minutes of range ambient / voices / practice swings. Number of spurious classifications ≤1 per minute.
- [ ] Dedup test: one real (or played-back) shot produces exactly one surfaced classification, not three.
- [ ] No `.pbxproj` edits crept in. `git diff` on it should be empty aside from XcodeGen-generated changes.

---

## Phase 7 — In-the-wild validation

**Goal**: accuracy numbers from a real driving range, with real users, not lab conditions.

**Steps**:
1. Add an **in-app confirmation UI**: after each classification, the user taps "correct" or overrides with the true label. Log `{clip, predicted_label, confidence, user_label, timestamp}` locally with explicit consent. This is both your accuracy monitor and your next training set.
2. Take the app to a driving range for a dedicated session (or several). Hit ≥30 shots per class. Collect labels.
3. Compute wild-accuracy per class. Compare to Phase 5 holdout numbers.
4. Identify the biggest gap between holdout and wild accuracy — that's your next data collection target.

**Validation gate**:
- [ ] Wild per-class recall ≥70% (prototype) or ≥85% (product quality).
- [ ] Confusion matrix from the wild session matches the shape of the holdout matrix (same classes confused for each other). If a completely new failure mode appears, something about the deployment conditions wasn't represented in training — figure out what.
- [ ] User-confirmed clips are archived for the next training round.

---

## Phase 8 — Iterate

**Goal**: close the gap between what you have and what you'd trust.

**Cycle**:
1. Collect new user-confirmed clips.
2. Re-preprocess, re-augment.
3. Retrain (`models/vN+1_<date>.mlmodel`).
4. Evaluate on the **same frozen holdout set** as before (not a new one each time — you need to compare versions like-for-like).
5. Ship the new model only if holdout metrics are at least as good as the previous version, class-by-class. A model that gains overall accuracy by tanking one class is a regression, not an upgrade.

**Ongoing validation**:
- [ ] Frozen holdout metrics tracked per model version in a simple CSV: `version, date, holdout_acc, per_class_recall, wild_acc`.
- [ ] Watch for distribution drift: accuracy dips on a new phone model, cold weather, a new turf type. Each is a data gap — log it, fix it.

---

## What "done enough to ship" looks like

- ≥500 clips per class, recorded across multiple sessions / players / conditions.
- 5-fold cross-val accuracy ≥85%, per-class recall ≥80%.
- Holdout accuracy within 10 points of cross-val.
- Wild-session accuracy ≥80% per class with confidence gating.
- `not_a_shot` rejection works: ≤1 spurious classification per minute of ambient listening.
- In-app confirmation UI is live from day one of real usage — you never ship a version you can't measure.

## What to resist

- Celebrating prototype-stage accuracy numbers. Overfitting on 10 clips per class produces 100% "accuracy" and 0% real-world usefulness.
- Skipping the manual listening check in Phase 2. A broken onset detector turns the whole pipeline into noise in a way no downstream metric will make obvious.
- Adding features (swing tempo, ball speed, UI polish) before the core classifier is trustworthy.
- Editing `.pbxproj` by hand.
