# Stage 1 — Golf Shot Detector

This document focuses narrowly on Stage 1: given a continuous audio stream from an iPhone mic at a driving range, detect the moments a golf ball is struck and emit a 500ms clip centered on each impact. Classifying that clip as pure / fat / topped is Stage 2 and lives in `PROJECT.md`.

## Architecture

Two sub-stages, not one end-to-end model:

```
mic stream ──► [1a onset detector (DSP)] ──► candidate windows ──► [1b shot verifier (ML)] ──► confirmed shot events
              high recall, deterministic                             high precision, tiny model
```

**Why this split instead of one big ML model**:
- 1a is free, deterministic, and debuggable. Spectral-flux onset detection on 16 kHz audio is a ~20-line `librosa` call and will find every real impact.
- 1b gets a clean binary task (`shot` / `not_shot`) on fixed-size 500ms windows, which is the easiest regime for Create ML Sound Classifier with small data.
- If something breaks in the wild, you can tell whether you're missing shots (1a recall problem) or firing on ambient noise (1b precision problem). In an end-to-end model, you can't.

## What Stage 1 does NOT need

- **Frame-level timestamp labels.** Clip-level binary labels are sufficient. The millisecond-precision moment of impact falls out at inference as the window with peak `shot` confidence; we do not need to train a model to regress it.
- **A separate model for each shot type.** Pure, fat, and topped are all the positive class here. Stage 2 handles the discrimination.
- **Spotless positives.** An ambiguous "1mm fat" clip is still a shot. Include it in the positive class.

---

## Phase A — Data collection (for Stage 1 specifically)

Stage 1 needs two classes: `shot` and `not_shot`. The second one is where most of the work is.

### A.1 — Positive class (`shot`)

**Source**: all 9 existing samples + anything new you record. All shot types get lumped into one class.

**Targets**:
- Prototype: 100 effective positives (9 real × ~11 augmentations, or grow real count).
- Trust: 500+ effective positives.

**Recording for more positives**: every session you do for Stage 2 naturally produces Stage 1 positives, so this piggybacks on the main data collection effort.

### A.2 — Negative class (`not_a_shot`) — the critical bottleneck

You currently have **zero** of these. A shot detector trained without them will fire on every loud transient.

Target composition (aim for a similar total count to positives):
- **Range ambient** (30%): just record 5–10 minute chunks of you sitting at a driving range or golf course with no one hitting, or between shots. Distant voices, wind, bird calls, cart noise, golf bag zip, club clink.
- **Non-shot golf sounds** (30%): practice swings without hitting a ball (the "whoosh" is acoustically similar to a pre-shot), club head tapping the ground / mat, teeing up the ball, picking up the ball.
- **Human percussive sounds** (20%): claps, coughs, a dropped club, footsteps on mat, bag being set down. These are the worst false-positive candidates — sharp transients that aren't shots.
- **Environmental transients** (20%): doors closing, car doors, distant machinery, phone bumps, mic handling noise. Free-source candidates: `freesound.org` tagged "impact," "clap," "thud."

**Concretely**: one 10-minute recording at a driving range between your shots gives you dozens of negatives. Prioritize this over everything else.

### A.3 — Holdout for Stage 1

Set aside one full session (different day, different location if possible) as `data/holdout/stage1/`. Never touched during Stage 1 training. Contains both positives and negatives so we can measure both recall and precision on unseen data.

### Validation gate — Phase A

- [ ] ≥100 effective positive clips (real + augmented) in `data/processed/shot/`.
- [ ] ≥100 negative clips in `data/processed/not_a_shot/`, covering all four sub-categories above.
- [ ] Negative class includes ≥10 "hard negatives": practice-swing whooshes, claps, other sharp transients. These are what stops the model from being a glorified amplitude threshold.
- [ ] Stage 1 holdout set is locked and untouched.

---

## Phase B — Stage 1a: classical onset detection

**Goal**: a deterministic function that takes an audio stream or long recording and returns candidate impact timestamps with ~100% recall on real shots (over-triggering is fine).

### Steps

1. `scripts/detect_onset.py` using `librosa.onset.onset_detect` with `onset_envelope = librosa.onset.onset_strength(..., aggregate=np.median)`. Tune `delta` and `wait` parameters so no real shot is missed on training data.
2. For each detected onset, output a candidate window: 100ms pre, 400ms post.
3. Also emit an energy/RMS gate — windows below a threshold are silence and can be rejected immediately without even invoking 1b.

### Validation gate — Phase B

- [ ] On the 9 known positive recordings, the onset detector finds the true impact every time (100% recall). Verify by eye on spectrograms + listen.
- [ ] False-positive rate on a 10-minute range-ambient recording is recorded. **Expected**: many false positives — that's fine, 1b's job is to filter them. But log the number; this is your baseline for measuring 1b's contribution.
- [ ] The energy gate alone rejects ≥90% of obvious silence windows, reducing downstream inference cost.

---

## Phase C — Stage 1b: shot verifier training

**Goal**: a Create ML Sound Classifier with two classes (`shot`, `not_a_shot`) that takes a 500ms 16 kHz mono window and outputs probabilities.

### Steps

1. Organize training data:
   ```
   data/stage1_training/
   ├── shot/         # all positives (pure + fat + topped + augmented)
   └── not_a_shot/   # all negatives from Phase A.2
   ```
2. Train with Create ML Sound Classifier. Save as `models/stage1_vN_<date>.mlmodel`.
3. Run 5-fold cross-validation at the source-clip level (augmented variants grouped with their source — never split across folds).
4. Report per-class precision, recall, F1. For a detector, **recall on `shot` matters more than overall accuracy** — missing a shot is worse than the occasional false positive (which can be filtered at the application layer).

### Validation gate — Phase C

- [ ] 5-fold CV `shot` recall ≥90%. (Missing 1 in 10 shots is the upper limit of acceptable for a live app.)
- [ ] 5-fold CV `shot` precision ≥80%. (False positives get expensive at scale, but precision can be tightened later with a confidence threshold.)
- [ ] On the Phase A.3 holdout, run the detector end-to-end (1a → 1b):
  - Every true shot in the holdout is detected (100% recall on the shot-level task).
  - False-positive rate on ambient sections is ≤1 per minute.
- [ ] Listen to every false positive from the holdout. Is it a consistent pattern (e.g. "bag zips always trigger")? If so, that's the next negative-class collection task.
- [ ] Listen to every false negative (missed shot) from the holdout. If there's a pattern (e.g. "topped shots at 2m mic distance are missed"), it's a positive-class collection task.

---

## Phase D — End-to-end streaming test on device

**Goal**: confirm the 1a + 1b pipeline works on a real iPhone with live mic.

### Steps

1. Integrate the `.mlmodel` into a minimal Xcode project (XcodeGen, no hand-edited `.pbxproj`).
2. `AVAudioEngine` input tap at 16 kHz mono.
3. Onset detector runs on the buffer (either ported to Swift using vDSP, or the simpler path: let `SNAudioStreamAnalyzer` do sliding-window classification and treat 1b confidence spikes as detections, skipping 1a entirely on device).
4. Emit `ShotDetectedEvent { timestamp, confidence, audioBuffer[500ms] }`.
5. Debug UI: log every detection with timestamp and confidence. No UI polish yet.

### Note on 1a on device

You may find that on device, the simplest architecture is to **skip 1a entirely** and run 1b as a sliding-window classifier via `SNAudioStreamAnalyzer` (which is exactly what that API is designed for). The onset detector was mostly useful for offline batch processing of training data. If the sliding-window 1b classifier runs fast enough on-device (it will — these models are tiny), 1a is redundant in production. Keep the option open — don't over-engineer.

### Validation gate — Phase D

- [ ] Play back 10 known shot recordings through a speaker to the iPhone mic. ≥9 are detected within ±100ms of the true impact.
- [ ] Let the app listen to 5 minutes of real driving range ambient. Spurious detections ≤5 total.
- [ ] Detected clips, when dumped to disk and listened to, actually contain a golf impact centered at ~100ms into the 500ms buffer.
- [ ] CPU/battery: the continuously-running classifier uses <5% CPU on a modern iPhone. (If it doesn't, Core ML is probably not using the Neural Engine — check the model compute unit config.)

---

## What this gets you

At the end of Phase D, you have a working **golf shot detector**: a component that takes live mic input and emits clean 500ms buffers every time a ball is struck. Stage 2 (pure/fat/topped) becomes a much easier problem because it no longer has to deal with silence, ambient, or the streaming / gating / dedup logic — it just takes a confirmed shot buffer as input.

This also gives you a **data-flywheel tool**: point the detector at any recording session and it auto-crops the shots for you. Labeling Stage 2 data becomes "scroll through auto-detected shots and tag each one" instead of "manually find every impact in a 30-minute recording." That will speed up every subsequent data collection.

## Open questions to resolve before starting

1. **Do we use 1a at all on device, or only offline for preprocessing?** Decide after Phase D testing.
2. **Negative class augmentation strategy**: do we augment negatives with the same pitch/time/noise operations as positives, or is real diversity enough? Suggest: minimal augmentation of negatives (just gain + time shift) so the model doesn't learn augmentation artifacts as "shot-like."
3. **Confidence threshold for emitting an event**: default to 0.7 but tune from the Phase C calibration plot.
