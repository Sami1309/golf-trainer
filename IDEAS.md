# IDEAS — sequencing, risks, gotchas, alternatives

A companion to `PLAN_STAGE1B_STAGE2.md`. The plan there is sound; this file is the commentary — **how to get the classifier trained as fast as possible**, and the specific ways this project can go sideways if we don't watch for them.

Current recommendation after inspecting the repo: do not start with browser ONNX integration or backend work. Build a reproducible offline training loop first, get real confusion matrices for Stage 1b and Stage 2, then decide what is worth exporting. The existing app is Phase 0 only; there are no `scripts/`, no extracted clips, no embedding cache, no Python env, and no model artifacts yet.

Actual dataset count in `data/labels.json`: 28 positives total = 11 pure, 11 fat, 2 `fat_borderline`, 4 topped. If the two borderlines are folded into fat, Stage 2 has 11 pure / 13 fat / 4 topped.

---

## Recommended sequencing — fastest path to a working Stage 2

The plan as written proposes doing **Stage 1b first, then Stage 2**. Good for shipping order, bad for iteration speed. Training them together is strictly faster because they share the expensive part (YAMNet feature extraction) and the dataset preparation.

### Proposed fast-path

Rough calendar — each "step" is a few hours of focused work, not a full day.

**Step 1 — data extraction (one Python script, one sitting)**
`scripts/extract_clips.py` reads `data/labels.json`, decodes each `.m4a`, resamples to 16 kHz mono, re-centers on peak |amplitude| within ±60 ms of the labeled time, normalizes to -3 dBFS, writes `data/clips/<class>/<sourcefile>.wav` + a `clip_manifest.csv` row per clip. One run produces all 28 labeled positives. Use relative path or a stable hash as `source_id`; `data/labels.json` is currently keyed by basename, which will collide as soon as two future recordings share a filename.

Same script (or a sibling `extract_negatives.py`) slides a 500 ms window across every recording excluding the shot ± 600 ms, rejects sub −35 dBFS windows, writes `data/clips/not_a_shot/*.wav`. The current recordings total ~216 seconds; after excluding shot windows, expect roughly 180 seconds of edge audio and a few hundred overlapping seed negatives.

**Validation**: eyeball & listen to 5 clips per class. Run immediately.

**Step 2 — cache YAMNet embeddings (one-time cost)**
`scripts/embed_clips.py`: load YAMNet once, compute a single fixed-length embedding per clip, save `data/embeddings.npz` keyed by source filename. Keep the script idempotent and hash-aware so re-running on unchanged clips is a no-op. From here on, training is purely numerical — seconds per experiment, no audio decoding.

**Step 3 — train both heads together**
`scripts/train_heads.py`: loads `embeddings.npz` + `clip_manifest.csv`, runs 5-fold CV grouped by source clip (so augmented variants stay together), trains Stage 1b (binary) and Stage 2 (3-class, plus fat_borderline as a 4th option) in the same run. Reports CV metrics + confusion matrix for both heads.

Expected outcome on 28 positives + ~400 negatives with no augmentation:
- Stage 1b CV recall ≥0.95 (shot vs edge-ambient is an easy binary task).
- Stage 2 CV recall ≥0.60 on pure and fat; topped will likely be ≤0.50 with 4 samples — expected.

**Step 4 — decide Stage 2 scope before augmenting**
If Stage 2 is saved only by augmentation, it isn't really saved. Look at the confusion matrix:
- If pure/fat are separable but topped collapses — **ship pure-vs-fat v0** and delay topped until we have ≥15 topped recordings. Document this choice.
- If all three classes confuse heavily — something structural is wrong (label noise, feature mismatch, the 1mm-fat labels are polluting fat). Fix before spending compute on augmentation.

**Step 5 — augmentation, retrain**
`scripts/augment.py` writes 10 variants per source to `data/clips_aug/`. Re-run `embed_clips.py` (incremental), re-run `train_heads.py`. Compare to the unaugmented baseline per-class. If augmentation helps training-accuracy but not validation-accuracy, that's overfitting — don't celebrate training numbers.

**Step 6 — export both heads to ONNX**
Two small ONNX head files plus the shared embedding model. Do this only after offline metrics justify it; converting/deploying YAMNet is engineering work, not proof the classifier is viable. Verify numerical parity on 20 clips, check size and iPhone Safari inference latency.

**Step 7 — wire into the web app**
One new JS module: `loadAndRun(yamnetSession, stage1bSession, stage2Session, samples) → {verified, label, confidence}`. Hook it into `onLiveDetected` in `app.js`. Add a "show rejected" debug toggle.

**Step 8 — HTTPS deploy + iPhone test**
Deploy `frontend/` to Vercel / Cloudflare Pages (static, free, auto-HTTPS). Test on iPhone at the range. First real signal of whether this approach works in the wild.

**Step 9 — in-app feedback UI**
Thumbs-up / override buttons per detection. Persist locally, export JSON. This unlocks the data flywheel even before the backend in Phase 2.3.

**Step 10 — backend only when needed**
Don't build the backend until the web app has produced enough corrected data locally that syncing is worth the complexity. One practice session might produce 30–60 labeled clips — that can live in `localStorage` + manual export for a while.

### Why this order vs the plan's order

- **Parallelize Stage 1b and Stage 2 training.** The plan shows them as two Phases; from the code side they are one script with two loss heads. The embedding cache makes retraining near-free.
- **Decide topped scope before burning effort on augmentation.** If augmentation is what's keeping topped afloat, the model is pretending. Honesty first.
- **Defer the backend.** `localStorage` + export is plenty for the first few weeks of real-world use. The backend is an operational commitment that only pays off once we've proven the model is worth training on more data.
- **HTTPS + iPhone Safari test is the biggest unknown after "does the model work."** Surface it early so a ORT-web / Safari / WebGL problem doesn't blindside us at the end.

---

## Risks, gotchas, concerns

### A. Dataset-level risks

**A1 — Topped has 4 samples. That is not a class, it's a rumor.**
5-fold CV at source-clip level puts 0–1 topped samples in each validation fold. Any accuracy number you read off that is noise. Three honest options:
- Drop topped for v0 (pure-vs-fat).
- Leave-one-out CV specifically for topped to at least get a failure mode inventory.
- Block the Stage 2 work until ≥15 topped recordings exist (cheap — one range session with deliberate tops).

Recommendation: **ship pure-vs-fat v0, collect topped in parallel, add it in v1**. This is faster than waiting, and a pure/fat-only classifier is still useful.

**A2 — 1mm-fat handling isn't decided.**
Two labeled borderline clips ("7 - 1mm in front", "20 - great example of 1mm fat"). Folding into `fat` adds noise to the most confident-sounding fat examples; keeping as its own class trains a 4-way model on 2 samples (not a class either); dropping wastes two hand-labeled recordings.

Recommendation for v0: **fold into fat**, mark them `confidence=low` in the manifest, watch the confusion matrix. If they turn into the dominant source of pure↔fat confusion, revisit.

**A3 — Single recording location.**
Every clip is Rancho Park / Fareways, same mic setup (iPhone on ground, ≤1 m). A model that hits 95% on this dataset is a 95% Rancho Park classifier. Before quoting any accuracy number to a user, collect a holdout from a second course — even 5 clips per class is enough to calibrate how optimistic the in-sample numbers are.

**A4 — Same-session near-duplicates inflate CV.**
The 11 pure samples may have been recorded in one or two sessions. Clips from the same session share mic noise floor, distance, ambient — a standard k-fold at the clip level will split those, and the model will "learn the session" and look better than it is. Group by session (or by recording day) when splitting, not just by clip.

Action: add a `session_id` column to `clip_manifest.csv` (derive from date of `labeledAt` or manually tag if we know the sessions). Use GroupKFold over session ID for CV.

**A5 — Negatives cut from edges are nearly all "quiet ambient".**
The seed non-shot audio from existing recordings is only about 3 minutes before overlap-windowing, and it is dominated by near-silence + occasional voice. That's a weak negative class: Stage 1b can learn "anything loud = shot" and pass our validation trivially.

What's missing: **hard negatives** — practice-swing whooshes, claps, bag drops, footsteps, club clinks. Without at least 20–50 of these, Stage 1b's precision in the wild will collapse at the first clap.

Cheapest source: record 2 minutes of deliberate hard-negative sounds (clap your hands, drop a glove, tap a club on the mat, cough, bump the phone). That alone doubles the diversity of the negative class.

**A6 — The labeled "peak" isn't the amplitude peak.**
Labels are where the user clicked during playback; the amplitude peak is 28 ms later (click reaction time). Phase 0 re-centers detections on the amplitude peak, which is correct. But the **label time itself is not the training target** — clips should be centered on the amplitude peak within ±60 ms of the label, not on the label. This is subtle but matters: train the model to see the sound as it actually fires in the detector.

### B. Model-level risks

**B1 — YAMNet expects 0.96 s windows, we have 0.50 s.**
YAMNet's native window is ~0.96 s of 16 kHz audio. Feeding 500 ms requires a choice:
- Pad with silence to 0.96 s (cheap, may reduce embedding quality).
- Take the **patch-level** embeddings YAMNet computes internally (it runs on 0.96 s chunks but exposes 0.48 s hop patches — each 500 ms clip yields ~1 patch embedding).
- Repeat the 500 ms clip to fill 0.96 s (introduces periodicity, bad).

Recommendation: **silence-pad to 0.96 s and center the clip**. Document the padding strategy in `clip_manifest.csv` so downstream inference matches exactly.

**B2 — YAMNet ONNX on iPhone Safari is an unknown.**
~5 MB download on first page load over mobile is painful; WebGL acceleration on iOS Safari has historically been patchy; ORT-web's WASM backend is slower than WebGL. If YAMNet inference takes 200 ms per detection on iPhone, the UX is fine (we only run it once per shot) but loading time to first classification is bad.

Mitigations to have ready:
- Serve YAMNet gzipped + `Cache-Control: immutable`; first visit is painful, subsequent are free.
- Show a "loading model…" state in the UI explicitly.
- Consider INT8 quantization of YAMNet (smaller, potentially faster on WASM).

**B3 — Alternative to YAMNet: tiny log-mel CNN.**
YAMNet is trained on 2M general-purpose audio clips. For a single-class-of-sounds problem (golf impact vs. not), a small 1D CNN over log-mel features (say 64 mel bands × 32 frames × 3 conv blocks) trained from scratch could be **smaller (<500 KB), faster on mobile, and no worse on this specific data**. Worth spending half a day to train both and compare, because if the log-mel CNN is competitive we avoid the YAMNet ONNX deployment headache entirely.

Both approaches share the same clip-extraction pipeline, so branching is cheap.

**B4 — Overfitting on 28 clips is all-but-guaranteed.**
With a 1024→128→3 head and 28 examples (even with 10x augmentation → 280), regularization dominates. Expect:
- Training accuracy >95%, validation accuracy 55-70%.
- Heavy dropout (0.5), weight decay (1e-3), early stopping on validation F1.
- Sub-1-epoch learning curves — don't train for 50 epochs, train for 10 with aggressive early stopping.

The real fix is more data, not more regularization. Augmentation is a stopgap for data, not a substitute.

**B5 — Pitch shift ±2 semitones may distort the acoustic signature.**
Pure vs fat vs topped differ partly in frequency content (pure has the most high-frequency "crack"). Pitch shifting by ±2 semitones rescales that by ~12%, which could push a pure toward a topped in the training set. This is an augmentation that models for mic variability but might confuse the classifier on what is largely a frequency-content task.

Recommendation: **start with narrow pitch augmentation (±0.5 semitones or skip entirely), add time-shift and gain + additive noise.** Revisit pitch augmentation once you have a baseline.

**B6 — Time-shift ±50 ms is under-specified.**
The detector's median offset is 27 ms; max is 68 ms. Stage 1b sees clips from the detector in production, so training with ±50 ms time shift might miss clips that land ±68 ms. Use **±75 ms time-shift augmentation** for Stage 1b so training coverage exceeds observed detector variance.

For Stage 2, the input is already Stage-1b-verified — the cropping is more consistent, and ±50 ms is plenty.

**B7 — The class imbalance interacts with augmentation.**
If you fold borderline into fat and 10x-augment all clips equally, the current 11:13:4 ratio becomes 110:130:40. A class-weighted loss compensates, but with this few topped examples, weighting pushes the model to over-predict topped on any ambiguous sample. Prefer to **oversample topped during augmentation** (say 30x topped, 10x others) to balance without weight-hacking the loss.

### C. Pipeline / deployment risks

**C1 — Audio-spec drift is silent and catastrophic.**
Two common ways this breaks:
- Training at 16 kHz, inference at Safari's native 48 kHz with resampling done differently → spectra don't match → accuracy plummets and there's no error.
- Training on -3 dBFS normalized clips, inference on whatever amplitude the mic gave → features shift.

The current Phase 0 app resamples and re-centers clips, but it does **not** peak-normalize detections before adding them to the UI. That is acceptable for labeling, but model inference must normalize.

Fix: add a shared normalization helper used by both `scripts/extract_clips.py` and the browser model path. Add a runtime assertion in the web app that every clip fed to YAMNet is 8000 samples, peak-normalized to a known level. Fail loud in the console if not.

**C2 — ONNX export parity checks are non-negotiable.**
Silent accuracy loss between PyTorch training and ONNX export is a recurring bug: different operator implementations, quantization, shape-inference edge cases. The plan already calls for numerical parity on 20 test clips — **do not skip this**. A 1e-2 divergence is fine; 1e-1 is suspicious; class flips on any clip is a hard stop.

**C3 — iPhone Safari AudioContext runs at 48 kHz.**
The existing app already handles this — it uses the native context rate for live detection and resamples only when writing clips. Keep that pattern. Any new code that assumes 16 kHz in the live path is a bug.

**C4 — CV fold leakage via augmentation.**
Augmented variants of clip X in train + clip X's original in validation = 100% "accuracy" that doesn't survive deployment. The plan says to group augmented variants with their source — **write this as a test** in `train_heads.py` that asserts no source_id appears in both train and val splits of any fold. Mechanical, cheap, catches a whole category of bugs.

**C5 — `localStorage` for labels is fragile.**
The labeling UI currently auto-saves to `localStorage`. Clearing site data / private browsing / storage quota → all labels lost. The plan has an export/import round-trip, and the canonical labels are already exported to `data/labels.json` — so this is covered for Phase 0, but keep the discipline: after any labeling session, immediately export.

**C6 — Practice-swing whooshes are the likely deadly false positive.**
A practice swing produces a swoosh of air moving past the club head. Acoustically it has a lot of broadband content — exactly what spectral flux fires on. If we don't have these as negatives, Stage 1b will confidently classify every practice swing as a shot in production. Collect ≥10 practice-swing clips specifically and label them as `not_a_shot`.

**C7 — Label identity must move from filename to source path.**
The Phase 0 label store uses filenames as keys because the current 28 files have unique basenames. That will break once new sessions create another `Rancho Park Golf Course 1.m4a` or a phone reuses a filename. The training manifest should use a stable `source_id` derived from normalized relative path plus file hash, while preserving the original filename for display.

### D. Process / scope risks

**D1 — The backend is a trap door.**
`PLAN_STAGE1B_STAGE2.md` §2.3 describes a FastAPI + SQLite backend on Fly.io. That's three deployables (frontend, backend, model artifacts), one ops surface, and one more thing to debug on the day of the range test. Defer it.

For the first N practice sessions, the in-browser export button + a `git add data/sessions/<date>/*.wav` workflow is enough. Only build the backend when manual curation is the bottleneck — not before.

**D2 — Over-indexing on "one giant retraining run."**
With 28 samples, the difference between a great and a mediocre model is the data, not the head architecture. Don't spend two days tuning the head when you could spend two hours recording 40 more shots. Every training experiment you run on this dataset size is overfitting to noise in the training-val split.

Rule of thumb: if you've retrained the head more than 5 times without new data, stop. Collect data instead.

**D3 — The "demo a live session" trap.**
It's tempting to first take the web app to the range and hit a bunch of shots to "test it." That's valuable, but only *after* the model has been validated offline on held-out data. Going to the range with an untested model means you get two new debugging layers at once (mic quirks + model bugs) and can't tell them apart. Verify offline first.

**D4 — Docs drift.**
`STAGE1.md` and `PROJECT.md` partly contradict `PLAN_STAGE1B_STAGE2.md` now (STAGE1 assumes Create ML; PLAN assumes YAMNet + ONNX). CLAUDE.md has been updated to reflect current direction; keep the others as historical artifacts and add headers noting what's superseded so we don't act on stale plans by accident.

### E. Nice-to-haves / open questions

**E1 — Could we train Stage 2 to also predict *where* on the face the ball was hit?**
Heel vs toe, high vs low. Acoustically probable: each impact location has a different mass/stiffness signature. Would require new labels (harder — the user can hear it's fat, can they reliably tell heel vs center?). **Out of scope for v0** but worth flagging as a potential "ball flight" proxy if the basic classifier works.

**E2 — Confidence calibration via temperature scaling.**
Small datasets produce overconfident models (almost always). Post-training temperature scaling (one scalar, fit on validation) typically fixes this and makes the "below threshold → unsure" gating honest. One extra line in the training script; big UX win.

**E3 — Progressive disclosure UI.**
Big label in the center ("PURE") but also show the per-class probability bars, and grey out the label when confidence is below threshold. Users trust a calibrated "unsure" more than a wrong confident label.

**E4 — Ship the v0 to a friend.**
The cheapest real-world test is one other person trying it on their phone. Catches: does the label parsing work in a new `localStorage`, does the ORT-web model download over their mobile connection, does the UI survive a rotate-to-landscape, does the mic permission flow on their iOS version actually work.

**E5 — The MOV files.**
Every shot folder has both an `.m4a` and an `.MOV`. The `.MOV` is a synced video — potentially useful for later ground-truthing (did the ball actually top vs go fat?) or for expanding to a visual-confirmation UI. Don't use for audio training (the audio track is the same as the `.m4a`), but don't throw away either.

---

## TL;DR

1. **Train Stage 1b and Stage 2 in one Python run over a shared embedding cache.** Don't serialize them.
2. **Collect hard negatives (practice swings, claps) before training Stage 1b.** Edge-silence alone will trivially pass validation and fail in production.
3. **Decide topped's fate early.** Ship pure-vs-fat v0 while collecting more topped recordings; don't let 4 samples gate the entire Stage 2.
4. **Use YAMNet but have a tiny log-mel CNN as a Plan B.** Two ONNX exports, compare, pick the winner on honest metrics.
5. **Deploy to iPhone over HTTPS as step 7, not step 20.** The ORT-web × Safari interaction is the biggest unknown after model quality.
6. **Defer the backend.** `localStorage` + export → commit → retrain is enough until it isn't.
7. **Group CV by source clip *and* by recording session.** Otherwise the model memorizes the session, not the shot.
