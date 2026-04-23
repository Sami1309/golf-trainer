# Audio Data Sourcing Guide

This guide is for the agent that will download and organize external audio for the golf shot detector project.

Goal: collect legally usable audio into `data/` with enough metadata that training scripts can safely include or exclude files by source, license, and class.

## Critical Rules

- Do **not** put downloaded audio in the repo root. Everything goes under `data/`.
- Do **not** overwrite `data/labels.json`; that is the existing hand-labeled golf shot manifest.
- Do **not** use sources whose terms prohibit ML/AI training.
- Do **not** use YouTube audio unless the specific video is Creative Commons and the download/use is permitted. In practice, treat YouTube as reference-only unless explicitly approved.
- Prefer `CC0` first, then `CC-BY`. Avoid `CC-BY-NC` unless files are marked `research_only`.
- Keep every source URL, license, attribution string, and download date in a manifest.
- If license terms are ambiguous, download into `data/external/reference_only/`, not into trainable folders.

## Desired Folder Layout

Create this structure:

```text
data/
├── external/
│   ├── README.md
│   ├── SOURCES.md
│   ├── manifest.csv
│   ├── manifest.jsonl
│   ├── negatives/
│   │   ├── cc0/
│   │   │   ├── whoosh_swing/
│   │   │   ├── club_bag_equipment/
│   │   │   ├── human_percussive/
│   │   │   ├── outdoor_ambient/
│   │   │   ├── non_golf_impacts/
│   │   │   └── phone_handling/
│   │   ├── cc_by/
│   │   │   └── <same category folders>
│   │   └── research_only/
│   │       └── <same category folders>
│   ├── positives/
│   │   ├── cc0/
│   │   │   ├── golf_driver/
│   │   │   ├── golf_iron/
│   │   │   ├── golf_wedge/
│   │   │   ├── golf_putt/
│   │   │   ├── golf_generic_impact/
│   │   │   └── golf_ambient_range/
│   │   ├── cc_by/
│   │   │   └── <same category folders>
│   │   └── research_only/
│   │       └── <same category folders>
│   └── reference_only/
│       ├── README.md
│       ├── negatives/
│       └── positives/
```

Also create a short table of contents in `data/external/README.md` listing the folders, counts, and source datasets.

## Manifest Requirements

Every downloaded audio file must have one row in both:

- `data/external/manifest.csv`
- `data/external/manifest.jsonl`

Required columns / fields:

```text
local_path
split_candidate
polarity
category
source_name
source_url
download_url
original_filename
license
license_url
ai_training_permission
commercial_permission
attribution_required
attribution_text
creator
duration_sec
sample_rate_hz
channels
file_format
downloaded_at
notes
```

Allowed values:

```text
polarity: negative | positive | reference_only
split_candidate: trainable | research_only | reference_only
ai_training_permission: yes | no | unclear
commercial_permission: yes | no | unclear
attribution_required: yes | no | unclear
```

If any of `ai_training_permission` or `commercial_permission` is `no` or `unclear`, set `split_candidate` to `reference_only` or `research_only`, not `trainable`.

## Audio Format

Preserve the original file if practical. Do not destructively normalize or trim during sourcing.

Preferred:

- WAV, FLAC, AIFF, or original-quality source file.
- 44.1 kHz or 48 kHz is fine.
- Mono or stereo is fine.
- Duration target: 0.2 to 10 seconds for discrete sound effects; longer ambience is acceptable.

Training scripts will later resample, crop, and normalize.

If a source only provides previews and originals require OAuth or login, record that in `notes`.

## Negative Dataset Targets

Collect `not_a_shot` negatives that may confuse onset detection or the shot verifier.

Priority categories:

- `whoosh_swing`: whoosh, swoosh, air swing, bat swing, club swing-like sounds.
- `club_bag_equipment`: metal clicks, club taps, zippers, keys, balls rattling, bag movement.
- `human_percussive`: claps, snaps, coughs, footsteps, hand slaps.
- `outdoor_ambient`: wind, birds, voices, cars, golf carts, range ambience.
- `non_golf_impacts`: wood hits, plastic hits, metal hits, knocks, thuds.
- `phone_handling`: mic bumps, pocket rustle, fabric rustle, phone handling.

Initial target counts:

```text
whoosh_swing: 100+
club_bag_equipment: 200+
human_percussive: 300+
outdoor_ambient: 300+
non_golf_impacts: 300+
phone_handling: 100+
```

Minimum useful first batch: 1,000 trainable negative clips.

## Positive Golf Dataset Targets

Collect real golf impact sounds where licensing permits training.

Priority categories:

- `golf_driver`
- `golf_iron`
- `golf_wedge`
- `golf_putt`
- `golf_generic_impact`
- `golf_ambient_range`

For positives, quality beats quantity. Avoid synthetic/gamey sounds unless they are clearly marked as reference-only.

Initial target counts:

```text
golf_generic_impact: 100+
golf_driver: 50+
golf_iron: 50+
golf_wedge: 25+
golf_putt: 25+
golf_ambient_range: 20 longer clips
```

If labels like pure/fat/topped are not explicit, do **not** invent them. Put class notes in `notes`; the project owner will audition and label later.

## Recommended Sources

### Freesound

Use first.

- URL: `https://freesound.org`
- API docs: `https://freesound.org/docs/api/`
- Best trainable licenses: `Creative Commons 0`, then `Attribution`
- Avoid for commercial training unless license/AI permission is clear.

Recommended negative searches:

```text
whoosh
swoosh
swing
air swing
bat swing
metal click
metal tap
zipper
keys jingle
ball rattle
bag
clap
finger snap
cough
footsteps grass
footsteps gravel
wind field
birds morning
crowd ambience
golf cart
car pass-by
wood hit
plastic hit
metal hit
knock
thud
mic handling
pocket rustle
fabric rustle
mic bump
phone handling
```

Recommended positive searches:

```text
golf
golf impact
golf hit
golf ball
golf club
golf swing
golf driver
iron golf
golf wedge
golf putt
tee shot
driving range
```

Suggested Freesound filters:

```text
duration: 0.2 to 10 sec for event clips
license: Creative Commons 0 OR Attribution
sort: rating_desc or downloads_desc for first pass
```

### FSD50K

Use for negative dataset if manageable.

- URL: `https://zenodo.org/records/4060432`
- Useful for labeled non-shot classes.
- Aggregate is CC-BY, but individual clips have per-file licenses. Preserve per-clip license metadata.

Useful negative classes:

```text
Wind
Wind noise
Bird vocalization
Applause
Clapping
Finger snapping
Cough
Walk/footsteps
Knock
Thump/thud
Tap
Crack
Crackle
Car
Motor vehicle
Engine
Zipper
Keys jangling
Slap, smack
Wood
Drawer open or close
```

Only put files into `trainable` if their individual license and permissions are acceptable.

### Self-Recorded Audio

If the downloader agent receives user-provided recordings, place them under:

```text
data/external/negatives/cc0/self_recorded_<date>/
data/external/positives/cc0/self_recorded_<date>/
```

Use `source_name=self_recorded` and `license=owned_by_project`.

Self-recorded target-phone audio is especially valuable.

## Sources To Avoid For Training

Do not place these in trainable folders unless there is explicit written permission:

- Sonniss GDC bundles: current license prohibits AI/ML training.
- Pixabay sound effects: terms prohibit ML training / scraping.
- BBC Sound Effects RemArc: research/personal/educational only; not commercial-safe.
- ESC-50: CC-BY-NC, research-only.
- UrbanSound8K: CC-BY-NC, research-only.
- YouTube Audio Library: scoped to YouTube video use, not ML training.
- YouTube golf videos: copyrighted and TOS-restricted unless explicitly Creative Commons and approved.
- AudioSet raw YouTube audio: labels are usable, audio download/legal status is not clean.
- Pond5, Soundsnap, Storyblocks, ZapSplat, Epidemic Sound: standard media licenses do not clearly grant ML training rights.

If collected for listening/reference, place under `data/external/reference_only/` and mark `split_candidate=reference_only`.

## Naming Convention

Use stable, readable filenames:

```text
<source>__<source_id>__<short_slug>.<ext>
```

Examples:

```text
freesound__123456__metal-click-short.wav
fsd50k__987654__footsteps-gravel.wav
self__20260423_001__practice-swing.wav
```

Do not rely on filenames as labels. The manifest is the source of truth.

## Quality Control Checklist

After each source batch:

- Confirm files play locally.
- Confirm manifest row count equals audio file count.
- Confirm every local path exists.
- Confirm no `reference_only` files are inside trainable folders.
- Remove duplicates where obvious.
- Spot-listen to at least 20 files per category.
- Put questionable files in `reference_only` rather than guessing.

## Expected Deliverables

At the end, provide:

- `data/external/README.md` with table of contents and counts.
- `data/external/SOURCES.md` with source/license summaries.
- `data/external/manifest.csv`.
- `data/external/manifest.jsonl`.
- Organized audio folders under `data/external/negatives/`, `data/external/positives/`, and `data/external/reference_only/`.
- A short final note listing:
  - trainable negative count
  - trainable positive count
  - research-only count
  - reference-only count
  - any license concerns
  - recommended files/categories for the project owner to audition first
