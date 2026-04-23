# External Audio Sources

## FSD50K

- Source: https://zenodo.org/records/4060432
- Mirror used for file download: https://huggingface.co/datasets/Fhrozen/FSD50k
- Audio origin: Freesound clips with per-clip Creative Commons licenses.
- Included here: only CC0 and CC-BY clips selected from guide-relevant negative classes.
- AI training permission: marked `yes` for CC0/CC-BY based on Freesound's published AI training guidance supplied in the project reference.
- Commercial permission: marked `yes` for the individual CC0/CC-BY clips; note that FSD50K's dataset page asks commercial users to contact the dataset authors.
- Attribution: required for CC-BY rows; attribution text is included per row in the manifests.

## Self-Recorded Local Golf Audio

- Source: local `.m4a` files already present in this workspace on 2026-04-23.
- Location: copied into `data/external/positives/cc0/golf_generic_impact/`.
- License: `owned_by_project`.
- AI training permission: `yes`.
- Commercial permission: `yes`.
- Notes: source folder names such as `pure`, `fat`, and `topped` are preserved in manifest notes but not promoted into normalized ML labels.

## Not Used

- Freesound direct API: skipped because no Freesound API/OAuth token was available in the environment.
- Sonniss, Pixabay, YouTube, BBC RemArc, ESC-50, UrbanSound8K, ZapSplat, stock marketplaces: not used for trainable data because the guide marks them prohibited, non-commercial, or unclear for ML/commercial use.
- SoundBible: metadata was inspected, but FSD50K provided stronger labeled coverage for this first batch.
