# External Audio Dataset

Audio sourced for the golf shot detector app following `SOURCING_GUIDE.md`.

## Counts

| Split | Count |
| --- | ---: |
| Trainable negatives | 1300 |
| Trainable positives | 28 |
| Research-only | 0 |
| Reference-only | 0 |

## Folders

| Item | Count |
| --- | ---: |
| `data/external/negatives/cc0/club_bag_equipment` | 123 |
| `data/external/negatives/cc0/human_percussive` | 156 |
| `data/external/negatives/cc0/non_golf_impacts` | 162 |
| `data/external/negatives/cc0/outdoor_ambient` | 148 |
| `data/external/negatives/cc0/phone_handling` | 48 |
| `data/external/negatives/cc0/whoosh_swing` | 43 |
| `data/external/negatives/cc_by/club_bag_equipment` | 77 |
| `data/external/negatives/cc_by/human_percussive` | 144 |
| `data/external/negatives/cc_by/non_golf_impacts` | 138 |
| `data/external/negatives/cc_by/outdoor_ambient` | 152 |
| `data/external/negatives/cc_by/phone_handling` | 52 |
| `data/external/negatives/cc_by/whoosh_swing` | 57 |
| `data/external/positives/cc0/golf_generic_impact` | 28 |

## Categories

| Item | Count |
| --- | ---: |
| `club_bag_equipment` | 200 |
| `golf_generic_impact` | 28 |
| `human_percussive` | 300 |
| `non_golf_impacts` | 300 |
| `outdoor_ambient` | 300 |
| `phone_handling` | 100 |
| `whoosh_swing` | 100 |

## Sources

| Item | Count |
| --- | ---: |
| `FSD50K` | 1300 |
| `self_recorded` | 28 |

## Licenses

| Item | Count |
| --- | ---: |
| `Creative Commons 0` | 680 |
| `Creative Commons Attribution 3.0` | 620 |
| `owned_by_project` | 28 |

## Files

- `manifest.csv`: one provenance row per audio file.
- `manifest.jsonl`: JSONL copy of the same manifest.
- `SOURCES.md`: source and license notes.
- `negatives/`: trainable non-shot audio organized by license and category.
- `positives/`: trainable golf shot audio organized by license and category.
- `reference_only/`: reserved for sources that are not cleared for training.
