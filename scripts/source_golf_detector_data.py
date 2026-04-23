#!/usr/bin/env python3
"""Source external audio for the golf shot detector dataset.

This script follows SOURCING_GUIDE.md:
- keeps external audio under data/external
- preserves originals
- records source/license/provenance in CSV and JSONL manifests
- uses only CC0/CC-BY FSD50K clips for public trainable negatives
- treats local shot recordings as self-recorded trainable positives
"""

from __future__ import annotations

import csv
import json
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
EXTERNAL = DATA / "external"
TMP = EXTERNAL / "_tmp"
FSD_TMP = TMP / "fsd50k"

TODAY = "2026-04-23"
DOWNLOADED_AT = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

MANIFEST_COLUMNS = [
    "local_path",
    "split_candidate",
    "polarity",
    "category",
    "source_name",
    "source_url",
    "download_url",
    "original_filename",
    "license",
    "license_url",
    "ai_training_permission",
    "commercial_permission",
    "attribution_required",
    "attribution_text",
    "creator",
    "duration_sec",
    "sample_rate_hz",
    "channels",
    "file_format",
    "downloaded_at",
    "notes",
]

NEGATIVE_CATEGORIES = [
    "whoosh_swing",
    "club_bag_equipment",
    "human_percussive",
    "outdoor_ambient",
    "non_golf_impacts",
    "phone_handling",
]

POSITIVE_CATEGORIES = [
    "golf_driver",
    "golf_iron",
    "golf_wedge",
    "golf_putt",
    "golf_generic_impact",
    "golf_ambient_range",
]

FSD_TARGETS = {
    "whoosh_swing": 100,
    "club_bag_equipment": 200,
    "human_percussive": 300,
    "outdoor_ambient": 300,
    "non_golf_impacts": 300,
    "phone_handling": 100,
}

FSD_LABEL_MAP = {
    "whoosh_swing": ["Whoosh_and_swoosh_and_swish"],
    "club_bag_equipment": [
        "Zipper_(clothing)",
        "Keys_jangling",
        "Chink_and_clink",
        "Rattle",
        "Tap",
        "Tick",
        "Bell",
        "Bicycle_bell",
    ],
    "human_percussive": [
        "Applause",
        "Clapping",
        "Finger_snapping",
        "Cough",
        "Walk_and_footsteps",
    ],
    "outdoor_ambient": [
        "Wind",
        "Bird",
        "Bird_vocalization_and_bird_call_and_bird_song",
        "Car",
        "Car_passing_by",
        "Motor_vehicle_(road)",
        "Engine",
        "Traffic_noise_and_roadway_noise",
    ],
    "non_golf_impacts": [
        "Knock",
        "Thump_and_thud",
        "Crack",
        "Crackle",
        "Wood",
        "Drawer_open_or_close",
        "Slam",
    ],
    "phone_handling": [
        "Telephone",
        "Camera",
        "Typing",
        "Typewriter",
        "Mechanisms",
    ],
}

FSD_FILES = {
    "dev.csv": "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/labels/dev.csv",
    "eval.csv": "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/labels/eval.csv",
    "dev_clips_info_FSD50K.json": "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/metadata/dev_clips_info_FSD50K.json",
    "eval_clips_info_FSD50K.json": "https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/metadata/eval_clips_info_FSD50K.json",
}

LICENSES = {
    "http://creativecommons.org/publicdomain/zero/1.0/": {
        "bucket": "cc0",
        "name": "Creative Commons 0",
        "attribution_required": "no",
        "commercial_permission": "yes",
        "ai_training_permission": "yes",
    },
    "http://creativecommons.org/licenses/by/3.0/": {
        "bucket": "cc_by",
        "name": "Creative Commons Attribution 3.0",
        "attribution_required": "yes",
        "commercial_permission": "yes",
        "ai_training_permission": "yes",
    },
    "http://creativecommons.org/licenses/by/4.0/": {
        "bucket": "cc_by",
        "name": "Creative Commons Attribution 4.0",
        "attribution_required": "yes",
        "commercial_permission": "yes",
        "ai_training_permission": "yes",
    },
}


def log(message: str) -> None:
    print(message, flush=True)


def slugify(value: str, max_len: int = 48) -> str:
    value = value.lower()
    value = re.sub(r"\.[a-z0-9]{2,5}$", "", value)
    value = re.sub(r"[^a-z0-9]+", "-", value).strip("-")
    return (value[:max_len].strip("-") or "audio")


def download(url: str, dest: Path, retries: int = 4) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
                shutil.copyfileobj(response, out)
            tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if tmp.exists():
                tmp.unlink()
            if attempt == retries:
                raise RuntimeError(f"failed to download {url}: {exc}") from exc
            time.sleep(1.5 * attempt)


def ffprobe(path: Path) -> dict[str, str]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=sample_rate,channels,codec_type",
        "-of",
        "json",
        str(path),
    ]
    try:
        data = json.loads(subprocess.check_output(cmd, text=True))
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return {
            "duration_sec": "",
            "sample_rate_hz": "",
            "channels": "",
        }
    audio_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "audio"),
        {},
    )
    duration = data.get("format", {}).get("duration")
    return {
        "duration_sec": f"{float(duration):.3f}" if duration else "",
        "sample_rate_hz": str(audio_stream.get("sample_rate", "")),
        "channels": str(audio_stream.get("channels", "")),
    }


def ensure_layout() -> None:
    for bucket in ["cc0", "cc_by", "research_only"]:
        for category in NEGATIVE_CATEGORIES:
            (EXTERNAL / "negatives" / bucket / category).mkdir(parents=True, exist_ok=True)
        for category in POSITIVE_CATEGORIES:
            (EXTERNAL / "positives" / bucket / category).mkdir(parents=True, exist_ok=True)
    for polarity in ["negatives", "positives"]:
        (EXTERNAL / "reference_only" / polarity).mkdir(parents=True, exist_ok=True)
    (EXTERNAL / "reference_only" / "README.md").write_text(
        "# Reference-Only Audio\n\n"
        "Files placed here are not eligible for training unless their license and "
        "AI/commercial permissions are later cleared.\n",
        encoding="utf-8",
    )


def fetch_fsd_metadata() -> None:
    FSD_TMP.mkdir(parents=True, exist_ok=True)
    for filename, url in FSD_FILES.items():
        download(url, FSD_TMP / filename)


def load_fsd_rows() -> list[dict[str, object]]:
    metadata = {}
    for split in ["dev", "eval"]:
        with (FSD_TMP / f"{split}_clips_info_FSD50K.json").open(encoding="utf-8") as f:
            for key, value in json.load(f).items():
                metadata[str(key)] = value

    candidates = []
    used = set()
    priority = list(FSD_LABEL_MAP)
    for split in ["dev", "eval"]:
        with (FSD_TMP / f"{split}.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                fid = str(row["fname"])
                if fid in used:
                    continue
                info = metadata.get(fid)
                if not info or info.get("license") not in LICENSES:
                    continue
                labels = set(row["labels"].split(","))
                for category in priority:
                    matched = [label for label in FSD_LABEL_MAP[category] if label in labels]
                    if matched:
                        candidates.append(
                            {
                                "id": fid,
                                "split": split,
                                "labels": row["labels"],
                                "category": category,
                                "matched_labels": matched,
                                "info": info,
                            }
                        )
                        used.add(fid)
                        break

    rng = random.Random(20260423)
    selected = []
    for category, target in FSD_TARGETS.items():
        category_rows = [row for row in candidates if row["category"] == category]
        rng.shuffle(category_rows)
        selected.extend(category_rows[:target])
        if len(category_rows) < target:
            log(f"warning: only {len(category_rows)} candidates for {category}, target {target}")
    selected.sort(key=lambda row: (str(row["category"]), str(row["id"])))
    return selected


def source_fsd50k_negatives() -> list[dict[str, str]]:
    fetch_fsd_metadata()
    selected = load_fsd_rows()
    log(f"Downloading {len(selected)} FSD50K negative clips...")

    def materialize(row: dict[str, object]) -> dict[str, str]:
        fid = str(row["id"])
        split = str(row["split"])
        category = str(row["category"])
        info = row["info"]
        assert isinstance(info, dict)
        license_url = str(info["license"])
        license_meta = LICENSES[license_url]
        bucket = str(license_meta["bucket"])
        title = str(info.get("title") or f"{fid}.wav")
        creator = str(info.get("uploader") or "")
        slug = slugify(title)
        local_path = EXTERNAL / "negatives" / bucket / category / f"fsd50k__{fid}__{slug}.wav"
        download_url = (
            f"https://huggingface.co/datasets/Fhrozen/FSD50k/resolve/main/"
            f"clips/{split}/{fid}.wav"
        )
        download(download_url, local_path)
        audio = ffprobe(local_path)
        attribution_required = str(license_meta["attribution_required"])
        if attribution_required == "yes":
            attribution = (
                f"{title} by {creator}, Freesound id {fid}, "
                f"{license_meta['name']}; included via FSD50K."
            )
        else:
            attribution = ""
        return {
            "local_path": str(local_path.relative_to(ROOT)),
            "split_candidate": "trainable",
            "polarity": "negative",
            "category": category,
            "source_name": "FSD50K",
            "source_url": f"https://freesound.org/s/{fid}/",
            "download_url": download_url,
            "original_filename": f"{fid}.wav",
            "license": str(license_meta["name"]),
            "license_url": license_url,
            "ai_training_permission": str(license_meta["ai_training_permission"]),
            "commercial_permission": str(license_meta["commercial_permission"]),
            "attribution_required": attribution_required,
            "attribution_text": attribution,
            "creator": creator,
            "duration_sec": audio["duration_sec"],
            "sample_rate_hz": audio["sample_rate_hz"],
            "channels": audio["channels"],
            "file_format": "wav",
            "downloaded_at": DOWNLOADED_AT,
            "notes": (
                "FSD50K label(s): "
                + ", ".join(row["matched_labels"])
                + "; source clip is CC0/CC-BY. FSD50K aggregate is CC-BY and asks "
                + "commercial users to contact dataset authors."
            ),
        }

    rows: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(materialize, row) for row in selected]
        for i, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if i % 100 == 0:
                log(f"  downloaded {i}/{len(selected)}")
    rows.sort(key=lambda row: row["local_path"])
    return rows


def find_local_positive_audio() -> list[Path]:
    paths = []
    for path in ROOT.iterdir():
        if not path.is_dir():
            continue
        if path.name in {".git", ".claude", "data", "frontend", "scripts"}:
            continue
        for audio_path in path.iterdir():
            if audio_path.suffix.lower() == ".m4a":
                paths.append(audio_path)
    return sorted(paths, key=lambda p: p.parent.name)


def source_self_recorded_positives() -> list[dict[str, str]]:
    source_paths = find_local_positive_audio()
    if not source_paths:
        return []
    dest_dir = EXTERNAL / "positives" / "cc0" / "golf_generic_impact"
    dest_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in source_paths:
        folder_slug = slugify(path.parent.name)
        file_slug = slugify(path.stem)
        dest = dest_dir / f"self__{folder_slug}__{file_slug}{path.suffix.lower()}"
        if not dest.exists():
            shutil.copy2(path, dest)
        audio = ffprobe(dest)
        label_note = path.parent.name.strip()
        rows.append(
            {
                "local_path": str(dest.relative_to(ROOT)),
                "split_candidate": "trainable",
                "polarity": "positive",
                "category": "golf_generic_impact",
                "source_name": "self_recorded",
                "source_url": f"self_recorded://{path.relative_to(ROOT)}",
                "download_url": f"self_recorded://{path.relative_to(ROOT)}",
                "original_filename": path.name,
                "license": "owned_by_project",
                "license_url": "",
                "ai_training_permission": "yes",
                "commercial_permission": "yes",
                "attribution_required": "no",
                "attribution_text": "",
                "creator": "project_owner",
                "duration_sec": audio["duration_sec"],
                "sample_rate_hz": audio["sample_rate_hz"],
                "channels": audio["channels"],
                "file_format": path.suffix.lower().lstrip("."),
                "downloaded_at": DOWNLOADED_AT,
                "notes": (
                    f"Local self-recorded golf shot audio copied from {path.relative_to(ROOT)}; "
                    f"folder label: {label_note}. Shot quality labels are source notes only, "
                    f"not normalized ML labels."
                ),
            }
        )
    return rows


def write_manifests(rows: list[dict[str, str]]) -> None:
    csv_path = EXTERNAL / "manifest.csv"
    jsonl_path = EXTERNAL / "manifest.jsonl"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def summarize(rows: list[dict[str, str]]) -> dict[str, object]:
    summary = {
        "trainable_negative": 0,
        "trainable_positive": 0,
        "research_only": 0,
        "reference_only": 0,
        "by_source": {},
        "by_folder": {},
        "by_category": {},
        "by_license": {},
    }
    for row in rows:
        key = f"{row['split_candidate']}_{row['polarity']}"
        if key == "trainable_negative":
            summary["trainable_negative"] += 1
        elif key == "trainable_positive":
            summary["trainable_positive"] += 1
        if row["split_candidate"] == "research_only":
            summary["research_only"] += 1
        if row["split_candidate"] == "reference_only":
            summary["reference_only"] += 1
        for field, bucket in [
            ("source_name", "by_source"),
            ("category", "by_category"),
            ("license", "by_license"),
        ]:
            value = row[field]
            summary[bucket][value] = summary[bucket].get(value, 0) + 1
        folder = str(Path(row["local_path"]).parent)
        summary["by_folder"][folder] = summary["by_folder"].get(folder, 0) + 1
    return summary


def markdown_count_table(title: str, counts: dict[str, int]) -> str:
    lines = [f"## {title}", "", "| Item | Count |", "| --- | ---: |"]
    for item, count in sorted(counts.items()):
        lines.append(f"| `{item}` | {count} |")
    return "\n".join(lines)


def write_docs(rows: list[dict[str, str]]) -> None:
    summary = summarize(rows)
    by_folder = summary["by_folder"]
    by_source = summary["by_source"]
    by_category = summary["by_category"]
    by_license = summary["by_license"]

    readme = [
        "# External Audio Dataset",
        "",
        "Audio sourced for the golf shot detector app following `SOURCING_GUIDE.md`.",
        "",
        "## Counts",
        "",
        "| Split | Count |",
        "| --- | ---: |",
        f"| Trainable negatives | {summary['trainable_negative']} |",
        f"| Trainable positives | {summary['trainable_positive']} |",
        f"| Research-only | {summary['research_only']} |",
        f"| Reference-only | {summary['reference_only']} |",
        "",
        markdown_count_table("Folders", by_folder),
        "",
        markdown_count_table("Categories", by_category),
        "",
        markdown_count_table("Sources", by_source),
        "",
        markdown_count_table("Licenses", by_license),
        "",
        "## Files",
        "",
        "- `manifest.csv`: one provenance row per audio file.",
        "- `manifest.jsonl`: JSONL copy of the same manifest.",
        "- `SOURCES.md`: source and license notes.",
        "- `negatives/`: trainable non-shot audio organized by license and category.",
        "- `positives/`: trainable golf shot audio organized by license and category.",
        "- `reference_only/`: reserved for sources that are not cleared for training.",
    ]
    (EXTERNAL / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")

    sources = [
        "# External Audio Sources",
        "",
        "## FSD50K",
        "",
        "- Source: https://zenodo.org/records/4060432",
        "- Mirror used for file download: https://huggingface.co/datasets/Fhrozen/FSD50k",
        "- Audio origin: Freesound clips with per-clip Creative Commons licenses.",
        "- Included here: only CC0 and CC-BY clips selected from guide-relevant negative classes.",
        "- AI training permission: marked `yes` for CC0/CC-BY based on Freesound's published AI training guidance supplied in the project reference.",
        "- Commercial permission: marked `yes` for the individual CC0/CC-BY clips; note that FSD50K's dataset page asks commercial users to contact the dataset authors.",
        "- Attribution: required for CC-BY rows; attribution text is included per row in the manifests.",
        "",
        "## Self-Recorded Local Golf Audio",
        "",
        f"- Source: local `.m4a` files already present in this workspace on {TODAY}.",
        "- Location: copied into `data/external/positives/cc0/golf_generic_impact/self_recorded_2026-04-23/`.",
        "- License: `owned_by_project`.",
        "- AI training permission: `yes`.",
        "- Commercial permission: `yes`.",
        "- Notes: source folder names such as `pure`, `fat`, and `topped` are preserved in manifest notes but not promoted into normalized ML labels.",
        "",
        "## Not Used",
        "",
        "- Freesound direct API: skipped because no Freesound API/OAuth token was available in the environment.",
        "- Sonniss, Pixabay, YouTube, BBC RemArc, ESC-50, UrbanSound8K, ZapSplat, stock marketplaces: not used for trainable data because the guide marks them prohibited, non-commercial, or unclear for ML/commercial use.",
        "- SoundBible: metadata was inspected, but FSD50K provided stronger labeled coverage for this first batch.",
    ]
    (EXTERNAL / "SOURCES.md").write_text("\n".join(sources) + "\n", encoding="utf-8")


def quality_checks(rows: list[dict[str, str]]) -> None:
    missing = [row["local_path"] for row in rows if not (ROOT / row["local_path"]).exists()]
    if missing:
        raise RuntimeError(f"{len(missing)} manifest files are missing, e.g. {missing[:3]}")
    audio_files = [
        path
        for path in EXTERNAL.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".wav", ".flac", ".aiff", ".aif", ".mp3", ".m4a", ".aac", ".caf"}
    ]
    if len(audio_files) != len(rows):
        raise RuntimeError(f"audio file count {len(audio_files)} != manifest rows {len(rows)}")
    bad_trainable = [
        row["local_path"]
        for row in rows
        if row["split_candidate"] == "trainable"
        and ("reference_only" in Path(row["local_path"]).parts or "research_only" in Path(row["local_path"]).parts)
    ]
    if bad_trainable:
        raise RuntimeError(f"trainable files in non-trainable folders: {bad_trainable[:3]}")


def main() -> int:
    ensure_layout()
    rows = []
    rows.extend(source_self_recorded_positives())
    rows.extend(source_fsd50k_negatives())
    rows.sort(key=lambda row: row["local_path"])
    write_manifests(rows)
    write_docs(rows)
    quality_checks(rows)
    summary = summarize(rows)
    log(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
