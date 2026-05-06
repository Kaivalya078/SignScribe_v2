"""
Extract and normalize hand landmarks from ASL image datasets.

The extractor accepts one or more dataset roots. Each root can contain class
folders directly, or nested inside Kaggle-style folders such as Train_Alphabet.

Usage, from the project root:
    python utils/extract_landmark.py

    python utils/extract_landmark.py ^
        --datasets dataset/asl_alphabet_train dataset/kaggle/lexset_synthetic_asl_alphabet ^
                   dataset/kaggle/prathumarikeri_american_sign_language_09az ^
        --output dataset/asl_landmarks.csv
"""

from __future__ import annotations

import argparse
import os
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

_DEFAULT_DATASETS = [
    _PROJECT_ROOT / "dataset" / "asl_alphabet_train",
    _PROJECT_ROOT / "dataset" / "kaggle" / "American",
    _PROJECT_ROOT / "dataset" / "kaggle" / "Train_Alphabet",
]
_DEFAULT_OUTPUT = _PROJECT_ROOT / "dataset" / "asl_landmarks.csv"
_MODEL_PATH = _PROJECT_ROOT / "checkpoints" / "hand_landmarker.task"
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
_SUPPORTED_LABELS = {
    *(chr(code) for code in range(ord("A"), ord("Z") + 1)),
    "del",
    "nothing",
    "space",
}
_LABEL_ALIASES = {
    "background": "nothing",
    "backgrounds": "nothing",
    "delete": "del",
    "deletion": "del",
    "none": "nothing",
    "blank": "nothing",
}


def _ensure_model() -> Path:
    """Download hand_landmarker.task if it is not already present."""
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _MODEL_PATH.exists():
        print(f"Downloading hand_landmarker.task to {_MODEL_PATH} (~9 MB)...")
        urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
        print("[OK] Download complete.")
    return _MODEL_PATH


def _normalize_label(raw_label: str) -> str | None:
    """Map a folder name to a training label, or return None if unsupported."""
    clean = raw_label.strip().replace("-", "_").replace(" ", "_")
    lower = clean.lower()

    if len(clean) == 1 and clean.isalpha():
        return clean.upper()

    mapped = _LABEL_ALIASES.get(lower, lower)
    if mapped in _SUPPORTED_LABELS:
        return mapped

    return None


def _has_images(path: Path) -> bool:
    try:
        return any(
            child.is_file() and child.suffix.lower() in _IMAGE_EXTENSIONS
            for child in path.iterdir()
        )
    except OSError:
        return False


def _discover_label_folders(dataset_path: Path) -> list[tuple[str, Path]]:
    """Find direct or nested class folders in a dataset root."""
    label_folders: list[tuple[str, Path]] = []

    for current, dir_names, _ in os.walk(dataset_path):
        current_path = Path(current)
        label = _normalize_label(current_path.name)
        if label and _has_images(current_path):
            label_folders.append((label, current_path))
            dir_names[:] = []

    return sorted(label_folders, key=lambda item: (item[0], str(item[1]).lower()))


def _extract_from_label_folder(label: str, class_path: Path, landmarker) -> list[list]:
    """Extract normalized landmarks from one label folder."""
    landmark_data: list[list] = []
    extracted = 0
    skipped = 0

    for image_path in sorted(class_path.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in _IMAGE_EXTENSIONS:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)

        if not result.hand_landmarks:
            skipped += 1
            continue

        raw_landmarks = result.hand_landmarks[0]
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in raw_landmarks])

        base_point = landmarks[0].copy()
        relative = landmarks - base_point
        max_abs = np.max(np.abs(relative))
        normalized = relative / max_abs if max_abs > 0 else relative

        landmark_data.append([label] + normalized.flatten().tolist())
        extracted += 1

    print(f"    {label} ({class_path}): {extracted} extracted, {skipped} skipped")
    return landmark_data


def _extract_from_folder(dataset_path: Path, landmarker) -> list[list]:
    """Extract landmarks from a dataset root."""
    if not dataset_path.is_dir():
        print(f"  [SKIP] Not found: {dataset_path}")
        return []

    label_folders = _discover_label_folders(dataset_path)
    if not label_folders:
        print(f"  [SKIP] No supported label folders with images found in: {dataset_path}")
        return []

    landmark_data: list[list] = []
    for label, class_path in label_folders:
        landmark_data.extend(_extract_from_label_folder(label, class_path, landmarker))

    print(f"  Subtotal: {len(landmark_data)} extracted from {dataset_path.name}")
    return landmark_data


def extract_and_normalize_landmarks(
    dataset_paths: list[str | Path],
    output_csv_path: str | Path,
) -> None:
    """Extract, normalize, and save hand landmarks from one or more image roots."""
    output_csv_path = Path(output_csv_path)
    paths = [Path(p) for p in dataset_paths]

    print(f"\nDatasets to process ({len(paths)}):")
    for path in paths:
        status = "found" if path.is_dir() else "NOT FOUND"
        print(f"  [{status}] {path}")
    print()

    model_path = _ensure_model()

    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    header = ["label"] + [
        f"{axis}{idx}" for idx in range(21) for axis in ("x", "y", "z")
    ]

    all_data: list[list] = []

    with HandLandmarker.create_from_options(options) as landmarker:
        for idx, dataset_path in enumerate(paths, start=1):
            print(f"[{idx}/{len(paths)}] Extracting from: {dataset_path}")
            all_data.extend(_extract_from_folder(dataset_path, landmarker))
            print()

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_data, columns=header)
    df.to_csv(output_csv_path, index=False)

    if df.empty:
        print("WARNING: No landmarks were extracted. Check dataset paths and image quality.")
    else:
        print("Class distribution in merged CSV:")
        counts = df["label"].value_counts().sort_index()
        for label, count in counts.items():
            print(f"  {label}: {count}")

    print(f"\nDone. {len(df)} total samples saved to {output_csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe hand landmarks from ASL image datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="PATH",
        help=(
            "One or more dataset roots. Nested Kaggle folders are supported. "
            "Defaults include dataset/asl_alphabet_train and the recommended "
            "local Kaggle dataset folders under dataset/kaggle/."
        ),
    )
    parser.add_argument(
        "--output",
        default=str(_DEFAULT_OUTPUT),
        metavar="PATH",
        help=f"Output CSV path. Default: {_DEFAULT_OUTPUT}",
    )
    args = parser.parse_args()

    dataset_paths = [Path(p) for p in args.datasets] if args.datasets else _DEFAULT_DATASETS
    extract_and_normalize_landmarks(dataset_paths, args.output)


if __name__ == "__main__":
    main()
