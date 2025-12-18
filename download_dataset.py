"""
Download and extract the Kaggle Chest X-Ray Pneumonia dataset.

Steps:
1) Ensure kaggle.json credentials are available (prefer project root; fall back to ~/.kaggle).
2) Use Kaggle API to download the dataset zip.
3) Extract to the target data directory.
4) Verify expected structure: data/chest_xray/train|val|test with NORMAL/PNEUMONIA.

Default target: C:\\CECS456_Project\\data
Override with env var DATA_DIR if desired.
"""

import os
import zipfile
from pathlib import Path
from typing import Dict

from kaggle import api


DATASET = "paultimothymooney/chest-xray-pneumonia"
DEFAULT_DATA_ROOT = Path(r"C:\CECS456_Project\data")
EXPECTED_SPLITS = ("train", "val", "test")
EXPECTED_CLASSES = ("NORMAL", "PNEUMONIA")


def ensure_kaggle_credentials(project_root: Path) -> None:
    """Place kaggle.json in ~/.kaggle if not already present."""
    home_kaggle_dir = Path.home() / ".kaggle"
    home_kaggle_dir.mkdir(exist_ok=True)
    home_creds = home_kaggle_dir / "kaggle.json"

    project_creds = project_root / "kaggle.json"
    if home_creds.exists():
        print(f"Found existing credentials at {home_creds}")
        return
    if not project_creds.exists():
        raise FileNotFoundError(
            f"kaggle.json not found. Place it either at {project_creds} or {home_creds}."
        )
    home_creds.write_bytes(project_creds.read_bytes())
    try:
        os.chmod(home_creds, 0o600)
    except PermissionError:
        # On Windows this may fail; safe to ignore.
        pass
    print(f"Copied kaggle.json to {home_creds}")


def download_zip(target_dir: Path) -> Path:
    """Download dataset zip to target_dir and return zip path."""
    target_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET} to {target_dir} ...")
    api.dataset_download_files(DATASET, path=str(target_dir), unzip=False, quiet=False)

    zip_path = target_dir / "chest-xray-pneumonia.zip"
    if not zip_path.exists():
        # Kaggle names it after the dataset slug.
        zip_candidates = list(target_dir.glob("*.zip"))
        if not zip_candidates:
            raise FileNotFoundError("Zip file not found after download.")
        zip_path = zip_candidates[0]
    print(f"Downloaded zip: {zip_path}")
    return zip_path


def extract_zip(zip_path: Path, target_dir: Path) -> Path:
    """Extract zip into target_dir and return the path to the chest_xray folder."""
    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(target_dir)

    chest_dir = target_dir / "chest_xray"
    if not chest_dir.exists():
        # Some archives might nest differently; attempt to locate.
        candidates = list(target_dir.glob("**/chest_xray"))
        if candidates:
            chest_dir = candidates[0]
        else:
            raise FileNotFoundError("Could not locate chest_xray folder after extraction.")
    print(f"Dataset extracted to {chest_dir}")
    return chest_dir


def verify_structure(chest_dir: Path) -> Dict[str, Dict[str, int]]:
    """Verify expected split/class folders and return counts."""
    counts = {}
    for split in EXPECTED_SPLITS:
        split_dir = chest_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing split folder: {split_dir}")
        counts[split] = {}
        for cls in EXPECTED_CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {cls_dir}")
            n_imgs = len(list(cls_dir.glob("*.*")))
            counts[split][cls] = n_imgs
    return counts


def main():
    project_root = Path.cwd()
    ensure_kaggle_credentials(project_root)

    data_root = Path(os.environ.get("DATA_DIR", DEFAULT_DATA_ROOT))
    zip_path = download_zip(data_root)
    chest_dir = extract_zip(zip_path, data_root)

    counts = verify_structure(chest_dir)
    print("Verified structure and counts:")
    for split, cls_counts in counts.items():
        for cls, n in cls_counts.items():
            print(f"{split}/{cls}: {n} images")
    print("\nDone. Run notebooks from the project root to train/evaluate.")


if __name__ == "__main__":
    main()

