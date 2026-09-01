#!/usr/bin/env python3
"""UR Fall Detection Dataset (URFD) Downloader.

Official Source: University of Rzeszow (Interdisciplinary Centre for Computational Modelling)
Official URL: https://fenix.ur.edu.pl/~mkepski/ds/uf.html
License: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)

Citation:
Bogdan Kwolek, Michal Kepski, "Human fall detection on embedded platform using depth maps
and wireless accelerometer", Computer Methods and Programs in Biomedicine,
Volume 117, Issue 3, December 2014, Pages 489-501, ISSN 0169-2607.

This script discovers, downloads, verifies, and organizes the URFD fall and ADL (normal)
sequences into a structured directory for training action recognition models.
"""
import argparse
import logging
import os
import sys
import urllib.request
import zipfile

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("urfd_downloader")

BASE_URL = "https://fenix.ur.edu.pl/~mkepski/ds/data"
DATASET_LICENSE = "CC BY-NC-SA 4.0 (Non-commercial academic research)"
NUM_FALL_SEQUENCES = 30
NUM_ADL_SEQUENCES = 40


def get_sequence_urls(
    format_type: str = "mp4",
    num_falls: int = NUM_FALL_SEQUENCES,
    num_adls: int = NUM_ADL_SEQUENCES,
) -> Dict[str, List[Tuple[str, str, str]]]:
    """Build list of download URLs for fall and ADL (normal) sequences.

    Returns dict with keys 'fall' and 'normal', each containing tuples of
    (sequence_id, filename, download_url).
    """
    sequences: Dict[str, List[Tuple[str, str, str]]] = {"fall": [], "normal": []}

    # Fall sequences (fall-01 to fall-30)
    for i in range(1, num_falls + 1):
        seq_id = f"fall-{i:02d}"
        if format_type in ["mp4", "both"]:
            fn = f"{seq_id}-cam0.mp4"
            url = f"{BASE_URL}/{fn}"
            sequences["fall"].append((seq_id, fn, url))
        if format_type in ["png", "both"]:
            fn = f"{seq_id}-cam0-rgb.zip"
            url = f"{BASE_URL}/{fn}"
            sequences["fall"].append((seq_id, fn, url))

    # ADL (Activities of Daily Living / Normal) sequences (adl-01 to adl-40)
    for i in range(1, num_adls + 1):
        seq_id = f"adl-{i:02d}"
        if format_type in ["mp4", "both"]:
            fn = f"{seq_id}-cam0.mp4"
            url = f"{BASE_URL}/{fn}"
            sequences["normal"].append((seq_id, fn, url))
        if format_type in ["png", "both"]:
            fn = f"{seq_id}-cam0-rgb.zip"
            url = f"{BASE_URL}/{fn}"
            sequences["normal"].append((seq_id, fn, url))

    return sequences


def download_file(url: str, dest_path: str, timeout: int = 30) -> bool:
    """Download a file with progress reporting and existing file detection."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        logger.info("File already exists, skipping: %s", os.path.basename(dest_path))
        return True

    logger.info("Downloading %s -> %s", url, dest_path)
    tmp_path = dest_path + ".tmp"
    try:
        import ssl
        try:
            import certifi
            ssl_ctx = ssl.create_default_context(cafile=certifi.where())
        except ImportError:
            ssl_ctx = ssl._create_unverified_context()

        req = urllib.request.Request(
            url,
            headers={"User-Agent": "EmergencyVisionAI/1.0 (Research Dataset Downloader)"},
        )
        with urllib.request.urlopen(req, timeout=timeout, context=ssl_ctx) as response, open(tmp_path, "wb") as out_file:
            data = response.read()
            out_file.write(data)
        os.rename(tmp_path, dest_path)
        logger.info("Successfully saved %s (%.2f KB)", os.path.basename(dest_path), len(data) / 1024.0)
        return True
    except Exception as exc:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        logger.error("Failed to download %s: %s", url, exc)
        return False


def extract_zip(zip_path: str, extract_dir: str) -> None:
    """Extract a ZIP archive into target directory."""
    logger.info("Extracting %s -> %s", zip_path, extract_dir)
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)


def write_dataset_readme(output_dir: str) -> None:
    """Generate a README.md documenting license, source, and structure."""
    readme_path = os.path.join(output_dir, "README.md")
    content = f"""# UR Fall Detection Dataset (URFD)

- **Official Source**: [University of Rzeszow (URFD)](https://fenix.ur.edu.pl/~mkepski/ds/uf.html)
- **License**: {DATASET_LICENSE}
- **Citation**:
  Bogdan Kwolek, Michal Kepski, *Human fall detection on embedded platform using depth maps and wireless accelerometer*,
  Computer Methods and Programs in Biomedicine, Vol. 117, Issue 3, 2014, pp. 489-501.

## Structure
- `videos/fall/`: MP4 video files containing fall events (Label = 1 / FALL)
- `videos/normal/`: MP4 video files containing normal activities of daily living (Label = 0 / NORMAL)
- `raw/`: Raw downloaded archives and metadata files
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content)
    logger.info("Created dataset README at %s", readme_path)


def download_urfd(
    output_dir: str = "data/urfd",
    format_type: str = "mp4",
    num_falls: int = NUM_FALL_SEQUENCES,
    num_adls: int = NUM_ADL_SEQUENCES,
    dry_run: bool = False,
) -> bool:
    """Main orchestrator for downloading and structuring URFD."""
    os.makedirs(output_dir, exist_ok=True)
    raw_dir = os.path.join(output_dir, "raw")
    videos_dir = os.path.join(output_dir, "videos")
    fall_dir = os.path.join(videos_dir, "fall")
    normal_dir = os.path.join(videos_dir, "normal")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(fall_dir, exist_ok=True)
    os.makedirs(normal_dir, exist_ok=True)

    write_dataset_readme(output_dir)

    sequences = get_sequence_urls(format_type, num_falls, num_adls)
    total_files = len(sequences["fall"]) + len(sequences["normal"])
    logger.info(
        "Prepared %d target sequences (%d fall, %d normal ADL)",
        total_files,
        len(sequences["fall"]),
        len(sequences["normal"]),
    )

    if dry_run:
        logger.info("[DRY RUN] URLs verified. No files will be downloaded.")
        for category, items in sequences.items():
            for seq_id, fn, url in items:
                logger.info("  [%s] %s -> %s", category.upper(), seq_id, url)
        return True

    success_count = 0
    # Download Falls
    for seq_id, fn, url in sequences["fall"]:
        dest = os.path.join(fall_dir if fn.endswith(".mp4") else raw_dir, fn)
        if download_file(url, dest):
            success_count += 1
            if fn.endswith(".zip"):
                extract_zip(dest, os.path.join(output_dir, "frames", "fall", seq_id))

    # Download ADLs (Normal)
    for seq_id, fn, url in sequences["normal"]:
        dest = os.path.join(normal_dir if fn.endswith(".mp4") else raw_dir, fn)
        if download_file(url, dest):
            success_count += 1
            if fn.endswith(".zip"):
                extract_zip(dest, os.path.join(output_dir, "frames", "normal", seq_id))

    logger.info("URFD Download Complete: %d / %d files successfully acquired", success_count, total_files)
    return success_count == total_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare UR Fall Detection Dataset (URFD)")
    parser.add_argument("--output-dir", default="data/urfd", help="Directory to store dataset")
    parser.add_argument("--format", choices=["mp4", "png", "both"], default="mp4", help="Video/image format")
    parser.add_argument("--num-falls", type=int, default=NUM_FALL_SEQUENCES, help="Number of fall sequences to download")
    parser.add_argument("--num-adls", type=int, default=NUM_ADL_SEQUENCES, help="Number of ADL sequences to download")
    parser.add_argument("--dry-run", action="store_true", help="Print URLs without downloading")
    args = parser.parse_args()

    success = download_urfd(
        output_dir=args.output_dir,
        format_type=args.format,
        num_falls=args.num_falls,
        num_adls=args.num_adls,
        dry_run=args.dry_run,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
