#!/usr/bin/env python3
"""
apply_cta_to_existing_clips.py

Apply background music and Instagram CTA fade transition to existing clips.

Default input folder: schedule_sosmed/clips/backup/
Use --input-folder to override.
"""

import argparse
import shutil
import sys
from pathlib import Path

# Add project root to path so we can import sosmed modules
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sosmed.cta import append_instagram_cta
from sosmed.config import get_cta_settings
from sosmed.music import apply_music_to_clip
from sosmed.utils import log


def apply_music_and_cta_to_all_clips(input_folder: Path):
    """Apply background music and CTA to all mp4 clips in input_folder."""
    clips_folder = Path(__file__).parent.parent / "clips"
    clips_folder.mkdir(parents=True, exist_ok=True)

    if not input_folder.exists():
        log("ERROR", f"Input folder not found: {input_folder}")
        return

    mp4_files = sorted(input_folder.glob("*.mp4"))
    if not mp4_files:
        log("INFO", f"No .mp4 files found in {input_folder}")
        return

    music_path = project_root / "assets" / "background_music.mp3"
    if not music_path.exists():
        log("WARN", f"Background music not found at {music_path}, skipping music step")
        music_path = None

    cta_config = get_cta_settings()
    if not cta_config.get("enabled", False):
        log("WARN", "CTA is not enabled in config.yaml — proceeding anyway")

    log("INFO", f"Processing {len(mp4_files)} clips from {input_folder}")
    if music_path:
        log("INFO", f"Background music: {music_path}")
    log("INFO", f"Output folder: {clips_folder}")

    processed = 0
    failed = 0

    for clip_path in mp4_files:
        output_path = clips_folder / clip_path.name
        log("INFO", f"Processing: {clip_path.name}...")

        try:
            if music_path:
                ok = apply_music_to_clip(str(clip_path), str(output_path), str(music_path))
                if not ok:
                    log("WARN", "  Music step failed, copying original instead")
                    shutil.copy2(str(clip_path), str(output_path))
            else:
                shutil.copy2(str(clip_path), str(output_path))

            append_instagram_cta(
                clip_path=str(output_path),
                output_path=str(output_path),
                name=str(cta_config.get("name", "Samuel Academy")),
                username=str(cta_config.get("username", "@samuelkoesnadi")),
                duration=float(cta_config.get("duration", 3.0)),
                fade_duration=float(cta_config.get("fade_duration", 0.5)),
            )

            log("OK", f"  Done: {output_path.name}")
            processed += 1

        except Exception as e:
            log("ERROR", f"  Failed: {e}")
            failed += 1

    log("INFO", f"Done! Processed: {processed}, Failed: {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply background music and Instagram CTA to existing clips."
    )
    parser.add_argument(
        "--input-folder",
        type=Path,
        default=Path(__file__).parent.parent / "clips" / "backup",
        help="Folder containing input .mp4 clips (default: schedule_sosmed/clips/backup/)",
    )
    args = parser.parse_args()

    apply_music_and_cta_to_all_clips(args.input_folder)
