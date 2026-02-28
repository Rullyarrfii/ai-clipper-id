#!/usr/bin/env python3
"""Temporary helper: upload every clip listed in clips_temp.json to TikTok.

Run this script from anywhere with:

    python schedule_sosmed/upload_temp.py

The script adds the containing directory to ``sys.path`` so it can import
``scheduler`` and ``config`` without needing package boilerplate.
"""

import os
import sys
import json
import time
import logging

# make it easy to import the modules in the parent folder
HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
if PARENT not in sys.path:
    sys.path.insert(0, PARENT)

import scheduler  # type: ignore
from scheduler import close_tiktok_uploader
from config import CLIPS_FOLDER


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    temp_path = os.path.join(CLIPS_FOLDER, "clips_temp.json")
    if not os.path.exists(temp_path):
        logging.error("clips_temp.json not found: %s", temp_path)
        return

    try:
        with open(temp_path, "r", encoding="utf-8") as f:
            clips = json.load(f)
    except Exception as e:
        logging.error("failed to read clips_temp.json: %s", e)
        return

    if not clips:
        logging.info("no clips in clips_temp.json")
        return

    # process clips and track which ones to remove
    clips_to_remove = []

    for i, clip in enumerate(clips):
        fn = clip.get("filename")
        if not fn:
            logging.warning("skipping entry without filename: %r", clip)
            continue

        video_path = os.path.join(CLIPS_FOLDER, fn)
        if not os.path.exists(video_path):
            logging.warning("video not found, skipping: %s", video_path)
            continue

        logging.info("uploading %s", fn)
        success = scheduler.upload_tiktok(video_path, clip)
        if success:
            logging.info("✅ uploaded %s", fn)
            clips_to_remove.append(clip)
            
            # remove this clip from clips_temp.json immediately
            remaining_clips = [c for c in clips if c != clip]
            try:
                with open(temp_path, "w", encoding="utf-8") as f:
                    json.dump(remaining_clips, f, indent=2, ensure_ascii=False)
                logging.info("📝 removed %s from clips_temp.json", fn)
            except Exception as e:
                logging.error("failed to update clips_temp.json: %s", e)
            
            # polite pause between uploads (still using shared browser)
            if i < len(clips) - 1:
                logging.info("⏳ waiting 123 seconds before next upload...")
                time.sleep(123)
        else:
            logging.error("❌ failed to upload %s", fn)


if __name__ == "__main__":
    try:
        main()
    finally:
        # ensure browser is closed when script exits
        close_tiktok_uploader()
