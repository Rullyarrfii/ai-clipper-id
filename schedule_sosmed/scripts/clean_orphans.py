#!/usr/bin/env python3
"""Delete orphan MP4 files in the schedule_sosmed clips folder.

By default the script looks for ``schedule_sosmed/clips/clips.json`` relative

run from the workspace root, but you can override the path with
``--clips-json``.  This is handy if you want to point at a different queue or
perform a dry run.

Examples
--------

    python schedule_sosmed/scripts/clean_orphans.py            # delete
    python schedule_sosmed/scripts/clean_orphans.py --dry-run   # don't

"""

import argparse
import json
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Remove any .mp4 files under a clips folder that aren't "
                    "listed in the accompanying clips.json")
    parser.add_argument(
        "--clips-json",
        help="Path to the clips.json file to consult (defaults to the one in "
             "the same directory as this script's parent +/clips).",
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be removed; do not delete anything.",
    )
    args = parser.parse_args()

    if args.clips_json is None:
        # derive default location: ../clips/clips.json relative to this file
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        args.clips_json = os.path.join(base, "clips", "clips.json")

    clips_json = os.path.abspath(args.clips_json)
    if not os.path.exists(clips_json):
        sys.stderr.write(f"clips.json not found: {clips_json}\n")
        return 1

    try:
        with open(clips_json, "r", encoding="utf-8") as f:
            clips = json.load(f)
    except Exception as e:
        sys.stderr.write(f"failed to load {clips_json}: {e}\n")
        return 1

    valid = {c.get("filename") for c in clips if c.get("filename")}
    clips_folder = os.path.dirname(clips_json)
    deleted = 0

    for fname in os.listdir(clips_folder):
        if not fname.lower().endswith(".mp4"):
            continue
        if fname not in valid:
            path = os.path.join(clips_folder, fname)
            if args.dry_run:
                print(f"Would remove orphan video file: {path}")
            else:
                try:
                    os.remove(path)
                    print(f"Removed orphan video file: {path}")
                    deleted += 1
                except Exception as e:
                    sys.stderr.write(f"failed to delete {path}: {e}\n")
    if not args.dry_run:
        print(f"Deleted {deleted} orphan file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
