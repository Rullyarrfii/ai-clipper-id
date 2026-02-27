"""Utility to add filename entries to existing clips.json files.

Run from workspace root (where the "clips" directory lives):

    python scripts/fix_clips.py

It will walk each subdirectory under clips/ and update any JSON that
lacks a "filename" field, inferring the name from existing MP4 files or
regenerating a safe name.
"""

import json
import os
import re
from pathlib import Path


def make_safe_name(clip: dict) -> str:
    rank = clip.get("rank", 0)
    safe = re.sub(r"[^\w\s-]", "", clip.get("title", f"clip_{rank}"))
    safe = re.sub(r"\s+", "_", safe)[:50]
    return f"rank{rank:02d}_{safe}_final.mp4"


def fix_directory(dirpath: Path) -> bool:
    changed = False
    clips_file = dirpath / "clips.json"
    if not clips_file.exists():
        return False
    
    text = clips_file.read_text(encoding='utf-8')
    if not text.strip():
        print(f"WARNING: {clips_file} is empty - skipping (regenerate using ai-clipper)")
        return False
    
    try:
        data = json.loads(text)
    except Exception as e:
        print(f"failed to load {clips_file}: {e}")
        return False
    
    if not data:
        print(f"WARNING: {clips_file} has no clips - skipping")
        return False
    
    # Only add filename field to existing entries, preserve all other fields
    for c in data:
        if "filename" not in c or not c["filename"]:
            # try to locate actual file by rank
            rank = c.get("rank")
            match = None
            for fname in os.listdir(dirpath):
                if fname.startswith(f"rank{rank:02d}_") and fname.endswith(".mp4"):
                    match = fname
                    break
            if match:
                c["filename"] = match
            else:
                c["filename"] = make_safe_name(c)
            changed = True
    
    if changed:
        clips_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')
        print(f"updated {clips_file}")
    return changed


def main():
    base = Path("clips")
    if not base.exists():
        print("no clips directory found")
        return
    for sub in base.iterdir():
        if sub.is_dir():
            fix_directory(sub)


if __name__ == "__main__":
    main()
