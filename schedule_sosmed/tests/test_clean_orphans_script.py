"""Tests for the standalone clean_orphans utility script."""

import json
import os
import subprocess
import sys


def test_clean_orphans_script(tmp_path):
    clips_folder = tmp_path / "clips"
    clips_folder.mkdir()

    # create a couple of files in the clips folder
    (clips_folder / "keep.mp4").write_text("x")
    (clips_folder / "delete.mp4").write_text("x")

    # prepare a clips.json that only references keep.mp4
    clips_json = clips_folder / "clips.json"
    clips_json.write_text(json.dumps([{"filename": "keep.mp4"}]))

    script = os.path.join(os.path.dirname(__file__), "..", "scripts", "clean_orphans.py")
    # run the script (using same interpreter) pointing at our temp queue
    result = subprocess.run(
        [sys.executable, script, "--clips-json", str(clips_json)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    out = result.stdout + result.stderr
    assert "Removed orphan video file" in out
    assert not (clips_folder / "delete.mp4").exists()
    assert (clips_folder / "keep.mp4").exists()
