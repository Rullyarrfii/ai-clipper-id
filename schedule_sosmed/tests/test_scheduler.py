import json
import os
import shutil
import tempfile
import pytest

import scheduler


def make_clip(filename, rank=1, title=""):  # helper
    return {"filename": filename, "rank": rank, "title": title}


def test_dedupe_clips():
    clips = [make_clip("a.mp4"), make_clip("b.mp4"), make_clip("a.mp4")]
    out = scheduler.dedupe_clips(clips)
    assert len(out) == 2
    assert out[0]["filename"] == "a.mp4"
    assert out[1]["filename"] == "b.mp4"


def test_mark_done_removes_by_filename(tmp_path, monkeypatch):
    # set up a fake clips folder and file
    clips_folder = tmp_path / "clips"
    clips_folder.mkdir()
    monkeypatch.setattr(scheduler, "CLIPS_FOLDER", str(clips_folder))
    queue_file = clips_folder / "clips.json"
    monkeypatch.setattr(scheduler, "CLIPS_JSON", str(queue_file))

    clips = [make_clip("one.mp4", rank=1), make_clip("two.mp4", rank=2)]
    with open(queue_file, "w", encoding="utf-8") as f:
        json.dump(clips, f)

    # create actual files
    for fn in ["one.mp4", "one_padded.mp4", "two.mp4"]:
        (clips_folder / fn).write_text("x")

    # remove first clip
    clip, path = scheduler.get_clip_by_filename("one.mp4")
    assert clip is not None
    result = scheduler.mark_done(clip)
    assert result
    assert not (clips_folder / "one.mp4").exists()
    assert not (clips_folder / "one_padded.mp4").exists()
    # queue now contains only 'two'
    remaining = json.load(open(queue_file))
    assert len(remaining) == 1
    assert remaining[0]["filename"] == "two.mp4"


def test_clean_orphans(tmp_path, monkeypatch):
    clips_folder = tmp_path / "clips"
    clips_folder.mkdir()
    monkeypatch.setattr(scheduler, "CLIPS_FOLDER", str(clips_folder))

    # create some files
    (clips_folder / "keep.mp4").write_text("x")
    (clips_folder / "delete.mp4").write_text("x")
    clips = [make_clip("keep.mp4")]
    deleted = scheduler.clean_orphan_files(clips)
    assert deleted == 1
    assert (clips_folder / "delete.mp4").exists() is False
    assert (clips_folder / "keep.mp4").exists()


def test_rebuild_queue(tmp_path, monkeypatch):
    # create master file with three clips
    master = [make_clip("a.mp4"), make_clip("b.mp4"), make_clip("c.mp4")]
    master_path = tmp_path / "master.json"
    master_path.write_text(json.dumps(master))

    # fake posted filenames
    monkeypatch.setattr(scheduler, "get_posted_filenames", lambda: {"b.mp4"})
    # create a dummy CLIPS_JSON for writing
    clips_folder = tmp_path / "clips"
    clips_folder.mkdir()
    queue = clips_folder / "clips.json"
    monkeypatch.setattr(scheduler, "CLIPS_FOLDER", str(clips_folder))
    monkeypatch.setattr(scheduler, "CLIPS_JSON", str(queue))

    scheduler.rebuild_queue(str(master_path))
    new = json.load(open(queue))
    assert len(new) == 2
    filenames = {c["filename"] for c in new}
    assert filenames == {"a.mp4", "c.mp4"}


def test_prune_posted(tmp_path, monkeypatch):
    clips_folder = tmp_path / "clips"
    clips_folder.mkdir()
    queue = clips_folder / "clips.json"
    monkeypatch.setattr(scheduler, "CLIPS_FOLDER", str(clips_folder))
    monkeypatch.setattr(scheduler, "CLIPS_JSON", str(queue))
    clips = [make_clip("a.mp4"), make_clip("b.mp4")]
    queue.write_text(json.dumps(clips))
    monkeypatch.setattr(scheduler, "get_posted_filenames", lambda: {"a.mp4"})
    scheduler.main(rebuild=False, master=None, clean_orphans_flag=False, prune_posted=True, dry_run=False)
    remaining = json.load(open(queue))
    assert len(remaining) == 1
    assert remaining[0]["filename"] == "b.mp4"


def test_load_clips_warns_and_dedup(tmp_path, monkeypatch, caplog):
    # create queue file with duplicate and missing filename
    clips_folder = tmp_path / "clips"
    clips_folder.mkdir()
    queue = clips_folder / "clips.json"
    monkeypatch.setattr(scheduler, "CLIPS_FOLDER", str(clips_folder))
    monkeypatch.setattr(scheduler, "CLIPS_JSON", str(queue))
    entries = [make_clip("x.mp4"), {"rank": 2, "title": "no name"}, make_clip("x.mp4")]
    queue.write_text(json.dumps(entries))
    caplog.set_level("WARNING")
    loaded = scheduler.load_clips()
    # duplicate removed; missing-filename entry is left but warned about
    assert len(loaded) == 2
    assert "duplicate clip entry removed" in caplog.text
    assert "missing 'filename'" in caplog.text
