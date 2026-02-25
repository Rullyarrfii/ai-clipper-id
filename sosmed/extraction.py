"""
Clip extraction: uses ffmpeg to extract video segments.
"""

import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .utils import get_ffmpeg, get_ffprobe, log


def _get_video_duration(path: str) -> float | None:
    """Get video duration in seconds via ffprobe."""
    try:
        out = subprocess.run(
            [
                get_ffprobe(), "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def _extract_one(
    video_path: str,
    clip: dict[str, Any],
    output_dir: Path,
    *,
    padding: float = 0.35,
) -> str:
    """Extract a single clip with ffmpeg (frame-accurate). Returns output file path.

    Uses *output-seeking* (``-ss`` after ``-i``) with ``-copyts`` so the
    seek is frame-accurate rather than snapping to keyframes.  A higher
    default padding (0.35 s) ensures we don't clip into speech.
    """
    start = max(0.0, clip["start"] - padding)
    end = clip["end"] + padding
    duration = end - start

    safe = re.sub(r"[^\w\s-]", "", clip.get("title", f"clip_{clip['rank']}"))
    safe = re.sub(r"\s+", "_", safe)[:50]
    out_path = output_dir / f"rank{clip['rank']:02d}_{safe}.mp4"

    # -i first, then -ss/-t  → output seeking = frame-accurate
    base_cmd = [
        get_ffmpeg(), "-y", "-hide_banner",
        "-i", video_path,
        "-ss", f"{start:.3f}",
        "-t",  f"{duration:.3f}",
        "-map", "0:v:0",
        "-map", "0:a:0?",
    ]
    encode_args = ["-c:v", "copy"]
    audio_flags = ["-c:a", "aac", "-b:a", "128k"]
    output_flags = ["-movflags", "+faststart", "-loglevel", "error", str(out_path)]

    cmd = base_cmd + encode_args + audio_flags + output_flags
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        log("DEBUG", f"Clip #{clip['rank']} encoded")
        return str(out_path)
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        error_detail = e.stderr[:150] if isinstance(e, subprocess.CalledProcessError) and e.stderr else str(e)
        error_msg = error_detail
        raise RuntimeError(f"Encoding failed for clip #{clip['rank']}: {error_msg}")


def extract_clips(
    video_path: str,
    clips: list[dict[str, Any]],
    output_dir: Path,
    max_workers: int = 4,
) -> list[str]:
    """Extract all clips in parallel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log("INFO", f"Extracting {len(clips)} clips → {output_dir}/ ({max_workers} workers)")

    results: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(_extract_one, video_path, c, output_dir): c
            for c in clips
        }
        for fut in as_completed(futs):
            clip = futs[fut]
            try:
                out = fut.result()
                mb = os.path.getsize(out) / 1_048_576
                log("OK", f"  #{clip['rank']:>2} {clip['title'][:40]:<40}  "
                          f"{Path(out).name} ({mb:.1f} MB)")
                results.append(out)
            except Exception as exc:
                log("ERROR", f"  #{clip['rank']:>2} failed: {exc}")

    return sorted(results)
