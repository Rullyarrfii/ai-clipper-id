"""
Post-processing orchestrator: subtitles + music + SFX.

Takes raw extracted clips and applies visual/audio enhancements
in a single FFmpeg pass for efficiency.
"""

import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .subtitles import generate_ass_subtitles, get_clip_words
from .audio_fx import ensure_sfx_exist, build_audio_filter
from .utils import log


def _escape_ass_path(path: str) -> str:
    """Escape file path for FFmpeg's libass subtitle filter.

    libass requires : and \\ to be escaped in paths on all platforms.
    """
    return path.replace("\\", "\\\\").replace(":", "\\:")


def _get_video_info(video_path: str) -> dict[str, Any]:
    """Get video width, height, duration, fps, and audio presence via ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate,duration",
                "-show_entries", "format=duration",
                "-of", "json",
                video_path,
            ],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        fmt = data.get("format", {})

        w = int(stream.get("width", 0))
        h = int(stream.get("height", 0))

        fps_str = stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 30.0
        else:
            fps = float(fps_str)

        dur = float(stream.get("duration", 0) or fmt.get("duration", 0))

        # Check for audio stream
        result2 = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a:0",
                "-show_entries", "stream=codec_type",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True, text=True,
        )
        has_audio = bool(result2.stdout.strip())

        return {
            "width": w, "height": h, "fps": fps,
            "duration": dur, "has_audio": has_audio,
        }
    except Exception as e:
        log("WARN", f"ffprobe failed: {e}")
        return {
            "width": 1920, "height": 1080, "fps": 30.0,
            "duration": 0, "has_audio": True,
        }


def _postprocess_one(
    raw_clip_path: str,
    clip: dict[str, Any],
    segments: list[dict[str, Any]],
    output_dir: Path,
    *,
    subtitles: bool = True,
    music_path: str | None = None,
    sfx: bool = True,
    sfx_paths: dict[str, Path] | None = None,
    subtitle_position: str = "center",
) -> str:
    """Post-process a single clip with all enhancements.

    Returns the path to the post-processed clip.
    """
    raw_path = Path(raw_clip_path)
    out_path = output_dir / f"{raw_path.stem}_final.mp4"
    
    # Ensure output directory is writable
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get info about the raw clip
    info = _get_video_info(raw_clip_path)
    src_w, src_h = info["width"], info["height"]
    clip_duration = info["duration"]
    has_audio = info["has_audio"]

    if clip_duration <= 0:
        clip_duration = clip["end"] - clip["start"]

    out_w = src_w
    out_h = src_h

    # ── 1. Generate subtitles ────────────────────────────────────────────────
    ass_path = None
    if subtitles:
        words = get_clip_words(
            segments,
            clip_start=clip["start"],
            clip_end=clip["end"],
        )
        if words:
            ass_content = generate_ass_subtitles(
                words,
                play_res_x=out_w,
                play_res_y=out_h,
                position=subtitle_position,
            )
            tmp = tempfile.NamedTemporaryFile(
                suffix=".ass", prefix="sosmed_sub_",
                delete=False, mode="w", encoding="utf-8",
            )
            tmp.write(ass_content)
            tmp.close()
            ass_path = tmp.name

    # ── 2. Build video filter chain ──────────────────────────────────────────
    vfilters: list[str] = []

    if ass_path:
        escaped = _escape_ass_path(ass_path)
        vfilters.append(f"ass={escaped}")

    # ── 3. Build audio filter chain ──────────────────────────────────────────
    audio_extra_inputs: list[str] = []
    audio_filter = ""

    use_sfx = sfx and sfx_paths
    use_music = bool(music_path)

    if use_music or use_sfx:
        audio_extra_inputs, audio_filter, _ = build_audio_filter(
            clip_duration=clip_duration,
            has_original_audio=has_audio,
            music_path=music_path,
            sfx_paths=sfx_paths if use_sfx else None,
        )

    # ── 4. Build FFmpeg command ──────────────────────────────────────────────
    cmd: list[str] = ["ffmpeg", "-y", "-hide_banner"]

    # Main input
    cmd.extend(["-i", raw_clip_path])

    # Extra audio inputs (music, SFX)
    cmd.extend(audio_extra_inputs)

    # Build filter_complex string
    filter_parts: list[str] = []

    if vfilters:
        vf_chain = ",".join(vfilters)
        filter_parts.append(f"[0:v]{vf_chain}[vout]")

    if audio_filter:
        filter_parts.append(audio_filter)

    if filter_parts:
        full_filter = ";".join(filter_parts)
        cmd.extend(["-filter_complex", full_filter])

        # Map outputs
        if vfilters:
            cmd.extend(["-map", "[vout]"])
        else:
            cmd.extend(["-map", "0:v:0"])

        if audio_filter:
            # Audio filter produces [aout]
            cmd.extend(["-map", "[aout]"])
        elif has_audio:
            # Direct audio passthrough (if audio input exists)
            cmd.extend(["-map", "0:a:0?"])  # ? makes stream optional
        else:
            # No audio at all
            cmd.append("-an")
    else:
        # No filters at all — simple re-encode
        cmd.extend(["-map", "0:v:0"])
        if has_audio:
            cmd.extend(["-map", "0:a:0?"])
        else:
            cmd.append("-an")

    # Encoding settings — try hardware encoders, fall back to CPU
    # Build audio encoding separately to handle missing audio gracefully
    if has_audio:
        audio_enc = ["-c:a", "aac", "-b:a", "192k"]
    else:
        audio_enc = ["-c:a", "aac"]  # Still set codec even if no input audio

    encode_attempts = [
        ["-c:v", "h264_videotoolbox", "-b:v", "6M"],
        ["-c:v", "h264_nvenc", "-preset", "fast", "-crf", "22"],
        ["-c:v", "libx264", "-preset", "fast", "-crf", "22"],
    ]
    
    # Always include libx264 with minimal settings as absolute fallback
    encode_attempts.append(["-c:v", "libx264", "-preset", "ultrafast", "-q:v", "5"])

    output_flags = ["-shortest", "-movflags", "+faststart", "-loglevel", "error"]

    last_error = None
    for i, enc in enumerate(encode_attempts):
        full_cmd = cmd + enc + audio_enc + output_flags + [str(out_path)]
        try:
            result = subprocess.run(full_cmd, check=True, capture_output=True, text=True)
            break
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            last_error = e
            if isinstance(e, subprocess.CalledProcessError) and e.stderr:
                log("DEBUG", f"Encode attempt {i+1} failed: {e.stderr[:200]}")
            continue
    else:
        error_msg = str(last_error) if last_error else "Unknown error"
        raise RuntimeError(
            f"All encode strategies failed for clip #{clip.get('rank', '?')}: {error_msg}"
        )

    # ── 5. Cleanup ───────────────────────────────────────────────────────────
    if ass_path:
        try:
            os.unlink(ass_path)
        except OSError:
            pass

    # Remove raw clip (replaced by post-processed version)
    try:
        if out_path.exists() and raw_path.exists() and raw_path != out_path:
            raw_path.unlink()
    except OSError:
        pass

    return str(out_path)


def postprocess_clips(
    raw_clip_paths: list[str],
    clips: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    output_dir: Path,
    *,
    max_workers: int = 2,
    subtitles: bool = True,
    music_path: str | None = None,
    sfx: bool = True,
    subtitle_position: str = "center",
) -> list[str]:
    """Post-process all extracted clips in parallel.

    Args:
        raw_clip_paths: Paths to raw extracted clips (from extract_clips).
        clips: Clip metadata list (with start/end/rank/title).
        segments: Original transcription segments (for word timestamps).
        output_dir: Directory for output files.
        max_workers: Parallel workers (lower than extraction due to heavier work).
        subtitles: Enable TikTok subtitles.
        music_path: Path to background music file (None = no music).
        sfx: Enable sound effects.
        subtitle_position: "center", "upper", or "lower".

    Returns:
        List of post-processed clip file paths.
    """
    if not raw_clip_paths:
        return []

    features = []
    if subtitles:
        features.append("subtitles")
    if music_path:
        features.append("music")
    if sfx:
        features.append("SFX")
    log("INFO", f"Post-processing {len(raw_clip_paths)} clips: "
                f"{', '.join(features) or 'none'}")

    # Prepare SFX files (shared across all clips)
    sfx_paths: dict[str, Path] = {}
    if sfx:
        sfx_paths = ensure_sfx_exist()
        if not sfx_paths:
            log("WARN", "SFX enabled but no SFX files were generated")

    # Build a rank→clip lookup for matching raw paths to clip metadata
    # Raw clips are named rank##_... so we match by rank
    rank_to_clip = {c["rank"]: c for c in clips}

    results: list[str] = []

    # Use lower parallelism for post-processing (more CPU intensive)
    effective_workers = min(max_workers, max(1, len(raw_clip_paths)))

    with ThreadPoolExecutor(max_workers=effective_workers) as pool:
        futures = {}
        for raw_path in raw_clip_paths:
            # Extract rank from filename: rank01_Title... or similar
            fname = Path(raw_path).stem
            rank = None
            for c in clips:
                expected_prefix = f"rank{c['rank']:02d}_"
                if fname.startswith(expected_prefix):
                    rank = c["rank"]
                    break

            if rank is None:
                log("WARN", f"Could not match {fname} to any clip — skipping postprocess")
                results.append(raw_path)
                continue

            clip = rank_to_clip[rank]
            fut = pool.submit(
                _postprocess_one,
                raw_path, clip, segments, output_dir,
                subtitles=subtitles,
                music_path=music_path,
                sfx=sfx,
                sfx_paths=sfx_paths,
                subtitle_position=subtitle_position,
            )
            futures[fut] = clip

        for fut in as_completed(futures):
            clip = futures[fut]
            try:
                out = fut.result()
                mb = os.path.getsize(out) / 1_048_576
                log("OK", f"  #{clip['rank']:>2} {clip['title'][:35]:<35}  "
                          f"post-processed ({mb:.1f} MB)")
                results.append(out)
            except Exception as exc:
                log("ERROR", f"  #{clip['rank']:>2} postprocess failed: {exc}")

    return sorted(results)
