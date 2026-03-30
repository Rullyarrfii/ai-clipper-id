"""
Process a single video (or a folder of videos) to extract the best clip with overlay subtitles.
Output: topic and title printed to console.
Usage:
  python sosmed/process_single.py path/to/video.mp4 [options]
  python sosmed/process_single.py path/to/folder/  [options]   # batch mode
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

from .transcription import transcribe
from .prefilter import prefilter_segments
from .llm import generate_single_clip_metadata, translate_subtitle_words
from .extraction import extract_clips, _get_video_duration
from .postprocess import postprocess_clips
from .config import get_defaults, get_cta_settings
from .utils import (
    log, BOLD, RESET, CYAN, GREEN, YELLOW,
    strip_internal_fields, get_internal_fields, get_clips_cache_dir
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv", ".m4v"}


def _get_transcript_cache_path(video_path: str) -> Path:
    """Return cache path for transcript based on video filename."""
    video = Path(video_path)
    cache_dir = Path.cwd() / ".cache" / "ai-video-clipper"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{video.stem}_transcript.json"
    return cache_file


def _build_full_video_clip(
    video: Path,
    video_duration: float,
    segments: list[dict],
    title: str | None = None,
    caption: str | None = None,
) -> dict:
    """Build a single clip that covers the entire video."""
    clip_start = 0.0
    if segments:
        clip_start = max(0.0, min(float(segments[0].get("start", 0.0)), video_duration))

    default_title = re.sub(r"[_-]+", " ", video.stem).strip() or video.stem
    final_title = title if title else default_title
    final_caption = caption if caption else default_title

    return {
        "rank": 1,
        "start": clip_start,
        "end": video_duration,
        "title": final_title,
        "topic": final_title,
        "caption": final_caption,
        "reason": "Single-video mode returns the full source video.",
        "hook": "",
        "score_hook": 100,
        "score_insight_density": 100,
        "score_retention": 100,
        "score_emotional_payoff": 100,
        "score_clarity": 100,
        "clip_score": 100,
    }


def _should_translate_to_indonesian(detected_language: dict | None) -> bool:
    """Return True when subtitle translation to Indonesian is still needed."""
    if not detected_language:
        return True

    lang = str(detected_language.get("language", "")).lower()
    prob = float(detected_language.get("language_probability", 0.0) or 0.0)
    return not (lang == "id" and prob > 0.6)


def process_single_video(
    video_path: str,
    model: str = "turbo",
    lang: str | None = "id",
    device: str = "auto",
    compute_type: str = "auto",
    no_vad: bool = False,
    vad_min_silence: int = 400,
    vad_speech_pad: int = 200,
    batch: int = 16,
    chunk_duration: float = 360.0,
    chunk_overlap: float = 60.0,
    min_duration: int = 5,
    max_duration: int = 60,
    output_dir: str = "output",
    api_key: str | None = None,
    llm_model: str | None = None,
    subtitles: bool = True,
    subtitle_position: str = "lower",
    subtitle_margin_pct: float | None = None,
    title: str | None = None,
    caption: str | None = None,
    cta: bool | None = None,
) -> dict:
    """
    Process a single video and extract the best clip.

    Returns:
        dict with keys: 'topic', 'title', 'video_path', 'start', 'end', 'clip' (full clip dict)
    """
    video = Path(video_path)
    if not video.exists():
        log("ERROR", f"File not found: {video}")
        sys.exit(1)

    output_path = Path(output_dir) / video.stem

    print(f"\n{BOLD}{CYAN}{'═' * 50}")
    print(f"   Single Video Processor")
    print(f"{'═' * 50}{RESET}")
    print(f"  Video     : {video.name}")
    print(f"  Model     : {model}")
    print(f"  Language  : {lang or 'auto-detect'}")
    print()

    t_total = time.time()

    # ── 1. Transcribe ────────────────────────────────────────────────────────
    cache_path = _get_transcript_cache_path(str(video))
    detected_language = {"language": "unknown", "language_probability": 0.0}
    if cache_path.exists():
        log("INFO", f"Loading cached transcript from {cache_path}")
        cache_data = json.loads(cache_path.read_text())
        # Handle both old format (list) and new format (dict with segments and language_info)
        if isinstance(cache_data, dict) and "segments" in cache_data:
            segments = cache_data["segments"]
            detected_language = cache_data.get("language_info", detected_language)
        else:
            segments = cache_data  # Old format, just a list
        log("OK", f"Loaded {len(segments)} segments from cache")
    else:
        segments, detected_language = transcribe(
            str(video),
            model_size=model,
            language=lang,
            device=device,
            compute_type=compute_type,
            vad_filter=not no_vad,
            vad_min_silence_ms=vad_min_silence,
            vad_speech_pad_ms=vad_speech_pad,
            batch_size=batch,
        )
        # Save to cache (new format with language info)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {"segments": segments, "language_info": detected_language}
        cache_path.write_text(json.dumps(cache_data, indent=2, ensure_ascii=False))
        log("OK", f"Transcript cached → {cache_path}")

    # ── 2. Pre-filter ────────────────────────────────────────────────────────
    filtered, stats = prefilter_segments(segments)
    log("OK", f"Pre-filtered: {stats['original']} → {stats['kept']} segments")

    if not filtered:
        log("ERROR", "All segments filtered out. Try a different whisper model or looser filters.")
        sys.exit(1)

    # ── 3. Full-video single clip ───────────────────────────────────────────
    output_path.mkdir(parents=True, exist_ok=True)
    video_dur = _get_video_duration(str(video))
    if not video_dur or video_dur <= 0:
        video_dur = float(segments[-1]["end"])

    clips = [_build_full_video_clip(video, video_dur, segments, title=title, caption=caption)]
    log("INFO", "Single-video mode skips LLM chunking and uses the entire video as one clip")
    log("INFO", "Generating title, topic, caption, reason, and hook from transcript...")
    clips[0] = generate_single_clip_metadata(
        clips[0],
        filtered,
        llm_model=llm_model,
        api_key=api_key,
    )

    # Select best clip (rank 1)
    best_clip = clips[0] if clips else None
    if not best_clip:
        log("ERROR", "No clip available for single-video processing")
        sys.exit(1)

    log("OK", f"Selected full video clip: rank {best_clip['rank']} (score: {best_clip.get('clip_score', '?')})")
    log("OK", f"  Start: {best_clip['start']:.1f}s, End: {best_clip['end']:.1f}s")
    log("OK", f"  Topic: {best_clip.get('topic', 'N/A')}")
    log("OK", f"  Title: {best_clip.get('title', 'N/A')}")

    # ── 3c. Translate subtitle words to Indonesian ───────────────────────────
    if subtitles:
        from .subtitles import get_clip_words
        raw_words = get_clip_words(
            segments,
            clip_start=best_clip["start"],
            clip_end=best_clip["end"],
        )
        if _should_translate_to_indonesian(detected_language):
            log("INFO", "Translating subtitle words to Indonesian...")
            best_clip["_subtitle_words"] = translate_subtitle_words(
                raw_words,
                llm_model=llm_model,
                api_key=api_key,
            )
            log("OK", f"Subtitle translation complete: {len(best_clip['_subtitle_words'])} words")
        else:
            best_clip["_subtitle_words"] = raw_words
            log("INFO", "Skipping subtitle translation because Whisper detected Indonesian")

    # ── 4. Extract best clip ─────────────────────────────────────────────────
    log("INFO", "Extracting best clip...")
    raw_outputs = extract_clips(
        str(video),
        [best_clip],  # Only best clip
        output_dir=output_path,
        max_workers=1,
    )

    if not raw_outputs:
        log("ERROR", "Failed to extract clip")
        sys.exit(1)

    # ── 5. Post-process with subtitles ──────────────────────────────────
    cta_defaults = get_cta_settings()
    cta_cfg = {**cta_defaults, "enabled": cta if cta is not None else cta_defaults.get("enabled", False)}

    if (subtitles or cta_cfg["enabled"]) and raw_outputs:
        log("INFO", "Adding subtitles overlay...")
        outputs = postprocess_clips(
            raw_outputs,
            [best_clip],
            segments,
            output_dir=output_path,
            subtitles=subtitles,
            subtitle_position=subtitle_position,
            subtitle_margin_pct=subtitle_margin_pct,
            cta_config=cta_cfg,
        )
    else:
        outputs = raw_outputs

    elapsed_total = time.time() - t_total

    # ── 6. Output and reporting ─────────────────────────────────────────
    if outputs:
        print(f"\n{GREEN}{BOLD}✓ Complete!{RESET}")
        print(f"  Output: {outputs[0]}")
        print(f"  Time: {elapsed_total:.0f}s")
        print()
        print(f"{BOLD}Topic:{RESET} {best_clip.get('topic', 'N/A')}")
        print(f"{BOLD}Title:{RESET} {best_clip.get('title', 'N/A')}")
        print(f"{BOLD}Caption:{RESET} {best_clip.get('caption', 'N/A')}")

        # Save clip JSONs (public to disk, internal to .cache)
        # Best clip
        best_clip_public = strip_internal_fields([best_clip])[0]
        best_clip_path = output_path / "best_clip.json"
        best_clip_path.write_text(json.dumps(best_clip_public, indent=2, ensure_ascii=False))
        log("OK", f"Saved best clip → {best_clip_path}")

        # Save internal fields to cache
        internal = get_internal_fields([best_clip])
        if internal:
            cache_dir = get_clips_cache_dir(output_path)
            cache_file = cache_dir / "clips_internal.json"
            cache_file.write_text(json.dumps(internal, indent=2, ensure_ascii=False))

        # Save all clips for reference
        all_clips_public = strip_internal_fields(clips)
        all_clips_path = output_path / "all_clips.json"
        all_clips_path.write_text(json.dumps(all_clips_public, indent=2, ensure_ascii=False))
        log("OK", f"Saved all {len(clips)} clips → {all_clips_path}")

        best_clip_public["video_path"] = str(outputs[0])
        return {
            "topic": best_clip.get("topic", ""),
            "title": best_clip.get("title", ""),
            "video_path": str(outputs[0]),
            "start": best_clip["start"],
            "end": best_clip["end"],
            "clip": best_clip_public,
        }
    else:
        log("ERROR", "No output generated")
        sys.exit(1)


def process_folder(
    folder_path: str,
    output_dir: str = "output",
    **kwargs,
) -> list[dict]:
    """
    Process all video files in a folder using process_single_video.

    Returns:
        list of result dicts from each processed video (same shape as process_single_video)
        Also writes a combined all_clips.json to the output_dir root.
    """
    folder = Path(folder_path)
    if not folder.is_dir():
        log("ERROR", f"Not a directory: {folder}")
        sys.exit(1)

    video_files = sorted(
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_files:
        log("ERROR", f"No video files found in {folder} (looked for {', '.join(VIDEO_EXTENSIONS)})")
        sys.exit(1)

    total = len(video_files)
    print(f"\n{BOLD}{CYAN}{'═' * 50}")
    print(f"   Batch Processor  ({total} video{'s' if total != 1 else ''})")
    print(f"{'═' * 50}{RESET}")
    for i, vf in enumerate(video_files, 1):
        print(f"  [{i}/{total}] {vf.name}")
    print()

    results: list[dict] = []
    all_clips: list[dict] = []
    failed: list[str] = []

    for idx, video_file in enumerate(video_files, 1):
        print(f"\n{BOLD}{YELLOW}── [{idx}/{total}] Processing: {video_file.name} ──{RESET}")
        try:
            result = process_single_video(
                video_path=str(video_file),
                output_dir=output_dir,
                **kwargs,
            )
            results.append(result)
            clip_entry = dict(result.get("clip") or {})
            clip_entry["source_video"] = video_file.name
            all_clips.append(clip_entry)
        except SystemExit:
            log("ERROR", f"Failed to process {video_file.name}, skipping")
            failed.append(video_file.name)

    # Write combined all_clips.json to output root
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    combined_path = out_root / "all_clips.json"
    combined_path.write_text(json.dumps(all_clips, indent=2, ensure_ascii=False))

    print(f"\n{BOLD}{GREEN}{'═' * 50}")
    print(f"   Batch Complete")
    print(f"{'═' * 50}{RESET}")
    print(f"  Processed : {len(results)}/{total}")
    if failed:
        print(f"  Failed    : {', '.join(failed)}")
    print(f"  All clips : {combined_path}")
    print()

    return results


def main() -> None:
    """CLI entry point for process_single.py"""
    defaults = get_defaults()

    ap = argparse.ArgumentParser(
        description="Process a single video (or folder of videos) and extract the best clip with overlay subtitles.",
    )
    ap.add_argument("video", help="Path to input video file or folder (folder = batch mode)")
    ap.add_argument("--model", default=defaults.get("whisper_model", "turbo"),
                    choices=["tiny", "base", "small", "medium",
                             "large-v2", "large-v3", "distil-large-v3", "turbo"],
                    help="Whisper model size (default: from config or turbo)")
    ap.add_argument("--lang", default="id",
                    help="Language code — 'id' Indonesian, 'en' English, "
                         "or None for auto-detect (default: id)")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "cpu"],
                    help="Compute device (default: auto)")
    ap.add_argument("--compute-type", default="auto",
                    choices=["auto", "float16", "int8", "int8_float16"],
                    help="Compute type (default: auto)")
    ap.add_argument("--no-vad", action="store_true",
                    help="Disable VAD filtering for transcription")
    ap.add_argument("--vad-min-silence", type=int, default=400,
                    help="VAD min silence duration in ms (default: 400)")
    ap.add_argument("--vad-speech-pad", type=int, default=200,
                    help="VAD speech padding in ms (default: 200)")
    ap.add_argument("--batch", type=int, default=16,
                    help="Whisper batch size (default: 16; lower if OOM)")
    ap.add_argument("--chunk-duration", type=float, default=360.0,
                    help="LLM chunk duration in seconds (default: 360)")
    ap.add_argument("--chunk-overlap", type=float, default=60.0,
                    help="Overlap between chunks in seconds (default: 60)")
    ap.add_argument("--min-duration", type=int, default=5,
                    help="Minimum clip duration in seconds (default: 5)")
    ap.add_argument("--max-duration", type=int, default=180,
                    help="Maximum clip duration in seconds (default: 180)")
    ap.add_argument("--output", default="output",
                    help="Output directory (default: ./output)")
    ap.add_argument("--api-key", default=None,
                    help="API key (overrides env vars)")
    ap.add_argument("--llm-model", default=None,
                    help="Override LLM model name for OpenRouter")
    ap.add_argument("--subtitles", action=argparse.BooleanOptionalAction,
                    default=defaults.get("subtitles_enabled", True),
                    help="TikTok-style word-by-word subtitles (default: from config or on)")
    ap.add_argument("--subtitle-position",
                    default=defaults.get("subtitle_position", "lower"),
                    choices=["center", "upper", "lower"],
                    help="Subtitle position (default: from config or lower)")
    ap.add_argument("--subtitle-margin", type=float,
                    default=defaults.get("subtitle_margin_pct"),
                    help="Subtitle margin from bottom for 'lower' position in %% (default: from config or 25)")
    ap.add_argument("--title", default=None,
                    help="Manually set the title (overrides auto-generated title; single-file mode only)")
    ap.add_argument("--caption", default=None,
                    help="Manually set the caption (overrides auto-generated caption; single-file mode only)")
    _cta_defaults = get_cta_settings()
    ap.add_argument("--cta", action=argparse.BooleanOptionalAction,
                    default=_cta_defaults.get("enabled", False),
                    help="Append Instagram follow CTA at the end "
                         f"(default: {'on' if _cta_defaults.get('enabled') else 'off'})")

    args = ap.parse_args()
    lang = None if args.lang.lower() == "none" else args.lang

    shared_kwargs = dict(
        model=args.model,
        lang=lang,
        device=args.device,
        compute_type=args.compute_type,
        no_vad=args.no_vad,
        vad_min_silence=args.vad_min_silence,
        vad_speech_pad=args.vad_speech_pad,
        batch=args.batch,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
        chunk_duration=args.chunk_duration,
        chunk_overlap=args.chunk_overlap,
        output_dir=args.output,
        api_key=args.api_key,
        llm_model=args.llm_model,
        subtitles=args.subtitles,
        subtitle_position=args.subtitle_position,
        subtitle_margin_pct=args.subtitle_margin,
        cta=args.cta,
    )

    input_path = Path(args.video)
    if input_path.is_dir():
        process_folder(str(input_path), **shared_kwargs)
    else:
        process_single_video(
            video_path=args.video,
            title=args.title,
            caption=args.caption,
            **shared_kwargs,
        )


if __name__ == "__main__":
    main()
