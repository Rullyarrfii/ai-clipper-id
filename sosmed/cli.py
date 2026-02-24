"""
CLI interface: argument parsing and orchestration.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from .transcription import transcribe
from .prefilter import prefilter_segments
from .llm import find_clips
from .extraction import extract_clips, _get_video_duration
from .utils import (
    log, BOLD, RESET, CYAN, GREEN, YELLOW,
    MAX_CLIPS_HARD_LIMIT
)


def _get_transcript_cache_path(video_path: str) -> Path:
    """Return cache path for transcript based on video filename."""
    video = Path(video_path)
    cache_dir = Path.cwd() / ".cache" / "ai-video-clipper"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{video.stem}_transcript.json"
    return cache_file


def main() -> None:
    """Main CLI entry point."""
    ap = argparse.ArgumentParser(
        description="AI Video Clipper — Indonesian-optimized, auto clip count",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment variables:\n"
            "  OPENROUTER_API_KEY   OpenRouter key (default backend, free model)\n"
            "  OPENROUTER_MODEL     Override default model on OpenRouter\n"
            "  ANTHROPIC_API_KEY    Anthropic Claude key\n"
            "  OPENAI_API_KEY       OpenAI key\n"
            "  OLLAMA_MODEL         Local Ollama model name (default: llama3.1)\n"
        ),
    )
    ap.add_argument("video", help="Path to input video")
    ap.add_argument("--model", default="medium",
                    choices=["tiny", "base", "small", "medium",
                             "large-v2", "large-v3", "distil-large-v3", "turbo"],
                    help="Whisper model size (default: tiny)")
    ap.add_argument("--lang", default="id",
                    help="Language code — 'id' Indonesian, 'en' English, "
                         "or None for auto-detect (default: id)")
    ap.add_argument("--min", type=int, default=15,
                    help="Min clip duration in seconds (default: 15)")
    ap.add_argument("--max", type=int, default=60,
                    help="Max clip duration in seconds (default: 60)")
    ap.add_argument("--max-clips", type=int, default=MAX_CLIPS_HARD_LIMIT,
                    help=f"Maximum number of clips (default: {MAX_CLIPS_HARD_LIMIT})")
    ap.add_argument("--min-score", type=int, default=60,
                    help="Minimum engagement score to keep a clip (default: 60)")
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
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel ffmpeg workers (default: 4)")
    ap.add_argument("--chunk-duration", type=float, default=480.0,
                    help="LLM chunk duration in seconds (default: 480)")
    ap.add_argument("--chunk-overlap", type=float, default=60.0,
                    help="Overlap between chunks in seconds (default: 60)")
    ap.add_argument("--output", default="clips",
                    help="Output directory (default: ./clips)")
    ap.add_argument("--save-transcript", action="store_true",
                    help="Save full transcript JSON")
    ap.add_argument("--api-key", default=None,
                    help="API key (overrides env vars)")
    ap.add_argument("--llm-model", default=None,
                    help="Override LLM model name for OpenRouter")

    args = ap.parse_args()
    lang = None if args.lang.lower() == "none" else args.lang

    video = Path(args.video)
    if not video.exists():
        log("ERROR", f"File not found: {video}")
        sys.exit(1)

    output_dir = Path(args.output)

    print(f"\n{BOLD}{CYAN}{'═' * 50}")
    print(f"   AI Video Clipper — Indonesian-optimized")
    print(f"{'═' * 50}{RESET}")
    print(f"  Video     : {video.name}")
    print(f"  Model     : {args.model}")
    print(f"  Language  : {lang or 'auto-detect'}")
    print(f"  Duration  : {args.min}–{args.max}s per clip")
    print(f"  Max clips : {args.max_clips} (LLM decides actual count)")
    print()

    # ── 1. Transcribe ────────────────────────────────────────────────────────
    t_total = time.time()

    # Check for cached transcript
    cache_path = _get_transcript_cache_path(str(video))
    if cache_path.exists():
        log("INFO", f"Loading cached transcript from {cache_path}")
        segments = json.loads(cache_path.read_text())
        log("OK", f"Loaded {len(segments)} segments from cache")
    else:
        segments = transcribe(
            str(video),
            model_size=args.model,
            language=lang,
            device=args.device,
            compute_type=args.compute_type,
            vad_filter=not args.no_vad,
            vad_min_silence_ms=args.vad_min_silence,
            vad_speech_pad_ms=args.vad_speech_pad,
            batch_size=args.batch,
        )
        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(segments, indent=2, ensure_ascii=False))
        log("OK", f"Transcript cached → {cache_path}")

    if args.save_transcript:
        output_dir.mkdir(parents=True, exist_ok=True)
        tx = output_dir / "transcript.json"
        tx.write_text(json.dumps(segments, indent=2, ensure_ascii=False))
        log("OK", f"Transcript → {tx}")

    # ── 2. Pre-filter ────────────────────────────────────────────────────────
    filtered, stats = prefilter_segments(segments)

    print(f"\n{BOLD}Pre-filter:{RESET}")
    print(f"  {stats['original']} → {stats['kept']} segments "
          f"({stats['dropped']} dropped, {stats['drop_pct']})")
    if stats["reasons"]:
        print(f"  Reasons: {', '.join(f'{k}={v}' for k, v in stats['reasons'].items())}")
    print()

    if not filtered:
        log("ERROR", "All segments filtered out. Try a different whisper model or looser filters.")
        sys.exit(1)

    # ── 3. LLM analysis ─────────────────────────────────────────────────────
    video_dur = _get_video_duration(str(video))
    clips = find_clips(
        filtered,
        min_duration=args.min,
        max_duration=args.max,
        max_clips=min(args.max_clips, MAX_CLIPS_HARD_LIMIT),
        min_score=args.min_score,
        llm_model=args.llm_model,
        api_key=args.api_key,
        video_duration=video_dur,
        chunk_duration=args.chunk_duration,
        chunk_overlap=args.chunk_overlap,
    )

    if not clips:
        log("WARN", "No engaging clips found. Exiting.")
        sys.exit(0)

    # Summary table
    print(f"\n{BOLD}{'#':<4} {'Score':<6} {'Start':>7} {'End':>7} {'Dur':>5}  Title{RESET}")
    print("─" * 70)
    for c in clips:
        d = c["end"] - c["start"]
        print(f"  {c['rank']:<3} {c.get('engagement_score', '?'):<6} "
              f"{c['start']:>7.1f} {c['end']:>7.1f} {d:>4.0f}s  {c['title']}")
    print()

    # ── 4. Extract ───────────────────────────────────────────────────────────
    outputs = extract_clips(
        str(video),
        clips,
        output_dir=output_dir,
        max_workers=args.workers,
    )

    # Save metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = output_dir / "clips.json"
    meta.write_text(json.dumps(clips, indent=2, ensure_ascii=False))

    elapsed_total = time.time() - t_total
    print(f"\n{GREEN}{BOLD}✓ Done!{RESET} "
          f"{len(outputs)}/{len(clips)} clips extracted → {output_dir}/ "
          f"({elapsed_total:.0f}s total)")
    print(f"  Metadata → {meta}")
