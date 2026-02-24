"""
LLM analysis: find engaging clips in transcript.
"""

import math
from typing import Any

from ..utils import log, SYSTEM_PROMPT, MAX_CLIPS_HARD_LIMIT
from .backends import call_llm


def _build_transcript_text(segments: list[dict[str, Any]]) -> str:
    """Ultra-compact transcript: [start-end] text — saves tokens for long videos."""
    lines: list[str] = []
    for s in segments:
        # Use integer seconds when possible to save chars
        st = f"{s['start']:.0f}" if s['start'] == int(s['start']) else f"{s['start']:.1f}"
        en = f"{s['end']:.0f}" if s['end'] == int(s['end']) else f"{s['end']:.1f}"
        lines.append(f"[{st}-{en}]{s['text']}")
    return "\n".join(lines)


def _chunk_segments(
    segments: list[dict[str, Any]],
    chunk_duration: float = 480.0,
    overlap_duration: float = 60.0,
) -> list[list[dict[str, Any]]]:
    """
    Split segments into time-based chunks for iterative LLM processing.

    Each chunk covers ~chunk_duration seconds of transcript with overlap_duration
    overlap to avoid missing clips that span chunk boundaries.
    """
    if not segments:
        return []

    total_start = segments[0]["start"]
    total_end = segments[-1]["end"]
    total_dur = total_end - total_start

    # If total duration fits in one chunk, no need to split
    if total_dur <= chunk_duration * 1.3:
        return [segments]

    chunks: list[list[dict[str, Any]]] = []
    window_start = total_start

    while window_start < total_end:
        window_end = window_start + chunk_duration
        # Gather segments that overlap with this window
        chunk = [
            s for s in segments
            if s["end"] > window_start and s["start"] < window_end + overlap_duration
        ]
        if chunk:
            chunks.append(chunk)
        window_start += chunk_duration

    log("INFO", f"Split {total_dur:.0f}s transcript into {len(chunks)} chunks "
               f"(~{chunk_duration:.0f}s each, {overlap_duration:.0f}s overlap)")
    return chunks


def _build_user_prompt(
    transcript: str,
    min_dur: int,
    max_dur: int,
    max_clips: int,
    min_score: int,
    chunk_info: str = "",
) -> str:
    """Build the user prompt for LLM."""
    header = (
        f"Temukan SEMUA klip menarik dari transkrip ini.\n"
        f"Durasi: {min_dur}-{max_dur}s. Maks {max_clips} klip. Score >= {min_score}.\n"
    )
    if chunk_info:
        header += f"{chunk_info}\n"
    return f"{header}\nTRANSKRIP:\n{transcript}"


def _validate_clips(
    clips: list[dict[str, Any]],
    min_dur: int,
    max_dur: int,
    max_clips: int,
    min_score: int,
    video_duration: float | None = None,
) -> list[dict[str, Any]]:
    """Sanitize, deduplicate, and cap the clip list."""
    valid: list[dict[str, Any]] = []
    seen_ranges: list[tuple[float, float]] = []

    for c in clips:
        try:
            s, e = float(c["start"]), float(c["end"])
        except (KeyError, ValueError, TypeError):
            continue
        dur = e - s
        if dur < min_dur * 0.5 or dur > max_dur * 1.5:
            continue  # allow 50% tolerance to keep more clips
        if video_duration and e > video_duration + 2:
            continue
        score = int(c.get("engagement_score", 0) or 0)
        if score < min_score * 0.8:  # softer threshold
            continue
        # Overlap check — allow up to 20% overlap instead of zero
        def _overlap_ratio(s1: float, e1: float, s2: float, e2: float) -> float:
            overlap = max(0, min(e1, e2) - max(s1, s2))
            shorter = min(e1 - s1, e2 - s2)
            return overlap / shorter if shorter > 0 else 0

        overlaps = any(_overlap_ratio(s, e, rs, re) > 0.2 for rs, re in seen_ranges)
        if overlaps:
            continue
        seen_ranges.append((s, e))
        # Ensure required fields
        c.setdefault("rank", len(valid) + 1)
        c.setdefault("title", f"Clip {c['rank']}")
        c.setdefault("engagement_score", score)
        c.setdefault("reason", "")
        c.setdefault("hook", "")
        valid.append(c)
        if len(valid) >= max_clips:
            break

    # Re-rank by score
    valid.sort(key=lambda x: -x.get("engagement_score", 0))
    for i, c in enumerate(valid, 1):
        c["rank"] = i

    return valid


def _merge_chunk_clips(
    all_clips: list[dict[str, Any]],
    min_dur: int,
    max_dur: int,
    max_clips: int,
    min_score: int,
    video_duration: float | None = None,
) -> list[dict[str, Any]]:
    """
    Merge clips from multiple LLM chunks, removing near-duplicates.

    Two clips are considered duplicates if they overlap by > 50%.
    When duplicates are found, keep the one with the higher score.
    """
    # Sort all clips by start time
    all_clips.sort(key=lambda c: (float(c.get("start", 0)), -int(c.get("engagement_score", 0))))

    deduped: list[dict[str, Any]] = []
    for clip in all_clips:
        try:
            s, e = float(clip["start"]), float(clip["end"])
        except (KeyError, ValueError, TypeError):
            continue

        # Check for near-duplicate with already accepted clips
        is_dup = False
        for i, existing in enumerate(deduped):
            es, ee = float(existing["start"]), float(existing["end"])
            overlap = max(0, min(e, ee) - max(s, es))
            shorter = min(e - s, ee - es)
            if shorter > 0 and overlap / shorter > 0.5:
                # Keep the higher-scoring one
                if int(clip.get("engagement_score", 0)) > int(existing.get("engagement_score", 0)):
                    deduped[i] = clip
                is_dup = True
                break
        if not is_dup:
            deduped.append(clip)

    log("INFO", f"Merged {len(all_clips)} raw clips → {len(deduped)} after dedup")

    # Now validate the deduped list
    return _validate_clips(deduped, min_dur, max_dur, max_clips, min_score, video_duration)


def find_clips(
    segments: list[dict[str, Any]],
    *,
    min_duration: int = 15,
    max_duration: int = 60,
    max_clips: int = MAX_CLIPS_HARD_LIMIT,
    min_score: int = 60,
    llm_model: str | None = None,
    api_key: str | None = None,
    video_duration: float | None = None,
    chunk_duration: float = 480.0,
    chunk_overlap: float = 60.0,
) -> list[dict[str, Any]]:
    """
    Ask LLM to find ALL engaging clips (up to *max_clips*).

    For long videos (> ~8 min of transcript), splits transcript into
    overlapping chunks and calls the LLM iteratively on each chunk,
    then merges and deduplicates results across all chunks.
    """
    # Split into chunks for iterative processing
    chunks = _chunk_segments(segments, chunk_duration, chunk_overlap)

    system = SYSTEM_PROMPT.format(
        min_dur=min_duration,
        max_dur=max_duration,
        max_clips=max_clips,
        min_score=min_score,
    )

    all_raw_clips: list[dict[str, Any]] = []
    n_chunks = len(chunks)

    for idx, chunk in enumerate(chunks, 1):
        chunk_start = chunk[0]["start"]
        chunk_end = chunk[-1]["end"]
        # Scale clips per chunk: for long videos ensure at least 8-15 per chunk
        clips_per_chunk = max(15, math.ceil(max_clips / max(n_chunks, 1) * 2.0))
        clips_per_chunk = min(clips_per_chunk, 40)  # cap per-chunk to avoid degraded output

        transcript = _build_transcript_text(chunk)
        chunk_info = ""
        if n_chunks > 1:
            chunk_info = (
                f"Ini bagian {idx}/{n_chunks} video "
                f"({chunk_start:.0f}s-{chunk_end:.0f}s). "
                f"Cari sebanyak mungkin klip menarik di bagian ini."
            )
            log("INFO", f"Chunk {idx}/{n_chunks}: {chunk_start:.0f}s → {chunk_end:.0f}s "
                       f"({len(chunk)} segments, asking for ≤{clips_per_chunk} clips)")

        user = _build_user_prompt(
            transcript, min_duration, max_duration,
            clips_per_chunk, min_score, chunk_info,
        )

        raw_clips = call_llm(system, user, api_key, llm_model)
        log("OK", f"Chunk {idx}/{n_chunks}: LLM returned {len(raw_clips)} clips")
        all_raw_clips.extend(raw_clips)

    log("INFO", f"Total raw clips from {n_chunks} chunk(s): {len(all_raw_clips)}")

    # Merge, deduplicate, and validate across all chunks
    if n_chunks > 1:
        clips = _merge_chunk_clips(
            all_raw_clips, min_duration, max_duration,
            max_clips, min_score, video_duration,
        )
    else:
        clips = _validate_clips(
            all_raw_clips, min_duration, max_duration,
            max_clips, min_score, video_duration,
        )

    if not clips:
        log("WARN", "LLM returned 0 valid clips. The video may not have engaging segments.")
    else:
        log("OK", f"{len(clips)} valid clips after validation")
    return clips
