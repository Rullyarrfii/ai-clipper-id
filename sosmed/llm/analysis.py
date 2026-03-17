"""
LLM analysis: find engaging clips in transcript.
"""

import math
import re
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
        f"Analyze this transcript and extract every clip with a realistic shot at high views.\n"
        f"Duration: {min_dur}–{max_dur}s. Max {max_clips} clips. clip_score ≥ {min_score}.\n"
    )
    if chunk_info:
        header += f"{chunk_info}\n"
    return f"{header}\nTRANSKRIP:\n{transcript}"


def _compute_clip_score(c: dict[str, Any]) -> int:
    """Compute clip_score as weighted average tuned for TikTok virality.

    Formula:
      hook*0.30 + shareability*0.25 + entertainment*0.25 + retention*0.15 + clarity*0.05
    """
    hook = int(c.get("score_hook", 0) or 0)
    retention = int(c.get("score_retention", 0) or 0)
    shareability = int(c.get("score_shareability", 0) or 0)
    entertainment = int(c.get("score_entertainment", 0) or 0)
    clarity = int(c.get("score_clarity", 0) or 0)

    # Also check legacy field names for backward compatibility
    if hook + retention + shareability + entertainment + clarity == 0:
        hook = int(c.get("score_energy", 0) or 0)
        retention = int(c.get("score_informative", 0) or 0)
        shareability = int(c.get("score_newsworthy", 0) or 0)
        clarity = int(c.get("score_easy", 0) or 0)
        entertainment = hook  # approximate

    total = hook + retention + shareability + entertainment + clarity
    if total > 0:
        return round(
            retention * 0.35
            + clarity * 0.25
            + hook * 0.20
            + shareability * 0.15
            + entertainment * 0.05
        )
    return int(c.get("clip_score", 0) or c.get("engagement_score", 0) or 0)


# Titles/topics that indicate non-viral content (intros, outros, etc.)
# NOTE: Q&A / tanya jawab is NOT excluded — often contains great insights
_LOW_VALUE_PATTERNS = re.compile(
    r"(?i)"
    r"(?:selamat datang|pembuka|opening|penutup|closing|terima kasih|"
    r"thank you|perkenalan|introduction|polling|vote|subscribe|"
    r"topik yang akan|webinar akan|rekaman|pendekatan materi|"
    r"ajak partisipasi|sapa penonton|ucapan|disclaimer|house\s*keeping)"
)


def _is_low_value_clip(c: dict[str, Any]) -> bool:
    """Check if clip is likely low-value (intro, outro, generic Q&A, etc.)."""
    title = (c.get("title", "") or "").strip()
    topic = (c.get("topic", "") or "").strip()
    combined = f"{title} {topic}"
    return bool(_LOW_VALUE_PATTERNS.search(combined))


def _validate_clips(
    clips: list[dict[str, Any]],
    min_dur: int,
    max_dur: int,
    max_clips: int,
    min_score: int,
    video_duration: float | None = None,
) -> list[dict[str, Any]]:
    """Sanitize, deduplicate, and cap the clip list — strict quality gate for viral hits only."""
    valid: list[dict[str, Any]] = []
    seen_ranges: list[tuple[float, float]] = []

    # Sort by score descending first so higher-quality clips get priority
    scored_clips = []
    for c in clips:
        try:
            s, e = float(c["start"]), float(c["end"])
        except (KeyError, ValueError, TypeError):
            continue
        score = _compute_clip_score(c)
        # Penalize low-value clips (intros, outros, disclaimers)
        if _is_low_value_clip(c):
            score = max(0, score - 25)
        c["_score"] = score
        scored_clips.append(c)
    # Tiebreaker: score_hook (stop-scrolling power is #1 virality predictor)
    scored_clips.sort(key=lambda x: (-x["_score"], -int(x.get("score_hook", 0) or 0)))

    for c in scored_clips:
        s, e = float(c["start"]), float(c["end"])
        dur = e - s
        if dur < min_dur or dur > max_dur:
            continue  # strict duration enforcement — no tolerance
        if video_duration and e > video_duration + 2:
            continue
        score = c["_score"]
        if score < min_score:
            continue  # strict score threshold — no leniency
        # Lenient hook requirement (educational content may not have a massive hook)
        hook_score = int(c.get("score_hook", 0) or 0)
        if hook_score < 60:
            continue  # lower hook threshold for education
        # Lenient overlap check
        def _overlap_ratio(s1: float, e1: float, s2: float, e2: float) -> float:
            overlap = max(0, min(e1, e2) - max(s1, s2))
            shorter = min(e1 - s1, e2 - s2)
            return overlap / shorter if shorter > 0 else 0

        overlaps = any(_overlap_ratio(s, e, rs, re) > 0.5 for rs, re in seen_ranges)
        if overlaps:
            continue  # stricter overlap detection — no near-duplicates
        seen_ranges.append((s, e))
        # Ensure required fields
        c.setdefault("rank", len(valid) + 1)
        c.setdefault("title", f"Clip {c['rank']}")
        c.setdefault("topic", "")
        c.setdefault("caption", "")
        c.setdefault("reason", "")
        c.setdefault("hook", "")
        c.setdefault("score_hook", 0)
        c.setdefault("score_retention", 0)
        c.setdefault("score_shareability", 0)
        c.setdefault("score_educational", 0)
        c.setdefault("score_clarity", 0)
        c["clip_score"] = score
        c.pop("_score", None)
        c.pop("engagement_score", None)
        # Remove legacy score fields if present
        for legacy in ("score_easy", "score_informative", "score_energy", "score_newsworthy"):
            c.pop(legacy, None)
        valid.append(c)
        if len(valid) >= max_clips:
            break

    # Re-rank by clip_score, tiebreak by hook strength (strongest virality signal)
    valid.sort(key=lambda x: (-x.get("clip_score", 0), -int(x.get("score_hook", 0) or 0)))
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

    Two clips are considered duplicates if they overlap by > 30%.
    When duplicates are found, keep the one with the higher score.
    """
    # Sort all clips by start time
    all_clips.sort(key=lambda c: (float(c.get("start", 0)), -_compute_clip_score(c)))

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
            if shorter > 0 and overlap / shorter > 0.7:
                # Keep the higher-scoring one
                if _compute_clip_score(clip) > _compute_clip_score(existing):
                    deduped[i] = clip
                is_dup = True
                break
        if not is_dup:
            deduped.append(clip)

    log("INFO", f"Merged {len(all_clips)} raw clips → {len(deduped)} after dedup")

    # Now validate the deduped list
    return _validate_clips(deduped, min_dur, max_dur, max_clips, min_score, video_duration)


def _find_gaps(
    clips: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    min_gap: float = 30.0,
) -> list[tuple[float, float]]:
    """
    Find time ranges in the transcript not covered by any clip.

    Returns list of (start, end) tuples for gaps >= min_gap seconds.
    """
    if not segments:
        return []

    total_start = segments[0]["start"]
    total_end = segments[-1]["end"]

    # Sort clips by start time
    sorted_clips = sorted(clips, key=lambda c: float(c.get("start", 0)))

    gaps: list[tuple[float, float]] = []
    cursor = total_start

    for c in sorted_clips:
        cs, ce = float(c["start"]), float(c["end"])
        if cs > cursor + min_gap:
            gaps.append((cursor, cs))
        cursor = max(cursor, ce)

    # Check trailing gap
    if total_end > cursor + min_gap:
        gaps.append((cursor, total_end))

    return gaps


def _segments_in_range(
    segments: list[dict[str, Any]],
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    """Return segments that overlap with the given time range."""
    return [s for s in segments if s["end"] > start and s["start"] < end]


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

    After the first pass, identifies large gaps (uncovered time ranges)
    and does a second LLM pass to extract additional clips from those gaps.
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
        # Generous per-chunk budget — maximize output
        clips_per_chunk = max(10, math.ceil(max_clips / max(n_chunks, 1) * 1.5))
        clips_per_chunk = min(clips_per_chunk, max_clips)

        transcript = _build_transcript_text(chunk)
        chunk_info = ""
        if n_chunks > 1:
            chunk_info = (
                f"Ini bagian {idx}/{n_chunks} video "
                f"({chunk_start:.0f}s-{chunk_end:.0f}s). "
                f"Ekstrak SEMUA momen menarik di bagian ini — setiap subtopik yang layak harus jadi klip."
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
