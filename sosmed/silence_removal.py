"""
Silence removal for video clips.

Detects and removes silent gaps from clips to improve pacing
and retention. Uses word-level timestamps from Whisper to identify
segments with no speech or interesting audio.
"""

from typing import Any

from .utils import log


def find_speech_regions(
    words: list[dict[str, Any]],
    clip_duration: float,
    max_silence: float = 1.5,
    padding: float = 0.2,
) -> list[tuple[float, float]]:
    """Find regions of the clip that contain speech.

    Groups words into continuous speech regions, merging gaps
    smaller than max_silence. Returns time ranges to keep.

    Args:
        words: Word-level timestamps (0-based relative to clip start)
        clip_duration: Total clip duration in seconds
        max_silence: Maximum allowed silence gap between words (seconds)
        padding: Padding to add before/after speech regions

    Returns:
        List of (start, end) tuples representing speech regions to keep
    """
    if not words:
        return [(0.0, clip_duration)]

    # Sort words by start time
    sorted_words = sorted(words, key=lambda w: w["start"])

    # Build speech regions by merging words within max_silence gap
    regions: list[tuple[float, float]] = []
    region_start = max(0.0, sorted_words[0]["start"] - padding)
    region_end = sorted_words[0]["end"]

    for word in sorted_words[1:]:
        gap = word["start"] - region_end
        if gap <= max_silence:
            # Extend current region
            region_end = word["end"]
        else:
            # Close current region, start new one
            regions.append((region_start, min(region_end + padding, clip_duration)))
            region_start = max(0.0, word["start"] - padding)
            region_end = word["end"]

    # Close final region
    regions.append((region_start, min(region_end + padding, clip_duration)))

    return regions


def compute_silence_removal(
    words: list[dict[str, Any]],
    clip_duration: float,
    max_silence: float = 1.5,
    min_kept_duration: float = 5.0,
    padding: float = 0.2,
) -> list[tuple[float, float]] | None:
    """Compute time ranges to keep after removing silence.

    Only removes silence if the result is still long enough
    and actually saves meaningful time.

    Args:
        words: Word-level timestamps (0-based)
        clip_duration: Total clip duration
        max_silence: Maximum silence gap to allow
        min_kept_duration: Minimum duration after removal
        padding: Time padding around speech

    Returns:
        List of (start, end) keep-regions, or None if no removal needed
    """
    if not words or clip_duration <= 0:
        return None

    regions = find_speech_regions(words, clip_duration, max_silence, padding)

    # Calculate total kept duration
    kept_duration = sum(end - start for start, end in regions)

    # Only remove silence if:
    # 1. We actually save time (>2s saved)
    # 2. The result is still long enough
    # 3. There are actual silent gaps to remove
    time_saved = clip_duration - kept_duration
    if time_saved < 2.0 or kept_duration < min_kept_duration or len(regions) <= 1:
        return None

    log("DEBUG", f"Silence removal: {clip_duration:.1f}s → {kept_duration:.1f}s "
                 f"(saved {time_saved:.1f}s, {len(regions)} regions)")
    return regions


def build_silence_removal_filter(
    keep_regions: list[tuple[float, float]],
) -> str:
    """Build FFmpeg filter to concatenate speech regions (remove silence).

    Uses the select/aselect filters to keep only speech regions
    and concat to join them.

    Args:
        keep_regions: List of (start, end) time ranges to keep

    Returns:
        FFmpeg filter_complex string
    """
    if not keep_regions:
        return ""

    # Build select expression: keep frames within any region
    conditions = []
    for start, end in keep_regions:
        conditions.append(f"between(t\\,{start:.3f}\\,{end:.3f})")

    select_expr = "+".join(conditions)

    # Video and audio select filters
    vfilter = f"select='{select_expr}',setpts=N/FRAME_RATE/TB"
    afilter = f"aselect='{select_expr}',asetpts=N/SR/TB"

    return vfilter, afilter


def adjust_subtitle_times(
    words: list[dict[str, Any]],
    keep_regions: list[tuple[float, float]],
) -> list[dict[str, Any]]:
    """Adjust word timestamps to account for removed silence.

    When silence is removed, the remaining segments are concatenated.
    This function remaps word timestamps to their new positions
    in the shortened clip.

    Args:
        words: Original word timestamps
        keep_regions: Regions that were kept (from compute_silence_removal)

    Returns:
        Words with adjusted timestamps
    """
    if not keep_regions or not words:
        return words

    # Build a mapping: for each keep region, compute its new start time
    cumulative_offset = 0.0
    region_offsets: list[tuple[float, float, float]] = []  # (orig_start, orig_end, new_start)

    for start, end in keep_regions:
        region_offsets.append((start, end, cumulative_offset))
        cumulative_offset += (end - start)

    adjusted: list[dict[str, Any]] = []
    for word in words:
        w_mid = (word["start"] + word["end"]) / 2.0

        # Find which region this word belongs to
        for orig_start, orig_end, new_start in region_offsets:
            if orig_start <= w_mid <= orig_end:
                offset = new_start - orig_start
                adjusted.append({
                    "word": word["word"],
                    "start": max(0.0, word["start"] + offset),
                    "end": max(0.0, word["end"] + offset),
                })
                break

    return adjusted
