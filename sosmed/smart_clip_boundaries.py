"""
Smart clip boundary adjustment: optimize for viral-worthy content.

This module intelligently adjusts clip start/end times to:
1. Start at strong hook points (power words, not filler)
2. End at natural sentence boundaries (complete thoughts, punchlines)
3. Validate and correct LLM's hook/closing_line to match actual transcript
4. Score and select the best ending points for maximum retention
"""

from typing import Any


# ── Filler words that make WEAK hooks ──────────────────────────────────────
WEAK_HOOK_WORDS = {
    # Indonesian fillers
    "uh", "um", "eh", "ah", "uhm", "em", "hmm", "mm", "mmm",
    "jadi", "terus", "nah", "ya", "iya", "oke", "ok",
    "gitu", "kayak", "maksudnya", "sebentar", "anu",
    "apa", "sih", "dong", "deh", "nih", "tuh", "lah", "kan", "kok",
    "gini", "gitulah", "lho", "loh", "aduh", "astaga",
    "eung", "euh", "oh", "hah", "wah",
    # English fillers
    "the", "a", "an", "so", "like", "you know", "i mean",
    "well", "actually", "basically", "right", "okay", "alright",
    # Hesitation
    "hello", "halo", "hai", "good morning", "good afternoon",
    "selamat pagi", "selamat siang", "selamat sore",
}

# Power words that make STRONG hooks (start here if possible)
POWER_HOOK_WORDS = {
    # Questions
    "kenapa", "mengapa", "bagaimana", "apa", "siapa", "kapan", "di mana", "kemana", "darimana",
    "how", "what", "why", "when", "who", "where", "which",
    # Bold claims
    "sebenarnya", "faktanya", "kenyataannya", "padahal", "nyatanya",
    "actually", "fact", "truth", "honestly", "believe", "percaya",
    # Numbers/quantifiers
    "satu", "dua", "tiga", "pertama", "kedua", "ketiga",
    "1", "2", "3", "first", "second", "third",
    # Emotional/emphatic
    "yang", "paling", "sangat", "benar", "tidak", "never", "ever",
    "harus", "wajib", "wajib", "penting", "critical", "crucial",
    # Direct address
    "kalian", "kamu", "anda", "lo", "gue", "you", "we", "kita", "kami",
    # Surprising/controversial
    "tapi", "tetapi", "namun", "meskipun", "walaupun",
    "but", "however", "although", "despite",
    # Action words
    "bikin", "buat", "buat", "create", "make", "do", "lakukan",
    "get", "dapatkan", "take", "ambil", "give", "kasih", "beri",
}

# Words that make BAD endings (incomplete thought)
BAD_ENDING_WORDS = {
    "jadi", "terus", "nah", "ya", "iya", "oke", "ok",
    "gitu", "kayak", "maksudnya", "sebentar", "anu",
    "uh", "um", "eh", "ah", "hmm",
    "the", "a", "an", "so", "like", "well",
    # Conjunctions (lead to more content)
    "dan", "atau", "tapi", "tetapi", "namun", "serta",
    "because", "since", "although", "while", "if", "when",
    "karena", "kalau", "jika", "saat", "ketika", "sedangkan",
    # Prepositions ending
    "di", "ke", "dari", "pada", "dalam", "untuk", "dengan",
    "in", "on", "at", "to", "for", "from", "with",
}

# Words that make GOOD endings (conclusive)
GOOD_ENDING_WORDS = {
    # Pronouns (often end complete thoughts)
    "itu", "ini", "dia", "mereka", "kita", "aku", "saya",
    # Concluding phrases
    "begitu", "demikian", "saja", "aja", "sih", "lah",
    # Past/completed
    "sudah", "telah", "udah", "done", "finished", "complete",
    # Emphatic endings
    "banget", "sekali", "very", "really", "totally", "definitely",
    # Strong statements
    "pasti", "yakin", "sure", "certain", "jelas", "clear",
}


def _find_words_in_range(
    segments: list[dict[str, Any]],
    clip_start: float,
    clip_end: float,
    tolerance: float = 0.5,
) -> list[dict[str, Any]]:
    """Extract all words within a time range with some tolerance."""
    words = []
    for seg in segments:
        if seg["end"] < clip_start - tolerance or seg["start"] > clip_end + tolerance:
            continue
        for w in seg.get("words", []):
            w_start = w.get("start", 0)
            w_end = w.get("end", 0)
            w_text = w.get("word", "").strip()
            if not w_text:
                continue
            if w_end > clip_start - tolerance and w_start < clip_end + tolerance:
                words.append({
                    "word": w_text,
                    "start": w_start,
                    "end": w_end,
                    "seg_id": seg.get("id"),
                })
    words.sort(key=lambda x: x["start"])
    return words


def _find_text_in_transcript(
    segments: list[dict[str, Any]],
    search_text: str,
    search_range: tuple[float, float] | None = None,
) -> list[dict[str, Any]] | None:
    """Find a text phrase in the transcript and return matching words."""
    search_normalized = search_text.lower().strip()
    search_words = search_normalized.split()

    if not search_words:
        return None

    if search_range:
        words = _find_words_in_range(segments, search_range[0], search_range[1])
    else:
        words = _find_words_in_range(segments, 0, float("inf"))

    if not words:
        return None

    word_texts = [w["word"].lower().strip(".,!?;:") for w in words]
    search_len = len(search_words)

    for i in range(len(word_texts) - search_len + 1):
        window = word_texts[i:i + search_len]
        if window == search_words:
            return words[i:i + search_len]
        # Fuzzy match
        matches = sum(1 for sw, ww in zip(search_words, window) if sw in ww or ww in sw)
        if matches >= max(1, search_len - 1):
            return words[i:i + search_len]

    return None


def _find_sentence_boundaries(
    words: list[dict[str, Any]],
    min_gap: float = 0.7,
) -> list[int]:
    """Find indices where sentence boundaries likely occur."""
    boundaries = []

    for i in range(len(words) - 1):
        gap = words[i + 1]["start"] - words[i]["end"]
        if gap >= min_gap:
            boundaries.append(i)

    # Punctuation-based boundaries
    sentence_enders = {".", "?", "!", "。", "؟"}
    for i, w in enumerate(words):
        if w["word"].strip() and w["word"].strip()[-1] in sentence_enders:
            if i not in boundaries:
                boundaries.append(i)

    return sorted(set(boundaries))


def _find_strong_hook_position(
    words: list[dict[str, Any]],
    proposed_start: float,
    lookforward_window: float = 4.0,
) -> float:
    """
    Find the strongest hook position - optimized for viral content.

    Priority:
    1. Power words (questions, bold claims, numbers, emotional words)
    2. First substantive content word (skip ALL fillers)
    3. Ensure speech begins within 0.5s
    """
    if not words:
        return proposed_start

    words_in_window = [w for w in words if w["start"] >= proposed_start - 0.2 and w["start"] <= proposed_start + lookforward_window]

    if not words_in_window:
        return proposed_start

    # Skip leading fillers (up to 5 words)
    first_content_idx = 0
    for i in range(min(5, len(words_in_window))):
        word_lower = words_in_window[i]["word"].lower().strip(".,!?;:")
        if word_lower not in WEAK_HOOK_WORDS:
            first_content_idx = i
            break

    # Look for power words in the next 3 positions
    for i in range(first_content_idx, min(first_content_idx + 3, len(words_in_window))):
        word_lower = words_in_window[i]["word"].lower().strip(".,!?;:")
        if word_lower in POWER_HOOK_WORDS:
            # Start slightly before power word for natural lead-in
            return max(proposed_start, words_in_window[i]["start"] - 0.1)

    # Default: start at first non-filler content word
    first_word = words_in_window[first_content_idx]
    return max(proposed_start, first_word["start"] - 0.12)


def _score_ending_quality(
    words: list[dict[str, Any]],
    boundary_idx: int,
) -> int:
    """
    Score how good an ending point is (higher = better for viral retention).

    Scoring factors:
    - Gap after word (natural pause = complete thought)
    - Not ending on filler/conjunction
    - Ending on strong/conclusive words
    - Punctuation ending
    """
    word = words[boundary_idx]
    word_text = word["word"].lower().strip(".,!?;:")
    word_end = word["end"]

    # Gap after this word
    gap_after = 0.0
    if boundary_idx < len(words) - 1:
        gap_after = words[boundary_idx + 1]["start"] - word_end

    score = 0

    # Large gap = strong sentence boundary (+30)
    if gap_after >= 1.2:
        score += 30
    elif gap_after >= 0.8:
        score += 22
    elif gap_after >= 0.5:
        score += 14
    elif gap_after >= 0.3:
        score += 8

    # Not ending on weak word (+25)
    if word_text not in BAD_ENDING_WORDS:
        score += 25

    # Ending on strong/conclusive word (+20)
    if word_text in GOOD_ENDING_WORDS:
        score += 20

    # Punctuation ending (+20)
    if word["word"].strip()[-1] in {".", "?", "!", "。"}:
        score += 20

    # Bonus for question ending (creates curiosity) (+10)
    if word["word"].strip()[-1] == "?":
        score += 10

    # Bonus for exclamation (emotional peak) (+10)
    if word["word"].strip()[-1] == "!":
        score += 10

    return score


def _find_best_ending(
    words: list[dict[str, Any]],
    proposed_end: float,
    lookback_window: float = 6.0,
    min_duration: float = 5.0,
) -> float:
    """
    Find the BEST ending point for viral retention.

    Strategy:
    1. Find all sentence boundaries in the lookback window
    2. Score each boundary for ending quality
    3. Pick the highest-scoring boundary that maintains min duration
    4. Prefer boundaries closer to proposed_end with good scores
    """
    if not words:
        return proposed_end

    end_window_start = max(0, proposed_end - lookback_window)
    words_in_window = [w for w in words if w["start"] >= end_window_start and w["end"] <= proposed_end + 0.5]

    if not words_in_window:
        return proposed_end

    boundaries = _find_sentence_boundaries(words_in_window, min_gap=0.6)

    first_word = words[0] if words else None
    min_end_time = (first_word["start"] if first_word else 0) + min_duration

    if not boundaries:
        # No clear boundaries - find last non-weak word
        for i in range(len(words_in_window) - 1, -1, -1):
            word_text = words_in_window[i]["word"].lower().strip(".,!?;:")
            if word_text not in BAD_ENDING_WORDS:
                return min(proposed_end, words_in_window[i]["end"] + 0.2)
        return min(proposed_end, words_in_window[-1]["end"] + 0.3)

    # Score all valid boundaries
    scored_boundaries = []
    for idx in boundaries:
        boundary_time = words_in_window[idx]["end"]
        if boundary_time >= min_end_time:
            score = _score_ending_quality(words_in_window, idx)
            # Slight penalty for being too far from proposed_end
            distance_penalty = (proposed_end - boundary_time) * 2
            adjusted_score = score - distance_penalty
            scored_boundaries.append((adjusted_score, boundary_time, idx))

    if not scored_boundaries:
        # Fallback to last boundary
        return min(proposed_end, words_in_window[boundaries[-1]]["end"] + 0.2)

    # Pick highest scored boundary
    scored_boundaries.sort(key=lambda x: -x[0])
    best_boundary_idx = scored_boundaries[0][2]

    boundary_word = words_in_window[best_boundary_idx]
    return boundary_word["end"] + 0.2


def _validate_and_fix_hook_closing(
    clip: dict[str, Any],
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Validate hook/closing_line match actual transcript.
    Auto-correct if they don't match exactly.
    """
    clip_start = clip.get("start", 0)
    clip_end = clip.get("end", 0)
    actual_words = _find_words_in_range(segments, clip_start, clip_end, tolerance=1.0)

    if not actual_words:
        return clip

    # Validate hook
    hook = clip.get("hook", "").strip()
    if hook:
        hook_search_end = min(clip_start + 5.0, clip_end)
        hook_words = _find_text_in_transcript(
            segments, hook,
            search_range=(clip_start, hook_search_end)
        )

        if hook_words:
            actual_hook = " ".join(w["word"] for w in hook_words)
            if actual_hook.lower() != hook.lower():
                clip["_hook_original"] = hook
                clip["hook"] = actual_hook
                clip["_hook_start_adjusted"] = clip["start"]
                clip["start"] = max(clip_start, hook_words[0]["start"] - 0.12)

    # Validate closing_line
    closing = clip.get("closing_line", "").strip()
    if closing:
        closing_search_start = max(clip_end - 8.0, clip_start)
        closing_words = _find_text_in_transcript(
            segments, closing,
            search_range=(closing_search_start, clip_end)
        )

        if closing_words:
            actual_closing = " ".join(w["word"] for w in closing_words)
            if actual_closing.lower() != closing.lower():
                clip["_closing_original"] = closing
                clip["closing_line"] = actual_closing
                clip["_closing_end_adjusted"] = clip["end"]
                clip["end"] = min(clip_end, closing_words[-1]["end"] + 0.2)

    return clip


def smart_adjust_clip_boundaries(
    clips: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    *,
    min_duration: float = 5.0,
    max_duration: float = 180.0,
    validate_hook_closing: bool = True,
    aggressive_optimization: bool = True,
) -> list[dict[str, Any]]:
    """
    Intelligently adjust clip boundaries for viral-worthy content.

    This function:
    1. Validates and corrects LLM's hook/closing_line to match actual transcript
    2. Finds power words for strong hooks (questions, bold claims, numbers)
    3. Scores ending points and selects the best for retention
    4. Ensures clips maintain minimum/maximum duration constraints

    Args:
        clips: List of clip dicts with 'start', 'end', 'hook', 'closing_line'
        segments: Transcript segments with word-level timestamps
        min_duration: Minimum clip duration in seconds
        max_duration: Maximum clip duration in seconds
        validate_hook_closing: Whether to validate/correct hook and closing_line
        aggressive_optimization: If True, be more aggressive about finding optimal points

    Returns:
        Updated clips with viral-optimized boundaries
    """
    for clip in clips:
        # Store original LLM boundaries
        if "_llm_start" not in clip:
            clip["_llm_start"] = clip["start"]
            clip["_llm_end"] = clip["end"]

        # Step 1: Validate hook/closing_line
        if validate_hook_closing:
            clip = _validate_and_fix_hook_closing(clip, segments)

        # Step 2: Get words in the clip range
        words = _find_words_in_range(
            segments,
            clip["start"],
            clip["end"],
            tolerance=0.5,
        )

        if not words:
            continue

        # Step 3: Find optimal hook position
        new_start = _find_strong_hook_position(words, clip["start"], lookforward_window=4.0)

        # Step 4: Find optimal ending position
        new_end = _find_best_ending(
            words,
            clip["end"],
            lookback_window=6.0,
            min_duration=min_duration,
        )

        # Step 5: Apply constraints
        if new_end - new_start < min_duration:
            if clip["end"] - new_start >= min_duration:
                new_end = clip["end"]
            elif new_end - clip["start"] >= min_duration:
                new_start = clip["start"]
            else:
                continue

        if new_end - new_start > max_duration:
            new_end = new_start + max_duration

        # Apply adjustments if they improve the clip
        original_dur = clip["end"] - clip["start"]
        new_dur = new_end - new_start

        # Apply if: start moved to skip filler, or end improved for sentence boundary
        if new_start > clip["start"] + 0.2 or abs(new_end - clip["end"]) > 0.4:
            clip["start"] = new_start
            clip["end"] = new_end

        # Log optimization details for debugging
        if new_start != clip.get("_llm_start") or new_end != clip.get("_llm_end"):
            clip["_boundary_optimized"] = True

    return clips


# Backward compatibility
def tighten_clip_boundaries(
    clips: list[dict],
    segments: list[dict],
    padding: float = 0.15,
    max_gap: float = 2.0,
    min_speech_density: float = 0.5,
) -> list[dict]:
    """Legacy function - redirects to smart_adjust_clip_boundaries."""
    return smart_adjust_clip_boundaries(
        clips,
        segments,
        min_duration=5.0,
        max_duration=180.0,
        validate_hook_closing=True,
    )
