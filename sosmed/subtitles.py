"""
TikTok-style subtitle generation (ASS format).

Generates word-by-word highlighted subtitles with karaoke fill effect.
Big bold text, white → yellow highlight, black outline.
"""

from typing import Any


# ── ASS color helpers (format: &HAABBGGRR) ──────────────────────────────────

def _rgb_to_ass(r: int, g: int, b: int, a: int = 0) -> str:
    """Convert RGB(A) to ASS color string &HAABBGGRR."""
    return f"&H{a:02X}{b:02X}{g:02X}{r:02X}"


# Default TikTok-style colors
COLOR_HIGHLIGHT = _rgb_to_ass(255, 225, 53)     # Yellow #FFE135 (spoken word)
COLOR_NORMAL    = _rgb_to_ass(255, 255, 255)     # White (upcoming words)
COLOR_OUTLINE   = _rgb_to_ass(0, 0, 0)           # Black outline
COLOR_SHADOW    = _rgb_to_ass(0, 0, 0, 128)      # Semi-transparent shadow


def _seconds_to_ass_time(seconds: float) -> str:
    """Convert seconds to ASS time format H:MM:SS.cc (centiseconds)."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _group_words(
    words: list[dict[str, Any]],
    max_words: int = 4,
    max_duration: float = 2.5,
    max_gap: float = 0.8,
) -> list[list[dict[str, Any]]]:
    """Group words into subtitle chunks.

    Groups are split when:
    - max_words reached
    - max_duration exceeded
    - gap between words > max_gap
    """
    if not words:
        return []

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []

    for word in words:
        if current:
            prev_end = current[-1]["end"]
            curr_start = word["start"]
            group_dur = word["end"] - current[0]["start"]
            gap = curr_start - prev_end

            if (len(current) >= max_words
                    or group_dur > max_duration
                    or gap > max_gap):
                groups.append(current)
                current = []

        current.append(word)

    if current:
        groups.append(current)

    return groups


def generate_ass_subtitles(
    words: list[dict[str, Any]],
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    font_name: str = "Arial",
    font_size: int = 58,
    highlight_color: str | None = None,
    normal_color: str | None = None,
    outline_width: int = 4,
    shadow_depth: int = 1,
    position: str = "center",
    max_words_per_group: int = 4,
) -> str:
    """Generate an ASS subtitle string with TikTok-style karaoke highlighting.

    Args:
        words: List of {"word": str, "start": float, "end": float} dicts.
                Timestamps should be relative to clip start (0-based).
        play_res_x: Subtitle canvas width (match output video).
        play_res_y: Subtitle canvas height (match output video).
        font_name: Font family name.
        font_size: Font size in ASS units.
        highlight_color: ASS color for highlighted word (default: yellow).
        normal_color: ASS color for normal text (default: white).
        outline_width: Text outline thickness.
        shadow_depth: Shadow distance.
        position: "center", "upper", or "lower".
        max_words_per_group: Max words per subtitle line.

    Returns:
        Complete ASS subtitle file as a string.
    """
    hi_color = highlight_color or COLOR_HIGHLIGHT
    nm_color = normal_color or COLOR_NORMAL

    # Alignment based on position
    alignment_map = {"upper": 8, "center": 5, "lower": 2}
    alignment = alignment_map.get(position, 5)

    # MarginV for positioning
    margin_v_map = {"upper": 450, "center": 0, "lower": 120}
    margin_v = margin_v_map.get(position, 0)

    # ASS header
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TikTok,{font_name},{font_size},{hi_color},{nm_color},{COLOR_OUTLINE},{COLOR_SHADOW},-1,0,0,0,100,100,2,0,1,{outline_width},{shadow_depth},{alignment},40,40,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    # Group words into subtitle chunks
    groups = _group_words(words, max_words=max_words_per_group)

    dialogue_lines: list[str] = []
    for group in groups:
        if not group:
            continue

        group_start = group[0]["start"]
        group_end = group[-1]["end"]

        # Add small padding at the end for readability
        group_end_padded = group_end + 0.15

        start_str = _seconds_to_ass_time(max(0, group_start))
        end_str = _seconds_to_ass_time(group_end_padded)

        # Build karaoke text with \kf tags
        # \kf = karaoke fill (smooth left→right sweep)
        # Duration in centiseconds
        parts: list[str] = []
        for i, word in enumerate(group):
            word_start = word["start"]
            word_end = word["end"]

            # For the first word, include any gap from group_start
            if i == 0:
                effective_start = group_start
            else:
                # Include gap before this word (assign to this word's fill)
                effective_start = group[i - 1]["end"]

            dur_cs = max(1, round((word_end - effective_start) * 100))
            clean_word = word["word"].strip()
            if clean_word:
                parts.append(f"{{\\kf{dur_cs}}}{clean_word}")

        text = " ".join(parts)

        # Add subtle pop-in animation: scale from 105% to 100% over 100ms
        text = f"{{\\fscx105\\fscy105\\t(0,100,\\fscx100\\fscy100)}}{text}"

        line = f"Dialogue: 0,{start_str},{end_str},TikTok,,0,0,0,,{text}"
        dialogue_lines.append(line)

    return header + "\n".join(dialogue_lines) + "\n"


def get_clip_words(
    segments: list[dict[str, Any]],
    clip_start: float,
    clip_end: float,
) -> list[dict[str, Any]]:
    """Extract word-level timestamps for a clip's time range.

    Returns words with timestamps adjusted to be relative to clip start (0-based).
    """
    words: list[dict[str, Any]] = []

    for seg in segments:
        if seg["end"] < clip_start or seg["start"] > clip_end:
            continue

        for w in seg.get("words", []):
            w_start = w.get("start", 0)
            w_end = w.get("end", 0)
            w_text = w.get("word", "").strip()

            # Only include words fully within clip boundaries
            if w_start >= clip_start - 0.1 and w_end <= clip_end + 0.1 and w_text:
                words.append({
                    "word": w_text,
                    "start": max(0, w_start - clip_start),
                    "end": max(0, w_end - clip_start),
                })

    return words
