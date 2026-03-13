"""
Subtitle generation (ASS format) — clean, non-overlapping, lower-third.

Generates word-by-word highlighted subtitles with karaoke fill effect.
Bold white text with yellow highlight sweep, black outline, positioned
at the bottom of the frame with no temporal overlap between lines.
"""

from typing import Any


# ── ASS color helpers (format: &HAABBGGRR) ──────────────────────────────────

def _rgb_to_ass(r: int, g: int, b: int, a: int = 0) -> str:
    """Convert RGB(A) to ASS color string &HAABBGGRR."""
    return f"&H{a:02X}{b:02X}{g:02X}{r:02X}"


COLOR_HIGHLIGHT = _rgb_to_ass(255, 225, 53)      # Yellow (spoken word)
COLOR_NORMAL    = _rgb_to_ass(255, 255, 255)      # White (upcoming words)
COLOR_OUTLINE   = _rgb_to_ass(0, 0, 0)            # Black outline
COLOR_SHADOW    = _rgb_to_ass(0, 0, 0, 80)        # Subtle shadow


def _seconds_to_ass_time(seconds: float) -> str:
    """Convert seconds to ASS time format H:MM:SS.cc (centiseconds)."""
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def _group_words(
    words: list[dict[str, Any]],
    max_words: int = 5,
    max_duration: float = 3.0,
    max_gap: float = 0.6,
) -> list[list[dict[str, Any]]]:
    """Group words into subtitle chunks.

    Splits when:
    - max_words reached
    - cumulative duration exceeds max_duration
    - silence gap between consecutive words exceeds max_gap
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


def _adapt_for_aspect_ratio(
    play_res_x: int,
    play_res_y: int,
    font_size_pct: float,
) -> float:
    """Increase font percentage for landscape / square video.

    When a landscape (16:9) clip is shown on a vertical phone, the video
    occupies roughly 56% of screen width-equivalent height.  To keep
    subtitles readable we scale the percentage up so the *perceived* size
    on a phone stays roughly consistent.

    Resulting perceived sizes (base 3.5%, 1080-wide phone):
      Portrait  1080×1920  → 67 px in video, ~67 px on phone
      Square    1080×1080  → 42 px in video, ~42 px on phone (min-clamp)
      Landscape 1920×1080  → 76 px in video, ~43 px on phone
      Landscape 1280×720   → 51 px in video, ~43 px on phone
    """
    aspect = play_res_x / max(1, play_res_y)
    if aspect <= 1.0:
        return font_size_pct
    # At 16:9 (≈1.78) we need ~2× the base pct so the font is big enough
    # in the video frame to survive the downscaling on a phone.
    return font_size_pct * (1.0 + (aspect - 1.0) * 1.3)


def _resolve_font_size(play_res_y: int, font_size_pct: float) -> int:
    """Calculate ASS font size as a percentage of vertical resolution.

    Enforces a minimum of 42 px so even low-res or square videos stay
    readable on a phone.
    """
    return max(42, round(play_res_y * font_size_pct / 100.0))


def _resolve_outline(play_res_y: int, font_size_pct: float) -> int:
    """Scale outline thickness relative to resolution and font size."""
    return max(2, round(play_res_y * font_size_pct * 0.07 / 100.0))


def generate_ass_subtitles(
    words: list[dict[str, Any]],
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    font_name: str = "Arial",
    font_size_pct: float = 3.5,
    highlight_color: str | None = None,
    normal_color: str | None = None,
    position: str = "lower",
    max_words_per_group: int = 5,
) -> str:
    """Generate an ASS subtitle string with karaoke word highlighting.

    Key design choices:
    - **No temporal overlap**: each line ends exactly when the next begins
      (or earlier if there is a natural gap).
    - **No fade / pop-in**: subtitles snap on/off cleanly.
    - **Lower-third default**: positioned near the bottom of the frame.
    - **Percentage-based sizing**: font and outline scale with resolution.

    Args:
        words: ``[{"word": str, "start": float, "end": float}, ...]``
               Timestamps relative to clip start (0-based).
        play_res_x: Subtitle canvas width  (match output video).
        play_res_y: Subtitle canvas height (match output video).
        font_name: Font family.
        font_size_pct: Font size as percentage of play_res_y (default 3.5%).
        highlight_color: ASS color for the highlighted word.
        normal_color: ASS color for not-yet-spoken words.
        position: ``"lower"`` (default), ``"upper"``, or ``"center"``.
        max_words_per_group: Max words per subtitle line.

    Returns:
        Complete ASS subtitle file as a string.
    """

    hi_color = highlight_color or COLOR_HIGHLIGHT
    nm_color = normal_color or COLOR_NORMAL

    # Auto-adapt for landscape videos viewed on vertical phone screens
    effective_pct = _adapt_for_aspect_ratio(play_res_x, play_res_y, font_size_pct)
    font_size = _resolve_font_size(play_res_y, effective_pct)
    outline_w = _resolve_outline(play_res_y, effective_pct)
    shadow_depth = max(1, outline_w // 2)

    # ── Alignment & margins ──────────────────────────────────────────────
    # ASS alignment numpad: 1-3 bottom, 4-6 middle, 7-9 top
    # We use center-horizontal for each row.
    margin_pct = {
        "lower":  33.3,  # 1/3 from bottom
        "center": 0.0,   # vertically centred (MarginV ignored for align 5)
        "upper":  5.0,   # 5% from top
    }
    alignment_map = {"lower": 2, "center": 5, "upper": 8}
    alignment = alignment_map.get(position, 2)
    margin_v = round(play_res_y * margin_pct.get(position, 5.0) / 100.0)
    margin_h = round(play_res_x * 10.0 / 100.0)   # 10% horizontal padding (TikTok UI clearance)

    # ── ASS header ───────────────────────────────────────────────────────
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_res_x}\n"
        f"PlayResY: {play_res_y}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Sub,{font_name},{font_size},"
        f"{hi_color},{nm_color},{COLOR_OUTLINE},{COLOR_SHADOW},"
        f"-1,0,0,0,100,100,1.5,0,1,{outline_w},{shadow_depth},"
        f"{alignment},{margin_h},{margin_h},{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, "
        "MarginL, MarginR, MarginV, Effect, Text\n"
    )

    # ── Group words ──────────────────────────────────────────────────────
    groups = _group_words(words, max_words=max_words_per_group)
    if not groups:
        return header

    # ── Resolve non-overlapping time ranges ──────────────────────────────
    # Each group is shown from its first word's start to its last word's
    # end, BUT we clamp the end so it never exceeds the next group's start.
    # A tiny gap (MIN_GAP) is enforced so renderers don't double-draw.
    MIN_GAP = 0.04  # 40 ms — 1 frame at 25 fps

    time_ranges: list[tuple[float, float]] = []
    for g in groups:
        time_ranges.append((g[0]["start"], g[-1]["end"]))

    clamped: list[tuple[float, float]] = []
    for i, (gs, ge) in enumerate(time_ranges):
        if i + 1 < len(time_ranges):
            next_start = time_ranges[i + 1][0]
            # end must be before the next group starts
            ge = min(ge, next_start - MIN_GAP)
        # ensure positive duration
        if ge <= gs:
            ge = gs + 0.1
        clamped.append((gs, ge))

    # ── Build dialogue lines ─────────────────────────────────────────────
    dialogue_lines: list[str] = []
    for group, (g_start, g_end) in zip(groups, clamped):
        start_str = _seconds_to_ass_time(g_start)
        end_str   = _seconds_to_ass_time(g_end)

        # Karaoke fill tags: \kf<centiseconds>
        parts: list[tuple[str, str]] = []  # (tagged_text, clean_text)
        for i, word in enumerate(group):
            if i == 0:
                effective_start = g_start
            else:
                effective_start = group[i - 1]["end"]

            dur_cs = max(1, round((word["end"] - effective_start) * 100))
            clean = word["word"].strip()
            if clean:
                tagged = f"{{\\kf{dur_cs}}}{clean}"
                parts.append((tagged, clean))

        # Join parts, but don't add space before punctuation marks
        text_parts = []
        for i, (tagged_text, clean_text) in enumerate(parts):
            # Don't add space before punctuation like - , . : ; ! ? ) ] }
            if i > 0 and clean_text and clean_text[0] not in "-,.:;!?)]}":
                text_parts.append(" ")
            text_parts.append(tagged_text)
        
        text = "".join(text_parts)
        line = f"Dialogue: 0,{start_str},{end_str},Sub,,0,0,0,,{text}"
        dialogue_lines.append(line)

    return header + "\n".join(dialogue_lines) + "\n"


def generate_title_overlay(
    title: str,
    play_res_x: int = 1080,
    play_res_y: int = 1920,
    duration: float = 3.0,
    font_name: str = "Arial",
) -> str:
    """Generate ASS title overlay without any animation.

    Single text element centered on screen. The title appears for the
    full duration with no scaling or fading effects (static display).

    Args:
        title: The title text to display (single element, no wrapping).
        play_res_x: Video width.
        play_res_y: Video height.
        duration: How long the title appears (default 3.0s).
        font_name: Font family.

    Returns:
        Complete ASS subtitle file as a string.
    """
    # ── Sizing ───────────────────────────────────────────────────────────────
    aspect = play_res_x / play_res_y
    if aspect < 1.0:
        font_size = int(play_res_x * 0.09)
    else:
        font_size = int(play_res_y * 0.10)

    outline_w = max(3, int(font_size * 0.08))
    shadow_d = max(2, outline_w // 2)

    # ── Colors ───────────────────────────────────────────────────────────────
    c_text = _rgb_to_ass(255, 255, 255)
    c_outline = _rgb_to_ass(0, 0, 0)
    c_shadow = _rgb_to_ass(0, 0, 0, 120)

    # ── Center positioning (30% from top) ────────────────────────────────────
    cx = play_res_x // 2
    cy = int(play_res_y * 0.30)

    # ── Timing ───────────────────────────────────────────────────────────────
    # Compute fade-out parameters. we always start fully opaque and then
    # transition to transparent near the end of the title duration.  For very
    # short clips the fade is shortened proportionally so it remains visible.
    if duration < 1.5:
        fade_out_dur = max(100, int(duration * 200))   # ~20% of clip, in ms
    else:
        fade_out_dur = int(0.4 * 1000)                 # 400 ms default
    fade_out_start = max(0, int((duration - (fade_out_dur / 1000)) * 1000))

    margin_h = round(play_res_x * 4.0 / 100.0)

    # ── ASS header ───────────────────────────────────────────────────────────
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_res_x}\n"
        f"PlayResY: {play_res_y}\n"
        "WrapStyle: 0\n"
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        # alignment 5 = center-center
        f"Style: Title,{font_name},{font_size},"
        f"{c_text},{c_text},{c_outline},{c_shadow},"
        f"-1,0,0,0,100,100,0,0,1,{outline_w},{shadow_d},"
        f"5,{margin_h},{margin_h},0,1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, "
        "MarginL, MarginR, MarginV, Effect, Text\n"
    )

    # ── Single dialogue event ────────────────────────────────────────────────
    # Centered text with fade-out near the end.  We keep the alpha at 0
    # (opaque) initially, then animate to &HFF (transparent).
    t0 = _seconds_to_ass_time(0.0)
    t1 = _seconds_to_ass_time(duration)

    anim = (
        f"{{\\pos({cx},{cy})\\an5"
        f"\\alpha&H00"
        f"\\t({fade_out_start},{fade_out_start + fade_out_dur},\\alpha&HFF)"
        f"}}"
    )

    event = f"Dialogue: 0,{t0},{t1},Title,,0,0,0,,{anim}{title}"

    return header + event + "\n"


def get_clip_words(
    segments: list[dict[str, Any]],
    clip_start: float,
    clip_end: float,
) -> list[dict[str, Any]]:
    """Extract word-level timestamps for a clip's time range.

    Returns words with timestamps adjusted to be relative to clip start
    (0-based) and sorted chronologically.
    """
    words: list[dict[str, Any]] = []

    for seg in segments:
        if seg["end"] < clip_start or seg["start"] > clip_end:
            continue

        for w in seg.get("words", []):
            w_start = w.get("start", 0)
            w_end   = w.get("end", 0)
            w_text  = w.get("word", "").strip()

            if not w_text:
                continue
            # Only include words whose midpoints fall inside the clip
            w_mid = (w_start + w_end) / 2.0
            if w_mid < clip_start or w_mid > clip_end:
                continue

            words.append({
                "word": w_text,
                "start": max(0.0, w_start - clip_start),
                "end":   max(0.0, w_end   - clip_start),
            })

    words.sort(key=lambda w: w["start"])
    return words
