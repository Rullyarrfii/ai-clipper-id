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
    margin_h = round(play_res_x * 4.0 / 100.0)   # 4% horizontal padding

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
    """Generate an ASS overlay for the clip title with per-letter animation.

    Creates a moderate-sized, bold title positioned 1/3 from top that:
    - Fits comfortably within video bounds with proper margins
    - Each letter animates in individually with scale, rotation, and fade
    - Uses 8% of video height for font size (readable but not overwhelming)
    - Has strong outline and shadow for readability
    - Centered horizontally with smart text wrapping
    - Keeps clear separation from subtitle area

    Args:
        title: The title text to display
        play_res_x: Video width
        play_res_y: Video height
        duration: How long the title appears (default 3.0s)
        font_name: Font family

    Returns:
        Complete ASS subtitle file as a string
    """
    # Moderate font size - 8% of video height (readable, doesn't overflow)
    font_size = int(play_res_y * 0.08)
    
    # Proportional outline for readability
    outline_w = max(2, int(font_size * 0.1))
    shadow_depth = max(1, outline_w // 2)
    
    # Position: 1/3 from top
    margin_v = int(play_res_y * 0.333)
    # Generous horizontal margins (12%) to prevent overflow
    margin_h = int(play_res_x * 0.12)
    
    # Colors: bold white text with black outline
    color_text = _rgb_to_ass(255, 255, 255)  # White
    color_outline = _rgb_to_ass(0, 0, 0)      # Black
    color_shadow = _rgb_to_ass(0, 0, 0, 100)  # Dark shadow
    
    # Center alignment (alignment 5 = middle center)
    alignment = 5
    
    header = (
        "[Script Info]\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {play_res_x}\n"
        f"PlayResY: {play_res_y}\n"
        "WrapStyle: 2\n"  # Smart wrapping to prevent overflow
        "ScaledBorderAndShadow: yes\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Title,{font_name},{font_size},"
        f"{color_text},{color_text},{color_outline},{color_shadow},"
        f"-1,0,0,0,100,100,0.8,0,1,{outline_w},{shadow_depth},"
        f"{alignment},{margin_h},{margin_h},{margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, "
        "MarginL, MarginR, MarginV, Effect, Text\n"
    )
    
    start_time = _seconds_to_ass_time(0.0)
    end_time = _seconds_to_ass_time(duration)
    
    # Create per-letter animation
    # Each letter: starts small (60% scale), rotated, and fades in
    # Animation spreads over first 1000ms, then holds, then fades out
    
    # Split title into characters (preserving spaces)
    chars = list(title)
    total_letters = len([c for c in chars if c.strip()])  # Count non-space chars
    
    # Animation parameters
    animation_duration = 1000  # ms - total time for all letters to animate in
    letter_delay = min(80, animation_duration // max(1, total_letters))  # delay between letters
    per_letter_anim = 250  # ms - duration for each letter's entrance animation
    fade_out_start = (duration - 0.3) * 1000  # Start fade out 300ms before end
    fade_out_duration = 300  # ms
    
    # Build animated text with per-letter effects
    animated_parts = []
    letter_index = 0
    
    for char in chars:
        if char.strip():  # Non-space character
            delay = letter_index * letter_delay
            anim_end = delay + per_letter_anim
            
            # Each letter animation:
            # 1. Start: 60% scale, 10° rotation, partially transparent
            # 2. Transform to: 100% scale, 0° rotation, fully opaque
            # 3. At end: fade out
            
            # Initial state (small, slightly rotated, semi-transparent)
            initial = f"\\alpha&H80\\fscx60\\fscy60\\frz10"
            # Transform to normal over per_letter_anim duration
            transform = f"\\t({delay},{anim_end},\\alpha&H00\\fscx100\\fscy100\\frz0)"
            # Fade out at the end
            fade_out = f"\\t({int(fade_out_start)},{int(fade_out_start + fade_out_duration)},\\alpha&HFF)"
            
            animated_parts.append(f"{{{initial}{transform}{fade_out}}}{char}")
            letter_index += 1
        else:
            # Space - no animation, just maintain the space
            animated_parts.append(char)
    
    text_with_animation = "".join(animated_parts)
    
    dialogue = f"Dialogue: 0,{start_time},{end_time},Title,,0,0,0,,{text_with_animation}\n"
    
    return header + dialogue


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
