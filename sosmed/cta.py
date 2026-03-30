"""
Instagram CTA — fades to black and shows a follow prompt at the end of clips.

Appends a professional 3-second CTA screen with:
  • Fade-from-video transition to black
  • Instagram camera icon (light-blue geometric)
  • Account name (large, white)
  • Username (light blue, with @)
  • "Follow" button (light-blue fill, dark text)
  • Click sound when the button appears
"""

import subprocess
from pathlib import Path

from .postprocess import _get_video_info
from .utils import get_ffmpeg, log


def _esc(text: str) -> str:
    """Escape text for FFmpeg drawtext filter value."""
    text = text.replace("\\", "\\\\")
    text = text.replace("'", "\\'")
    text = text.replace(":", "\\:")
    text = text.replace("%", "\\%")
    return text


def _fade(t0: float, t1: float) -> str:
    """Smooth 0→1 alpha expression for FFmpeg drawtext."""
    span = t1 - t0
    return f"if(lt(t,{t0:.3f}),0,if(lt(t,{t1:.3f}),(t-{t0:.3f})/{span:.3f},1))"


def append_instagram_cta(
    clip_path: str,
    output_path: str,
    name: str,
    username: str,
    duration: float = 3.0,
    fade_duration: float = 0.5,
) -> str:
    """Append an Instagram follow CTA to the end of a video clip.

    Fades from the clip to a black screen and shows a styled CTA with
    animated text, a follow button, and a click sound effect.

    Args:
        clip_path:     Path to the input (post-processed) clip.
        output_path:   Path to write the final clip with CTA appended.
        name:          Display name of the Instagram account.
        username:      Instagram handle (@ prefix added automatically if missing).
        duration:      Duration of the CTA screen in seconds.
        fade_duration: Length of the cross-fade transition.

    Returns:
        output_path on success; clip_path on failure (leaves input intact).
    """
    info = _get_video_info(clip_path)
    if not info.get("has_video"):
        log("WARN", "CTA: no video stream in clip, skipping")
        return clip_path

    w = info["width"] or 1080
    h = info["height"] or 1920
    fps = max(float(info.get("fps") or 30.0), 15.0)
    main_dur = float(info.get("duration") or 0)
    has_audio = info["has_audio"]

    if main_dur <= 0:
        log("WARN", "CTA: could not determine clip duration, skipping")
        return clip_path

    if username and not username.startswith("@"):
        username = "@" + username

    # Clamp fade so it never exceeds 40 % of the clip
    fade_duration = min(fade_duration, main_dur * 0.4)

    # ── Accent colour ─────────────────────────────────────────────────────────
    ACCENT = "0x29B6F6"   # light blue

    # ── Font sizes ────────────────────────────────────────────────────────────
    name_fs = max(58, int(h * 0.068))   # account name  (big & clear)
    user_fs = max(38, int(h * 0.040))   # @username     (medium)
    btn_fs  = max(30, int(h * 0.032))   # "Follow" inside button

    # ── Instagram camera icon ─────────────────────────────────────────────────
    icon_size = max(90, int(h * 0.057))   # outer square side
    icon_cx   = w // 2
    icon_top  = int(h * 0.290)
    icon_x    = icon_cx - icon_size // 2
    icon_bw   = max(5, int(icon_size * 0.075))   # border thickness

    lens_size = int(icon_size * 0.50)             # inner lens square
    lens_x    = icon_cx - lens_size // 2
    lens_y    = icon_top + (icon_size - lens_size) // 2
    lens_bw   = max(4, int(icon_size * 0.055))

    dot_size  = max(7, int(icon_size * 0.10))     # viewfinder dot (top-right)
    dot_x     = icon_x + icon_size - dot_size - int(icon_size * 0.16)
    dot_y     = icon_top + int(icon_size * 0.12)

    # ── Vertical positions ────────────────────────────────────────────────────
    name_y = int(h * 0.420)
    user_y = int(h * 0.510)

    btn_h  = max(60, int(h * 0.068))
    btn_w  = max(200, int(w * 0.400))
    btn_x  = (w - btn_w) // 2
    btn_y  = int(h * 0.578)
    btn_ty = btn_y + (btn_h - btn_fs) // 2

    # ── Animation timing (t = seconds into CTA segment) ──────────────────────
    t_icon = 0.10
    t_name = (0.25, 0.60)
    t_user = (0.42, 0.75)
    t_btn  = 1.00    # button pop-in + click sound

    name_esc = _esc(name)
    user_esc = _esc(username)

    # ── CTA visual filter chain (all applied to the lavfi black background) ──
    filters: list[str] = [
        # ── Instagram logo: camera icon (geometric flat design) ───────────────

        # Camera body — outer square border
        (f"drawbox=x={icon_x}:y={icon_top}"
         f":w={icon_size}:h={icon_size}"
         f":color={ACCENT}:t={icon_bw}"
         f":enable='gte(t,{t_icon:.3f})'"),

        # Camera lens — inner square border (centred inside body)
        (f"drawbox=x={lens_x}:y={lens_y}"
         f":w={lens_size}:h={lens_size}"
         f":color={ACCENT}:t={lens_bw}"
         f":enable='gte(t,{t_icon:.3f})'"),

        # Viewfinder dot — small filled square (top-right of body)
        (f"drawbox=x={dot_x}:y={dot_y}"
         f":w={dot_size}:h={dot_size}"
         f":color={ACCENT}:t=fill"
         f":enable='gte(t,{t_icon:.3f})'"),

        # ── Account name ──────────────────────────────────────────────────────

        # Drop shadow
        (f"drawtext=font=Montserrat:fontsize={name_fs}"
         f":text='{name_esc}'"
         f":x=(w-tw)/2+3:y={name_y + 3}"
         f":fontcolor=0x000000@0.60"
         f":alpha='{_fade(*t_name)}'"),

        # Main text (white)
        (f"drawtext=font=Montserrat:fontsize={name_fs}"
         f":text='{name_esc}'"
         f":x=(w-tw)/2:y={name_y}"
         f":fontcolor=white"
         f":bordercolor=0x111111:borderw=3"
         f":alpha='{_fade(*t_name)}'"),

        # ── @username (light blue) ────────────────────────────────────────────
        (f"drawtext=font=Montserrat:fontsize={user_fs}"
         f":text='{user_esc}'"
         f":x=(w-tw)/2:y={user_y}"
         f":fontcolor={ACCENT}"
         f":bordercolor=0x111111:borderw=2"
         f":alpha='{_fade(*t_user)}'"),

        # ── Follow button (light-blue fill, dark text) ────────────────────────
        (f"drawbox=x={btn_x}:y={btn_y}:w={btn_w}:h={btn_h}"
         f":color={ACCENT}:t=fill"
         f":enable='gte(t,{t_btn:.3f})'"),

        (f"drawtext=font=Montserrat:fontsize={btn_fs}"
         f":text='Follow'"
         f":x=(w-tw)/2:y={btn_ty}"
         f":fontcolor=0x0D1B2A"
         f":enable='gte(t,{t_btn:.3f})'"),
    ]

    cta_chain = ",".join(filters)

    # ── Timing calculations ───────────────────────────────────────────────────
    xfade_offset = max(0.0, main_dur - fade_duration)
    total_dur    = xfade_offset + duration      # output video duration

    # Click plays at t_btn seconds into the CTA portion of the full output
    click_ms = int((xfade_offset + t_btn) * 1000)

    # Synthetic click: sharp transient (high-freq + sub)
    click_expr = (
        "0.8*sin(2*PI*1200*t)*exp(-90*t)"
        "+0.4*sin(2*PI*2800*t)*exp(-110*t)"
        "+0.2*sin(2*PI*600*t)*exp(-70*t)"
    )

    # ── Build FFmpeg command ──────────────────────────────────────────────────
    ffmpeg = get_ffmpeg()
    cmd: list[str] = [ffmpeg, "-y", "-hide_banner"]

    # [0] main clip
    cmd.extend(["-i", clip_path])

    # [1] black CTA background  (lavfi color source, same size/fps as clip)
    cmd.extend([
        "-f", "lavfi", "-t", f"{duration:.4f}",
        "-i", f"color=c=black:size={w}x{h}:rate={fps:.4f}",
    ])

    # [2] click sound  (lavfi aevalsrc, very short)
    cmd.extend([
        "-f", "lavfi", "-t", "0.15",
        "-i", f"aevalsrc={click_expr}:s=44100",
    ])

    # ── filter_complex ────────────────────────────────────────────────────────
    fc: list[str] = []

    # Apply CTA visuals to the black background
    fc.append(f"[1:v]{cta_chain}[cta_v]")

    # Normalise the main clip's timebase to match the lavfi source before xfade.
    # libx264 uses 1/15360 while lavfi uses 1/fps; mismatched timebases cause
    # "do not match" errors in xfade.
    fc.append(f"[0:v]fps={fps:.4f}[v0norm]")

    # xfade: clip fades into CTA black screen
    fc.append(
        f"[v0norm][cta_v]xfade=transition=fade"
        f":duration={fade_duration:.4f}"
        f":offset={xfade_offset:.4f}[vout]"
    )

    # Audio path
    if has_audio:
        audio_fade_st  = max(0.0, main_dur - fade_duration - 0.15)
        audio_fade_dur = fade_duration + 0.15
        fc.append(
            f"[0:a]afade=t=out:st={audio_fade_st:.4f}:d={audio_fade_dur:.4f}"
            f",apad=whole_dur={total_dur:.4f}[main_a]"
        )
        fc.append(f"[2:a]adelay=delays={click_ms}:all=1[click_a]")
        fc.append("[main_a][click_a]amix=inputs=2:duration=first[aout]")
    else:
        # No original audio — just place the click with silence padding
        fc.append(
            f"[2:a]adelay=delays={click_ms}:all=1"
            f",apad=whole_dur={total_dur:.4f}[aout]"
        )

    cmd.extend(["-filter_complex", ";".join(fc)])
    cmd.extend(["-map", "[vout]", "-map", "[aout]"])
    cmd.extend(["-c:v", "libx264", "-preset", "fast", "-crf", "23"])
    cmd.extend(["-c:a", "aac", "-b:a", "192k"])
    cmd.extend(["-movflags", "+faststart", "-loglevel", "error"])

    # Write to a temp file next to the target, then atomic-rename
    out  = Path(output_path)
    tmp  = out.parent / (out.stem + "_ctatmp" + out.suffix)
    cmd.append(str(tmp))

    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        tmp.replace(out)
        log("DEBUG", f"CTA appended → {out.name}")
        return str(out)
    except subprocess.CalledProcessError as e:
        detail = (e.stderr or "")[:400]
        log("ERROR", f"CTA append failed: {detail}")
        log("DEBUG", f"CTA cmd: {' '.join(cmd)}")
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        return clip_path   # leave original untouched on failure
