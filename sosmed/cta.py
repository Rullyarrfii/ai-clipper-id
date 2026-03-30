"""
Instagram CTA — fades to black and shows a follow prompt at the end of clips.

Appends a professional 3-second CTA screen with:
  • Fade-from-video transition to black
  • Instagram accent bar
  • Account name (large, white)
  • Username (pink, with @)
  • "Follow" button (white pill, black text)
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

    # ── Layout (percentage-based so it works for any resolution) ─────────────
    margin_h = int(w * 0.10)   # 10 % left/right margin — comfortable breathing room

    # Font sizes
    label_fs = max(26, int(h * 0.022))   # "Follow on Instagram"
    name_fs  = max(58, int(h * 0.070))   # account name  (big & clear)
    user_fs  = max(40, int(h * 0.042))   # @username     (medium)
    btn_fs   = max(32, int(h * 0.034))   # "Follow" inside button

    # Vertical positions — block centered ~32 %–65 % of frame height
    accent_y = int(h * 0.320)
    accent_h = max(5, int(h * 0.006))    # slightly thicker accent bar
    label_y  = int(h * 0.355)
    name_y   = int(h * 0.400)
    user_y   = int(h * 0.505)

    btn_h        = max(64, int(h * 0.072))
    btn_w        = max(220, int(w * 0.420))   # 42 % width — compact, button-like
    btn_x        = (w - btn_w) // 2
    btn_y        = int(h * 0.580)
    btn_ty       = btn_y + (btn_h - btn_fs) // 2   # vertically centre "Follow" text
    btn_border   = max(3, int(h * 0.003))           # pink border thickness

    # ── Animation timing (t = seconds into CTA segment, i.e. starts at 0) ────
    t_accent = 0.10
    t_label  = (0.15, 0.55)
    t_name   = (0.30, 0.70)
    t_user   = (0.50, 0.90)
    t_btn    = 1.10    # button pop-in + click sound

    name_esc = _esc(name)
    user_esc = _esc(username)

    # ── CTA visual filter chain (all applied to the lavfi black background) ──
    filters: list[str] = [
        # Instagram accent bar (pink horizontal rule above text)
        (f"drawbox=x={margin_h}:y={accent_y}"
         f":w={w - 2 * margin_h}:h={accent_h}"
         f":color=0xE1306C:t=fill"
         f":enable='gte(t,{t_accent:.3f})'"),

        # "Follow on Instagram" label
        (f"drawtext=font=Montserrat:fontsize={label_fs}"
         f":text='Follow on Instagram'"
         f":x=(w-tw)/2:y={label_y}"
         f":fontcolor=0xCCCCCC"
         f":bordercolor=0x000000:borderw=1"
         f":alpha='{_fade(*t_label)}'"),

        # Account name — drop shadow
        (f"drawtext=font=Montserrat:fontsize={name_fs}"
         f":text='{name_esc}'"
         f":x=(w-tw)/2+4:y={name_y + 4}"
         f":fontcolor=0x000000@0.70"
         f":alpha='{_fade(*t_name)}'"),

        # Account name — main text (large, white, bold look via border)
        (f"drawtext=font=Montserrat:fontsize={name_fs}"
         f":text='{name_esc}'"
         f":x=(w-tw)/2:y={name_y}"
         f":fontcolor=white"
         f":bordercolor=0x111111:borderw=4"
         f":alpha='{_fade(*t_name)}'"),

        # Username — drop shadow
        (f"drawtext=font=Montserrat:fontsize={user_fs}"
         f":text='{user_esc}'"
         f":x=(w-tw)/2+2:y={user_y + 2}"
         f":fontcolor=0x000000@0.65"
         f":alpha='{_fade(*t_user)}'"),

        # Username — main text (Instagram pink)
        (f"drawtext=font=Montserrat:fontsize={user_fs}"
         f":text='{user_esc}'"
         f":x=(w-tw)/2:y={user_y}"
         f":fontcolor=0xF06292"
         f":bordercolor=0x111111:borderw=2"
         f":alpha='{_fade(*t_user)}'"),

        # Follow button — pink border (drawn first, slightly larger)
        (f"drawbox=x={btn_x - btn_border}:y={btn_y - btn_border}"
         f":w={btn_w + 2 * btn_border}:h={btn_h + 2 * btn_border}"
         f":color=0xE1306C:t=fill"
         f":enable='gte(t,{t_btn:.3f})'"),

        # Follow button — white fill (over the border)
        (f"drawbox=x={btn_x}:y={btn_y}:w={btn_w}:h={btn_h}"
         f":color=white:t=fill"
         f":enable='gte(t,{t_btn:.3f})'"),

        # Follow button — "Follow" label (black text on white)
        (f"drawtext=font=Montserrat:fontsize={btn_fs}"
         f":text='Follow'"
         f":x=(w-tw)/2:y={btn_ty}"
         f":fontcolor=black"
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

    # xfade: clip fades into CTA black screen
    fc.append(
        f"[0:v][cta_v]xfade=transition=fade"
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
