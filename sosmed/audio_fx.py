"""
Background music mixing and sound effects.

- Generates simple SFX (whoosh, impact) using FFmpeg audio synthesis
- Caches generated SFX in .cache directory
- Builds FFmpeg audio filter graphs for music ducking and SFX mixing
"""

import subprocess
from pathlib import Path
from typing import Any

from .utils import log

# ── SFX cache directory ─────────────────────────────────────────────────────

_SFX_CACHE_DIR = Path.cwd() / ".cache" / "ai-video-clipper" / "sfx"


def _ensure_sfx_dir() -> Path:
    """Create and return SFX cache directory."""
    _SFX_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _SFX_CACHE_DIR


def _generate_sfx_file(output_path: Path, lavfi_src: str) -> bool:
    """Generate an audio file using FFmpeg lavfi source."""
    if output_path.exists():
        return True
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", lavfi_src,
                "-c:a", "pcm_s16le",
                str(output_path),
            ],
            check=True, capture_output=True,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        log("WARN", f"Failed to generate SFX {output_path.name}: {e}")
        return False


def generate_whoosh(cache_dir: Path | None = None) -> Path | None:
    """Generate a subtle whoosh/sweep transition sound effect.

    A pink noise burst swept through a bandpass filter with fade envelope.
    """
    sfx_dir = cache_dir or _ensure_sfx_dir()
    path = sfx_dir / "whoosh.wav"
    lavfi = (
        "anoisesrc=d=0.45:c=pink:s=44100,"
        "highpass=f=2500,"
        "lowpass=f=9000,"
        "afade=t=in:ss=0:d=0.12,"
        "afade=t=out:st=0.22:d=0.23,"
        "volume=0.55"
    )
    return path if _generate_sfx_file(path, lavfi) else None


def generate_impact(cache_dir: Path | None = None) -> Path | None:
    """Generate a subtle low-end impact/thud for emphasis moments."""
    sfx_dir = cache_dir or _ensure_sfx_dir()
    path = sfx_dir / "impact.wav"
    lavfi = (
        "sine=frequency=55:duration=0.25:sample_rate=44100,"
        "afade=t=out:st=0.04:d=0.21,"
        "volume=0.65"
    )
    return path if _generate_sfx_file(path, lavfi) else None


def generate_rise(cache_dir: Path | None = None) -> Path | None:
    """Generate a subtle rising tone for energy build-up."""
    sfx_dir = cache_dir or _ensure_sfx_dir()
    path = sfx_dir / "rise.wav"
    lavfi = (
        "sine=frequency=300:duration=0.35:sample_rate=44100,"
        "asetrate=44100*1.4,"
        "aresample=44100,"
        "afade=t=in:ss=0:d=0.1,"
        "afade=t=out:st=0.18:d=0.17,"
        "volume=0.35"
    )
    return path if _generate_sfx_file(path, lavfi) else None


def ensure_sfx_exist() -> dict[str, Path]:
    """Generate all SFX files if they don't exist. Returns paths dict."""
    sfx_dir = _ensure_sfx_dir()
    paths: dict[str, Path] = {}

    whoosh = generate_whoosh(sfx_dir)
    if whoosh:
        paths["whoosh"] = whoosh

    impact = generate_impact(sfx_dir)
    if impact:
        paths["impact"] = impact

    rise = generate_rise(sfx_dir)
    if rise:
        paths["rise"] = rise

    if paths:
        log("OK", f"SFX ready ({', '.join(paths.keys())})")
    return paths


def build_audio_filter(
    clip_duration: float,
    has_original_audio: bool = True,
    music_path: str | None = None,
    sfx_paths: dict[str, Path] | None = None,
    music_volume: float = 0.18,
    sfx_volume: float = 0.55,
) -> tuple[list[str], str, int]:
    """Build FFmpeg audio filter graph for music ducking and SFX mixing.

    Returns:
        (extra_inputs, filter_complex_audio, audio_input_count):
        - extra_inputs: list of ["-i", path] args for additional audio inputs
        - filter_complex_audio: the audio portion of the filter_complex string
        - audio_input_count: total number of audio inputs (including the main video)

    Input mapping convention:
        [0:a] = original video audio
        [1:a] = music (if provided)
        [next:a] = SFX files
    """
    extra_inputs: list[str] = []
    filters: list[str] = []
    input_idx = 1  # 0 is the main video

    sfx_paths = sfx_paths or {}
    mix_inputs: list[str] = []
    mix_weights: list[str] = []

    # Original audio is always first in the mix
    if has_original_audio:
        mix_inputs.append("[0:a]")
        mix_weights.append("1")

    # ── Music with ducking ───────────────────────────────────────────────────
    if music_path:
        extra_inputs.extend(["-i", music_path])
        music_idx = input_idx
        input_idx += 1

        # Music processing: volume, fade in/out, loop to clip duration, duck under speech
        fade_out_start = max(0, clip_duration - 2.5)
        music_filter = (
            f"[{music_idx}:a]"
            f"aloop=loop=-1:size=2e+09,"  # loop music if shorter than clip
            f"atrim=0:{clip_duration + 1},"
            f"volume={music_volume},"
            f"afade=t=in:d=1.5,"
            f"afade=t=out:st={fade_out_start:.2f}:d=2.5"
        )

        if has_original_audio:
            # Side-chain compression: duck music when speech is present
            music_filter += f"[_m];"
            filters.append(music_filter)
            filters.append(
                f"[_m][0:a]sidechaincompress="
                f"threshold=0.015:ratio=8:attack=50:release=800"
                f"[music_ducked]"
            )
            mix_inputs.append("[music_ducked]")
        else:
            music_filter += f"[music_ducked]"
            filters.append(music_filter)
            mix_inputs.append("[music_ducked]")

        mix_weights.append("1")

    # ── Sound effects ────────────────────────────────────────────────────────
    # Whoosh at the start
    if "whoosh" in sfx_paths:
        extra_inputs.extend(["-i", str(sfx_paths["whoosh"])])
        sfx_idx = input_idx
        input_idx += 1
        filters.append(
            f"[{sfx_idx}:a]volume={sfx_volume},"
            f"adelay=50|50[sfx_whoosh]"  # 50ms delay
        )
        mix_inputs.append("[sfx_whoosh]")
        mix_weights.append("1")

    # Impact near the start (at ~0.5s for emphasis)
    if "impact" in sfx_paths:
        extra_inputs.extend(["-i", str(sfx_paths["impact"])])
        sfx_idx = input_idx
        input_idx += 1
        filters.append(
            f"[{sfx_idx}:a]volume={sfx_volume * 0.7},"
            f"adelay=400|400[sfx_impact]"  # 400ms delay
        )
        mix_inputs.append("[sfx_impact]")
        mix_weights.append("1")

    # Rise sound if clip is long enough (at 70% mark for build-up)
    if "rise" in sfx_paths and clip_duration > 20:
        extra_inputs.extend(["-i", str(sfx_paths["rise"])])
        sfx_idx = input_idx
        input_idx += 1
        rise_delay = int(clip_duration * 0.7 * 1000)  # 70% into clip, in ms
        filters.append(
            f"[{sfx_idx}:a]volume={sfx_volume * 0.5},"
            f"adelay={rise_delay}|{rise_delay}[sfx_rise]"
        )
        mix_inputs.append("[sfx_rise]")
        mix_weights.append("1")

    # ── Mix all audio streams ────────────────────────────────────────────────
    n_mix = len(mix_inputs)
    if n_mix == 0:
        # No audio at all — generate silence for the full clip duration
        dur = clip_duration + 0.1  # Add slight buffer to ensure full coverage
        return extra_inputs, f"anullsrc=r=44100:cl=stereo:d={dur}[aout]", input_idx
    elif n_mix == 1:
        # Single audio source
        src = mix_inputs[0]
        if filters:
            # We have processing filters applied, include them
            filter_str = ";".join(filters)
            # Append input mapping to output for the final result  
            filter_str += f";{src}acopy[aout]"
        else:
            # No filters, just copy input to named output
            filter_str = f"{src}acopy[aout]"
        return extra_inputs, filter_str, input_idx
    else:
        # Mix multiple sources
        mix_str = "".join(mix_inputs)
        weights_str = " ".join(mix_weights)
        filters.append(
            f"{mix_str}amix=inputs={n_mix}:"
            f"duration=first:"
            f"dropout_transition=2:"
            f"weights={weights_str}[aout]"
        )
        filter_str = ";".join(filters)
        return extra_inputs, filter_str, input_idx
