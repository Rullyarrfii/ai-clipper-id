"""
Background music selection and mixing for video clips.

Uses a curated library of royalty-free music organized by mood/category.
LLM matches clip content to appropriate background music.
Music is mixed at very low volume as ambient vibe.
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import requests

from .utils import get_ffmpeg, log


# ── Royalty-free music library ───────────────────────────────────────────────
# URLs point to royalty-free/Creative Commons music sources.
# Categories map to moods/vibes that match different clip content.
#
# These are placeholder URLs — users should download and place actual
# music files in the music/ directory, then update MUSIC_LIBRARY paths.
MUSIC_LIBRARY: list[dict[str, str]] = [
    {
        "id": "upbeat_pop",
        "category": "upbeat",
        "mood": "energetic, happy, motivational, fun",
        "description": "Upbeat pop/electronic beat — great for tips, tutorials, success stories",
        "file": "upbeat_pop.mp3",
    },
    {
        "id": "chill_lofi",
        "category": "chill",
        "mood": "relaxed, calm, thoughtful, cozy",
        "description": "Lo-fi hip hop chill beat — perfect for storytelling, explanations, reflections",
        "file": "chill_lofi.mp3",
    },
    {
        "id": "dramatic_cinematic",
        "category": "dramatic",
        "mood": "intense, serious, suspenseful, emotional",
        "description": "Cinematic orchestral — for shocking reveals, serious topics, emotional moments",
        "file": "dramatic_cinematic.mp3",
    },
    {
        "id": "inspiring_ambient",
        "category": "inspiring",
        "mood": "hopeful, uplifting, inspirational, warm",
        "description": "Ambient inspirational — for motivational content, success stories, advice",
        "file": "inspiring_ambient.mp3",
    },
    {
        "id": "tech_electronic",
        "category": "tech",
        "mood": "futuristic, modern, innovative, fast",
        "description": "Electronic/tech beat — for tech topics, innovation, how-tos, demonstrations",
        "file": "tech_electronic.mp3",
    },
    {
        "id": "funny_quirky",
        "category": "funny",
        "mood": "comedic, playful, silly, lighthearted",
        "description": "Quirky comedic — for funny moments, memes, roasts, lighthearted content",
        "file": "funny_quirky.mp3",
    },
    {
        "id": "sad_piano",
        "category": "emotional",
        "mood": "sad, melancholic, touching, sentimental",
        "description": "Soft piano — for emotional stories, heartfelt moments, personal sharing",
        "file": "sad_piano.mp3",
    },
    {
        "id": "hype_trap",
        "category": "hype",
        "mood": "aggressive, confident, bold, powerful",
        "description": "Trap/bass beat — for bold claims, confrontational takes, confidence, hype",
        "file": "hype_trap.mp3",
    },
]


# ── Pixabay search queries per music category ──────────────────────────────
# Maps each library entry to a Pixabay search query + category filter.
_PIXABAY_QUERIES: dict[str, dict[str, str]] = {
    "upbeat_pop": {"q": "upbeat pop energetic", "category": "music"},
    "chill_lofi": {"q": "lofi chill hip hop", "category": "music"},
    "dramatic_cinematic": {"q": "cinematic dramatic orchestral", "category": "music"},
    "inspiring_ambient": {"q": "inspirational ambient uplifting", "category": "music"},
    "tech_electronic": {"q": "electronic technology modern", "category": "music"},
    "funny_quirky": {"q": "funny quirky comedy", "category": "music"},
    "sad_piano": {"q": "sad piano emotional", "category": "music"},
    "hype_trap": {"q": "trap bass aggressive", "category": "music"},
}


def download_music_library(
    music_dir: str | Path = "music",
    api_key: str | None = None,
    min_duration: int = 30,
) -> list[str]:
    """Download copyright-free music from Pixabay for each mood category.

    Pixabay music is 100% royalty-free, no attribution required, safe for
    commercial use including YouTube, TikTok, etc.

    Get a free API key at: https://pixabay.com/api/docs/

    Args:
        music_dir: Directory to save music files into.
        api_key: Pixabay API key. Falls back to PIXABAY_API_KEY env var.
        min_duration: Minimum track duration in seconds (default: 30).

    Returns:
        List of downloaded file paths.
    """
    api_key = api_key or os.environ.get("PIXABAY_API_KEY", "")
    if not api_key:
        log("ERROR", "Pixabay API key required for music download. "
                      "Get a free key at https://pixabay.com/api/docs/ "
                      "then set PIXABAY_API_KEY env var.")
        return []

    music_path = Path(music_dir)
    music_path.mkdir(parents=True, exist_ok=True)

    downloaded: list[str] = []

    for entry in MUSIC_LIBRARY:
        mid = entry["id"]
        target_file = music_path / f"{mid}.mp3"

        # Skip if already downloaded
        if target_file.exists() and target_file.stat().st_size > 10_000:
            log("OK", f"  {mid}: already exists ({target_file.stat().st_size // 1024} KB)")
            downloaded.append(str(target_file))
            continue

        query_info = _PIXABAY_QUERIES.get(mid)
        if not query_info:
            log("WARN", f"  {mid}: no search query defined, skipping")
            continue

        # Search Pixabay audio API
        try:
            resp = requests.get(
                "https://pixabay.com/api/",
                params={
                    "key": api_key,
                    "q": query_info["q"],
                    "media_type": "music",
                    "per_page": 5,
                    "min_duration": min_duration,
                    "order": "popular",
                    "editors_choice": "true",
                },
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            # Try without editors_choice filter
            try:
                resp = requests.get(
                    "https://pixabay.com/api/",
                    params={
                        "key": api_key,
                        "q": query_info["q"],
                        "media_type": "music",
                        "per_page": 5,
                        "min_duration": min_duration,
                        "order": "popular",
                    },
                    timeout=15,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e2:
                log("WARN", f"  {mid}: Pixabay search failed: {e2}")
                continue

        hits = data.get("hits", [])
        if not hits:
            log("WARN", f"  {mid}: no results for query '{query_info['q']}'")
            continue

        # Pick the first (most popular) hit
        hit = hits[0]
        audio_url = hit.get("audio") or hit.get("previewURL") or ""
        if not audio_url:
            # Try alternative URL fields
            for h in hits:
                audio_url = h.get("audio") or h.get("previewURL") or ""
                if audio_url:
                    hit = h
                    break

        if not audio_url:
            log("WARN", f"  {mid}: no download URL found")
            continue

        # Download the audio file
        try:
            log("INFO", f"  {mid}: downloading '{hit.get('title', 'unknown')}'...")
            audio_resp = requests.get(audio_url, timeout=60, stream=True)
            audio_resp.raise_for_status()

            with open(target_file, "wb") as f:
                for chunk in audio_resp.iter_content(chunk_size=8192):
                    f.write(chunk)

            size_kb = target_file.stat().st_size // 1024
            log("OK", f"  {mid}: downloaded ({size_kb} KB) → {target_file}")
            downloaded.append(str(target_file))
        except Exception as e:
            log("WARN", f"  {mid}: download failed: {e}")
            if target_file.exists():
                target_file.unlink()

    return downloaded


def get_available_music(music_dir: str | Path | None = None) -> list[dict[str, str]]:
    """Return music entries that have actual files available on disk.

    Args:
        music_dir: Directory containing music files.
                   If None, uses "music/" relative to cwd.
    """
    base = Path(music_dir) if music_dir else Path("music")
    available = []
    for entry in MUSIC_LIBRARY:
        file_path = base / entry["file"]
        if file_path.exists() and file_path.stat().st_size > 10_000:
            available.append({**entry, "file": str(file_path)})

    return available


def match_music_to_clip(
    clip: dict[str, Any],
    available_music: list[dict[str, str]],
    llm_model: str | None = None,
    api_key: str | None = None,
) -> dict[str, str] | None:
    """Use LLM to match the best background music to a clip's content.

    Args:
        clip: Clip metadata with topic, title, caption, hook, scores
        available_music: List of available music entries
        llm_model: LLM model override
        api_key: API key override

    Returns:
        Best matching music entry, or None
    """
    if not available_music:
        return None

    from .llm.backends import call_llm

    music_options = "\n".join([
        f"- {m['id']}: {m['description']} (mood: {m['mood']})"
        for m in available_music
    ])

    clip_info = (
        f"Title: {clip.get('title', '')}\n"
        f"Topic: {clip.get('topic', '')}\n"
        f"Hook: {clip.get('hook', '')}\n"
        f"Caption: {clip.get('caption', '')}\n"
        f"Scores — hook: {clip.get('score_hook', 0)}, "
        f"emotional: {clip.get('score_emotional_payoff', 0)}, "
        f"retention: {clip.get('score_retention', 0)}"
    )

    system = (
        "You are a music supervisor for short-form video content. "
        "Select the best background music that fits the clip's mood and content. "
        "The music should enhance the video without being distracting — it plays at very low volume. "
        "Return ONLY a JSON object with the key 'music_id' containing the selected music ID."
    )

    user = (
        f"Select the best background music for this clip:\n\n"
        f"CLIP:\n{clip_info}\n\n"
        f"AVAILABLE MUSIC:\n{music_options}\n\n"
        f"Return ONLY: {{\"music_id\": \"selected_id\"}}"
    )

    try:
        result = call_llm(system, user, api_key, llm_model, enable_reasoning=False)
        if result and isinstance(result, list) and len(result) > 0:
            music_id = result[0].get("music_id", "")
        elif result and isinstance(result, dict):
            music_id = result.get("music_id", "")
        else:
            music_id = ""

        if music_id:
            for m in available_music:
                if m["id"] == music_id:
                    return m

        # Fallback: return first available if LLM fails
        log("WARN", f"LLM music match returned unknown ID '{music_id}', using default")
    except Exception as e:
        log("WARN", f"LLM music match failed: {e}")

    return available_music[0] if available_music else None


def match_music_batch(
    clips: list[dict[str, Any]],
    available_music: list[dict[str, str]],
    llm_model: str | None = None,
    api_key: str | None = None,
) -> dict[int, dict[str, str]]:
    """Match music to multiple clips in a single LLM call.

    Returns:
        Dict mapping clip rank → music entry
    """
    if not available_music or not clips:
        return {}

    from .llm.backends import call_llm

    music_options = "\n".join([
        f"- {m['id']}: {m['description']} (mood: {m['mood']})"
        for m in available_music
    ])

    clips_info = []
    for c in clips:
        clips_info.append(
            f"Clip #{c.get('rank', 0)}: "
            f"title=\"{c.get('title', '')}\", "
            f"topic=\"{c.get('topic', '')}\", "
            f"hook=\"{c.get('hook', '')}\", "
            f"scores(hook={c.get('score_hook', 0)}, "
            f"emotional={c.get('score_emotional_payoff', 0)})"
        )

    system = (
        "You are a music supervisor for short-form video. "
        "Match background music to each clip. Music plays at very low volume as ambient vibe. "
        "Return ONLY a JSON array of objects with 'rank' and 'music_id' keys."
    )

    user = (
        f"Match background music for these clips:\n\n"
        f"CLIPS:\n" + "\n".join(clips_info) + "\n\n"
        f"AVAILABLE MUSIC:\n{music_options}\n\n"
        f"Return: [{{\"rank\": 1, \"music_id\": \"...\"}}, ...]"
    )

    result_map: dict[int, dict[str, str]] = {}

    try:
        result = call_llm(system, user, api_key, llm_model, enable_reasoning=False)
        if result and isinstance(result, list):
            music_by_id = {m["id"]: m for m in available_music}
            for item in result:
                rank = item.get("rank")
                mid = item.get("music_id", "")
                if rank is not None and mid in music_by_id:
                    result_map[int(rank)] = music_by_id[mid]
    except Exception as e:
        log("WARN", f"Batch music matching failed: {e}")

    # Fill in any missing clips with default
    default = available_music[0] if available_music else None
    if default:
        for c in clips:
            rank = c.get("rank", 0)
            if rank not in result_map:
                result_map[rank] = default

    return result_map


def build_music_filter(
    music_path: str,
    clip_duration: float,
    volume: float = 0.06,
) -> tuple[list[str], str]:
    """Build FFmpeg inputs and filter for mixing background music.

    The music is:
    - Looped if shorter than clip duration
    - Faded in over first 1s and out over last 2s
    - Mixed at very low volume (default 6% = barely audible vibe)
    - Trimmed to match clip duration

    Args:
        music_path: Path to background music file
        clip_duration: Duration of the clip in seconds
        volume: Music volume (0.0–1.0), default 0.06 (very quiet)

    Returns:
        (extra_inputs, filter_string) to add to FFmpeg command
    """
    fade_in = min(1.0, clip_duration * 0.1)
    fade_out = min(2.0, clip_duration * 0.15)
    fade_out_start = max(0, clip_duration - fade_out)

    extra_inputs = ["-stream_loop", "-1", "-i", music_path]

    # Music filter: trim, fade, volume, then mix with original audio
    music_filter = (
        f"[{{music_idx}}:a]"
        f"atrim=0:{clip_duration:.3f},"
        f"afade=t=in:st=0:d={fade_in:.2f},"
        f"afade=t=out:st={fade_out_start:.2f}:d={fade_out:.2f},"
        f"volume={volume:.3f}"
        f"[bgm]"
    )

    return extra_inputs, music_filter
