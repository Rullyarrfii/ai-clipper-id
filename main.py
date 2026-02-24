#!/usr/bin/env python3
"""
AI Video Clipper — Indonesian-optimized
────────────────────────────────────────
1. Transcribes video with faster-whisper (GPU/CPU auto, VAD, batched)
2. Pre-filters noise, fillers, duplicates (tuned for Bahasa Indonesia)
3. LLM automatically decides how many engaging clips to extract (max 100)
4. Extracts clips in parallel with ffmpeg

LLM priority: OpenRouter (free default) → Anthropic → OpenAI → Ollama

Usage:
  python main.py video.mp4
  python main.py video.mp4 --model large-v3
  python main.py video.mp4 --min 15 --max 90 --max-clips 50
  OPENROUTER_API_KEY=sk-... python main.py video.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ ANSI helpers ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
MAGENTA = "\033[95m"

_LEVEL_COLOR = {"INFO": CYAN, "OK": GREEN, "WARN": YELLOW, "ERROR": RED, "LLM": MAGENTA}


def log(level: str, msg: str) -> None:
    c = _LEVEL_COLOR.get(level, RESET)
    print(f"{c}{BOLD}[{level}]{RESET} {msg}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━ CONFIG ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Default free model on OpenRouter — change via --llm-model or OPENROUTER_MODEL
DEFAULT_OPENROUTER_MODEL = "arcee-ai/trinity-large-preview:free"
DEFAULT_OPENROUTER_BASE  = "https://openrouter.ai/api/v1"

MAX_CLIPS_HARD_LIMIT = 100  # absolute ceiling no matter what

# Indonesian filler / noise patterns (comprehensive)
_ID_FILLERS = (
    # particles & interjections
    r"eh|ah|oh|uh|um|uhm|em|hm|hmm|mm|mmm|"
    r"anu|apa ya|ya|iya|yak|yah|yoi|oke|oks|ok|"
    r"nah|lah|deh|nih|tuh|sih|dong|deh|kan|kok|"
    # common verbal tics
    r"gitu|gini|kayak gitu|kayak gini|gitulah|gitu lho|"
    r"maksudnya|pokoknya|intinya|sebentar|"
    r"jadi|terus|trus|nah terus|ya kan|"
    r"gimana ya|apa namanya|apa sih|aduh|astaga|"
    r"wah|wow|duh|hah|lho|loh|"
    # English fillers (common in Indonesian content)
    r"like|you know|i mean|so|right|okay|basically|literally|actually|anyway|alright"
)

FILLER_RE = re.compile(rf"^({_ID_FILLERS})\W*$", re.IGNORECASE)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━ STEP 1 — Transcription ━━━━━━━━━━━━━━━━━━━━━━━━━


def _detect_device() -> str:
    """Return the best available device (cuda if available, else cpu)."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _resolve_compute_type(device: str, compute_type: str) -> str:
    """Resolve compute type with safe CPU/GPU defaults."""
    if compute_type == "auto":
        return "float16" if device == "cuda" else "int8"
    if device == "cpu" and compute_type in {"float16", "int8_float16"}:
        log("WARN", f"compute_type={compute_type} not supported on CPU; using int8")
        return "int8"
    return compute_type


def transcribe(
    video_path: str,
    model_size: str = "large-v3",
    language: str | None = "id",
    device: str = "auto",
    compute_type: str = "auto",
    vad_filter: bool = True,
    vad_min_silence_ms: int = 400,
    vad_speech_pad_ms: int = 200,
    batch_size: int = 16,
) -> list[dict]:
    """
    Transcribe *video_path* with faster-whisper.

    Returns list of segment dicts:
      [{"start": float, "end": float, "text": str, "words": [...], "no_speech_prob": float}]

    Tuned for Indonesian:
    - language defaults to "id" (skip auto-detect overhead)
    - VAD aggressively filters silence common in podcast/talk formats
    - word_timestamps for precise cut boundaries
    - batch_size > 1 uses BatchedInferencePipeline
    """
    try:
        from faster_whisper import WhisperModel, BatchedInferencePipeline
    except ImportError:
        log("ERROR", "faster-whisper not installed.  pip install faster-whisper")
        sys.exit(1)

    if device == "auto":
        device = _detect_device()

    compute_type = _resolve_compute_type(device, compute_type)

    log("INFO", f"Loading whisper {BOLD}{model_size}{RESET} on {device.upper()} ({compute_type})")
    t0 = time.time()
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        num_workers=min(4, os.cpu_count() or 1),
    )
    log("OK", f"Model loaded in {time.time() - t0:.1f}s")

    log("INFO", f"Transcribing {BOLD}{Path(video_path).name}{RESET} …")
    t0 = time.time()

    condition_prev = False if model_size.startswith("distil-") else True

    transcribe_kwargs = dict(
        language=language,
        word_timestamps=True,
        vad_filter=vad_filter,
        beam_size=5,
        best_of=5,
        temperature=0.0,
        condition_on_previous_text=condition_prev,
    )

    if vad_filter:
        transcribe_kwargs["vad_parameters"] = {
            "min_silence_duration_ms": vad_min_silence_ms,
            "speech_pad_ms": vad_speech_pad_ms,
        }

    if batch_size > 1:
        batched = BatchedInferencePipeline(model=model)
        segments_gen, info = batched.transcribe(
            video_path,
            batch_size=batch_size,
            **transcribe_kwargs,
        )
    else:
        segments_gen, info = model.transcribe(
            video_path,
            **transcribe_kwargs,
        )

    segments: list[dict] = []
    for seg in segments_gen:
        words = (
            [{"word": w.word, "start": w.start, "end": w.end} for w in seg.words]
            if seg.words else []
        )
        segments.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
            "words": words,
            "no_speech_prob": round(seg.no_speech_prob, 4),
        })

    elapsed = time.time() - t0
    duration = segments[-1]["end"] if segments else 0.0
    rtf = elapsed / duration if duration else 0
    log("OK",
        f"{duration:.0f}s audio → {len(segments)} segments "
        f"| RTF {rtf:.2f}x | {elapsed:.1f}s")
    if info.language:
        log("INFO", f"Language: {info.language} (p={info.language_probability:.0%})")

    return segments


# ━━━━━━━━━━━━━━━━━━━━━━━━━━ STEP 2 — Pre-filter ━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def _wps(seg: dict) -> float:
    d = seg["end"] - seg["start"]
    return len(seg["text"].split()) / d if d > 0 else 0.0


def _jaccard(a: str, b: str) -> float:
    sa, sb = set(a.lower().split()), set(b.lower().split())
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def prefilter_segments(
    segments: list[dict],
    *,
    min_words: int = 4,
    min_duration: float = 1.0,
    max_no_speech: float = 0.6,
    min_wps: float = 0.4,
    max_wps: float = 9.0,
    dup_threshold: float = 0.8,
    merge_gap: float = 1.5,
) -> tuple[list[dict], dict[str, Any]]:
    """
    Filter low-value segments before LLM analysis.

    Filters: too-short, too-few-words, high no-speech, pure filler,
    abnormal speech rate, near-duplicate. Then merges close neighbours.
    """
    kept: list[dict] = []
    reasons: dict[str, int] = {}
    seen: list[str] = []
    n_dropped = 0

    def _drop(reason: str) -> None:
        nonlocal n_dropped
        n_dropped += 1
        reasons[reason] = reasons.get(reason, 0) + 1

    for seg in segments:
        txt = seg["text"].strip()
        dur = seg["end"] - seg["start"]
        wps = _wps(seg)
        nsp = seg.get("no_speech_prob", 0.0)

        if dur < min_duration:          _drop("short_dur"); continue
        if len(txt.split()) < min_words:_drop("few_words"); continue
        if nsp > max_no_speech:         _drop("no_speech"); continue
        if FILLER_RE.match(txt):        _drop("filler"); continue
        if wps < min_wps:               _drop("slow_speech"); continue
        if wps > max_wps:               _drop("hallucination"); continue
        if any(_jaccard(txt, p) >= dup_threshold for p in seen[-12:]):
            _drop("duplicate"); continue

        seen.append(txt)
        kept.append(seg)

    # Merge adjacent segments separated by tiny gaps
    merged: list[dict] = []
    for seg in kept:
        if merged and (seg["start"] - merged[-1]["end"]) <= merge_gap:
            prev = merged[-1]
            prev["end"] = seg["end"]
            prev["text"] += " " + seg["text"]
            prev["words"] = prev.get("words", []) + seg.get("words", [])
        else:
            merged.append(dict(seg))

    stats: dict[str, Any] = {
        "original": len(segments),
        "kept": len(merged),
        "dropped": n_dropped,
        "drop_pct": f"{n_dropped / len(segments) * 100:.1f}%" if segments else "0%",
        "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }
    return merged, stats


# ━━━━━━━━━━━━━━━━━━━━━━━━━━ STEP 3 — LLM analysis ━━━━━━━━━━━━━━━━━━━━━━━━━━

_SYSTEM_PROMPT = """\
Kamu adalah editor video profesional dan ahli strategi media sosial Indonesia.

Tugasmu: analisis transkrip video dan temukan SEMUA segmen menarik yang cocok
untuk klip pendek viral (TikTok, YouTube Shorts, Instagram Reels).

PENTING — tentukan sendiri berapa banyak klip yang layak. Jangan memaksakan
klip yang tidak menarik. Kembalikan antara 0 hingga {max_clips} klip.

Kriteria penilaian (engagement_score 0-100):
• Hook kuat di awal — penonton harus langsung tertarik dalam 3 detik pertama
• Puncak emosi — lucu, mengejutkan, inspiratif, kontroversial, mengharukan
• Cerita / argumen yang utuh dan mandiri (self-contained)
• Kalimat quotable / memorable yang bisa jadi caption
• Hindari potongan di tengah kalimat — potong di jeda alami
• Konten yang relatable untuk audiens Indonesia
• Skor >= {min_score} berarti layak untuk diposting

Constraint:
• Setiap klip harus berdurasi {min_dur}–{max_dur} detik
• Tidak boleh ada klip yang overlap satu sama lain
• start/end harus menggunakan timestamp dari transkrip (detik, float)
• Mulai klip di batas kalimat/frase yang alami
• Urutkan berdasarkan engagement_score tertinggi (rank 1 = terbaik)

Format output: HANYA JSON array valid, tanpa penjelasan, tanpa markdown fence.

Schema per item:
{{"rank": 1, "start": 12.4, "end": 45.7, "title": "Judul pendek menarik", \
"reason": "Alasan 1 kalimat kenapa menarik", "hook": "Kalimat pembuka klip", \
"engagement_score": 92}}
"""


def _build_transcript_text(segments: list[dict]) -> str:
    """Compact transcript: [start → end] text"""
    return "\n".join(
        f"[{s['start']:.1f}→{s['end']:.1f}] {s['text']}" for s in segments
    )


def _build_user_prompt(
    transcript: str,
    min_dur: int,
    max_dur: int,
    max_clips: int,
    min_score: int,
) -> str:
    return (
        f"Analisis transkrip berikut dan temukan semua klip menarik.\n"
        f"Durasi per klip: {min_dur}–{max_dur} detik. Maksimum {max_clips} klip.\n"
        f"Kembalikan hanya klip yang benar-benar engaging (score >= {min_score}).\n\n"
        f"TRANSKRIP:\n{transcript}"
    )


def _parse_llm_json(raw: str) -> list[dict]:
    """Robustly extract a JSON array from LLM text."""
    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    # Try direct parse
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        # Wrapped in an object — find the array
        for v in data.values():
            if isinstance(v, list):
                return v
        return []
    except json.JSONDecodeError:
        pass
    # Last resort: find first [ ... ] block
    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if m:
        return json.loads(m.group())
    log("ERROR", "Could not parse JSON from LLM response")
    return []


def _validate_clips(
    clips: list[dict],
    min_dur: int,
    max_dur: int,
    max_clips: int,
    min_score: int,
    video_duration: float | None = None,
) -> list[dict]:
    """Sanitize, deduplicate, and cap the clip list."""
    valid: list[dict] = []
    seen_ranges: list[tuple[float, float]] = []

    for c in clips:
        try:
            s, e = float(c["start"]), float(c["end"])
        except (KeyError, ValueError, TypeError):
            continue
        dur = e - s
        if dur < min_dur * 0.8 or dur > max_dur * 1.2:
            continue  # allow 20% tolerance
        if video_duration and e > video_duration + 1:
            continue
        score = int(c.get("engagement_score", 0) or 0)
        if score < min_score:
            continue
        # Overlap check
        overlaps = any(not (e <= rs or s >= re) for rs, re in seen_ranges)
        if overlaps:
            continue
        seen_ranges.append((s, e))
        # Ensure required fields
        c.setdefault("rank", len(valid) + 1)
        c.setdefault("title", f"Clip {c['rank']}")
        c.setdefault("engagement_score", score)
        c.setdefault("reason", "")
        c.setdefault("hook", "")
        valid.append(c)
        if len(valid) >= max_clips:
            break

    # Re-rank by score
    valid.sort(key=lambda x: -x.get("engagement_score", 0))
    for i, c in enumerate(valid, 1):
        c["rank"] = i

    return valid


# ──────── LLM backends ────────


def _llm_openrouter(
    system: str,
    user: str,
    api_key: str,
    model: str = DEFAULT_OPENROUTER_MODEL,
    base_url: str = DEFAULT_OPENROUTER_BASE,
) -> list[dict]:
    """Call OpenRouter (OpenAI-compatible API). Default: free model."""
    try:
        from openai import OpenAI
    except ImportError:
        log("ERROR", "openai SDK not installed.  pip install openai")
        sys.exit(1)

    log("LLM", f"OpenRouter → {BOLD}{model}{RESET}")
    client = OpenAI(api_key=api_key, base_url=base_url)

    resp = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        temperature=0.3,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        extra_headers={
            "HTTP-Referer": "https://github.com/ai-video-clipper",
            "X-Title": "AI Video Clipper",
        },
    )
    raw = resp.choices[0].message.content or ""
    clips = _parse_llm_json(raw)
    log("OK", f"OpenRouter returned {len(clips)} clips")
    return clips


def _llm_anthropic(system: str, user: str, api_key: str) -> list[dict]:
    try:
        import anthropic
    except ImportError:
        log("ERROR", "anthropic SDK not installed.  pip install anthropic")
        sys.exit(1)

    log("LLM", f"Anthropic → {BOLD}claude-sonnet-4-20250514{RESET}")
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    raw = resp.content[0].text
    clips = _parse_llm_json(raw)
    log("OK", f"Anthropic returned {len(clips)} clips")
    return clips


def _llm_openai(system: str, user: str, api_key: str) -> list[dict]:
    try:
        from openai import OpenAI
    except ImportError:
        log("ERROR", "openai SDK not installed.  pip install openai")
        sys.exit(1)

    log("LLM", f"OpenAI → {BOLD}gpt-4o{RESET}")
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    raw = resp.choices[0].message.content or ""
    clips = _parse_llm_json(raw)
    log("OK", f"OpenAI returned {len(clips)} clips")
    return clips


def _llm_ollama(system: str, user: str) -> list[dict]:
    try:
        import requests
    except ImportError:
        log("ERROR", "requests not installed.  pip install requests")
        sys.exit(1)

    model = os.getenv("OLLAMA_MODEL", "llama3.1")
    log("LLM", f"Ollama (local) → {BOLD}{model}{RESET}")
    resp = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=300,
    )
    resp.raise_for_status()
    raw = resp.json()["message"]["content"]
    clips = _parse_llm_json(raw)
    log("OK", f"Ollama returned {len(clips)} clips")
    return clips


def find_clips(
    segments: list[dict],
    *,
    min_duration: int = 15,
    max_duration: int = 60,
    max_clips: int = MAX_CLIPS_HARD_LIMIT,
    min_score: int = 60,
    llm_model: str | None = None,
    api_key: str | None = None,
    video_duration: float | None = None,
) -> list[dict]:
    """
    Ask LLM to find ALL engaging clips (up to *max_clips*).

    Backend priority:
      1. OpenRouter  (OPENROUTER_API_KEY) — default free model
      2. Anthropic   (ANTHROPIC_API_KEY)
      3. OpenAI      (OPENAI_API_KEY)
      4. Ollama      (local, no key needed)
    """
    transcript = _build_transcript_text(segments)
    system = _SYSTEM_PROMPT.format(
        min_dur=min_duration,
        max_dur=max_duration,
        max_clips=max_clips,
        min_score=min_score,
    )
    user = _build_user_prompt(transcript, min_duration, max_duration, max_clips, min_score)

    # ── Resolve backend ──────────────────────────────────────────────────────
    or_key = api_key or os.getenv("OPENROUTER_API_KEY")
    ant_key = os.getenv("ANTHROPIC_API_KEY")
    oai_key = os.getenv("OPENAI_API_KEY")

    raw_clips: list[dict] = []

    if or_key:
        model = llm_model or os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL)
        raw_clips = _llm_openrouter(system, user, or_key, model=model)
    elif ant_key:
        raw_clips = _llm_anthropic(system, user, ant_key)
    elif oai_key:
        raw_clips = _llm_openai(system, user, oai_key)
    else:
        log("WARN", "No API key found. Trying local Ollama …")
        raw_clips = _llm_ollama(system, user)

    # Validate & clean
    clips = _validate_clips(
        raw_clips,
        min_duration,
        max_duration,
        max_clips,
        min_score,
        video_duration,
    )
    if not clips:
        log("WARN", "LLM returned 0 valid clips. The video may not have engaging segments.")
    else:
        log("OK", f"{len(clips)} valid clips after validation")
    return clips


# ━━━━━━━━━━━━━━━━━━━━━━━━━━ STEP 4 — Clip extraction ━━━━━━━━━━━━━━━━━━━━━━━


def _get_transcript_cache_path(video_path: str) -> Path:
    """Return cache path for transcript based on video filename."""
    video = Path(video_path)
    cache_dir = Path.cwd() / ".cache" / "ai-video-clipper"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{video.stem}_transcript.json"
    return cache_file


def _get_video_duration(path: str) -> float | None:
    """Get video duration in seconds via ffprobe."""
    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True, text=True, check=True,
        )
        return float(out.stdout.strip())
    except Exception:
        return None


def _extract_one(
    video_path: str,
    clip: dict,
    output_dir: Path,
    *,
    padding: float = 0.15,
) -> str:
    """Extract a single clip with ffmpeg. Returns output file path."""
    start = max(0.0, clip["start"] - padding)
    end = clip["end"] + padding
    duration = end - start

    safe = re.sub(r"[^\w\s-]", "", clip.get("title", f"clip_{clip['rank']}"))
    safe = re.sub(r"\s+", "_", safe)[:50]
    out_path = output_dir / f"rank{clip['rank']:02d}_{safe}.mp4"

    # Try hardware encode first on macOS (VideoToolbox), then NVENC, then CPU
    encode_attempts = [
        # macOS VideoToolbox
        ["-c:v", "h264_videotoolbox", "-b:v", "5M"],
        # NVIDIA
        ["-c:v", "h264_nvenc", "-preset", "fast", "-crf", "23"],
        # CPU fallback
        ["-c:v", "libx264", "-preset", "fast", "-crf", "23"],
    ]

    base_cmd = [
        "ffmpeg", "-y", "-hide_banner",
        "-ss", f"{start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-map", "0:v:0",
        "-map", "0:a:0?",
        "-shortest",
    ]
    audio_flags = ["-c:a", "aac", "-b:a", "128k"]
    output_flags = ["-movflags", "+faststart", "-loglevel", "error", str(out_path)]

    for enc in encode_attempts:
        cmd = base_cmd + enc + audio_flags + output_flags
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return str(out_path)
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue

    raise RuntimeError(f"All ffmpeg encode strategies failed for clip #{clip['rank']}")


def extract_clips(
    video_path: str,
    clips: list[dict],
    output_dir: Path,
    max_workers: int = 4,
) -> list[str]:
    """Extract all clips in parallel."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log("INFO", f"Extracting {len(clips)} clips → {output_dir}/ ({max_workers} workers)")

    results: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {
            pool.submit(_extract_one, video_path, c, output_dir): c
            for c in clips
        }
        for fut in as_completed(futs):
            clip = futs[fut]
            try:
                out = fut.result()
                mb = os.path.getsize(out) / 1_048_576
                log("OK", f"  #{clip['rank']:>2} {clip['title'][:40]:<40}  "
                          f"{Path(out).name} ({mb:.1f} MB)")
                results.append(out)
            except Exception as exc:
                log("ERROR", f"  #{clip['rank']:>2} failed: {exc}")

    return sorted(results)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━ CLI ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def main() -> None:
    ap = argparse.ArgumentParser(
        description="AI Video Clipper — Indonesian-optimized, auto clip count",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Environment variables:\n"
            "  OPENROUTER_API_KEY   OpenRouter key (default backend, free model)\n"
            "  OPENROUTER_MODEL     Override default model on OpenRouter\n"
            "  ANTHROPIC_API_KEY    Anthropic Claude key\n"
            "  OPENAI_API_KEY       OpenAI key\n"
            "  OLLAMA_MODEL         Local Ollama model name (default: llama3.1)\n"
        ),
    )
    ap.add_argument("video", help="Path to input video")
    ap.add_argument("--model", default="small",
                    choices=["tiny", "base", "small", "medium",
                             "large-v2", "large-v3", "distil-large-v3", "turbo"],
                    help="Whisper model size (default: tiny)")
    ap.add_argument("--lang", default="id",
                    help="Language code — 'id' Indonesian, 'en' English, "
                         "or None for auto-detect (default: id)")
    ap.add_argument("--min", type=int, default=15,
                    help="Min clip duration in seconds (default: 15)")
    ap.add_argument("--max", type=int, default=60,
                    help="Max clip duration in seconds (default: 60)")
    ap.add_argument("--max-clips", type=int, default=MAX_CLIPS_HARD_LIMIT,
                    help=f"Maximum number of clips (default: {MAX_CLIPS_HARD_LIMIT})")
    ap.add_argument("--min-score", type=int, default=60,
                    help="Minimum engagement score to keep a clip (default: 60)")
    ap.add_argument("--device", default="auto",
                    choices=["auto", "cuda", "cpu"],
                    help="Compute device (default: auto)")
    ap.add_argument("--compute-type", default="auto",
                    choices=["auto", "float16", "int8", "int8_float16"],
                    help="Compute type (default: auto)")
    ap.add_argument("--no-vad", action="store_true",
                    help="Disable VAD filtering for transcription")
    ap.add_argument("--vad-min-silence", type=int, default=400,
                    help="VAD min silence duration in ms (default: 400)")
    ap.add_argument("--vad-speech-pad", type=int, default=200,
                    help="VAD speech padding in ms (default: 200)")
    ap.add_argument("--batch", type=int, default=16,
                    help="Whisper batch size (default: 16; lower if OOM)")
    ap.add_argument("--workers", type=int, default=4,
                    help="Parallel ffmpeg workers (default: 4)")
    ap.add_argument("--output", default="clips",
                    help="Output directory (default: ./clips)")
    ap.add_argument("--save-transcript", action="store_true",
                    help="Save full transcript JSON")
    ap.add_argument("--api-key", default=None,
                    help="API key (overrides env vars)")
    ap.add_argument("--llm-model", default=None,
                    help="Override LLM model name for OpenRouter")

    args = ap.parse_args()
    lang = None if args.lang.lower() == "none" else args.lang

    video = Path(args.video)
    if not video.exists():
        log("ERROR", f"File not found: {video}")
        sys.exit(1)

    output_dir = Path(args.output)

    print(f"\n{BOLD}{CYAN}{'═' * 50}")
    print(f"   AI Video Clipper — Indonesian-optimized")
    print(f"{'═' * 50}{RESET}")
    print(f"  Video     : {video.name}")
    print(f"  Model     : {args.model}")
    print(f"  Language  : {lang or 'auto-detect'}")
    print(f"  Duration  : {args.min}–{args.max}s per clip")
    print(f"  Max clips : {args.max_clips} (LLM decides actual count)")
    print()

    # ── 1. Transcribe ────────────────────────────────────────────────────────
    t_total = time.time()
    
    # Check for cached transcript
    cache_path = _get_transcript_cache_path(str(video))
    if cache_path.exists():
        log("INFO", f"Loading cached transcript from {cache_path}")
        segments = json.loads(cache_path.read_text())
        log("OK", f"Loaded {len(segments)} segments from cache")
    else:
        segments = transcribe(
            str(video),
            model_size=args.model,
            language=lang,
            device=args.device,
            compute_type=args.compute_type,
            vad_filter=not args.no_vad,
            vad_min_silence_ms=args.vad_min_silence,
            vad_speech_pad_ms=args.vad_speech_pad,
            batch_size=args.batch,
        )
        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(segments, indent=2, ensure_ascii=False))
        log("OK", f"Transcript cached → {cache_path}")

    if args.save_transcript:
        output_dir.mkdir(parents=True, exist_ok=True)
        tx = output_dir / "transcript.json"
        tx.write_text(json.dumps(segments, indent=2, ensure_ascii=False))
        log("OK", f"Transcript → {tx}")

    # ── 2. Pre-filter ────────────────────────────────────────────────────────
    filtered, stats = prefilter_segments(segments)

    print(f"\n{BOLD}Pre-filter:{RESET}")
    print(f"  {stats['original']} → {stats['kept']} segments "
          f"({stats['dropped']} dropped, {stats['drop_pct']})")
    if stats["reasons"]:
        print(f"  Reasons: {', '.join(f'{k}={v}' for k, v in stats['reasons'].items())}")
    print()

    if not filtered:
        log("ERROR", "All segments filtered out. Try a different whisper model or looser filters.")
        sys.exit(1)

    # ── 3. LLM analysis ─────────────────────────────────────────────────────
    video_dur = _get_video_duration(str(video))
    clips = find_clips(
        filtered,
        min_duration=args.min,
        max_duration=args.max,
        max_clips=min(args.max_clips, MAX_CLIPS_HARD_LIMIT),
        min_score=args.min_score,
        llm_model=args.llm_model,
        api_key=args.api_key,
        video_duration=video_dur,
    )

    if not clips:
        log("WARN", "No engaging clips found. Exiting.")
        sys.exit(0)

    # Summary table
    print(f"\n{BOLD}{'#':<4} {'Score':<6} {'Start':>7} {'End':>7} {'Dur':>5}  Title{RESET}")
    print("─" * 70)
    for c in clips:
        d = c["end"] - c["start"]
        print(f"  {c['rank']:<3} {c.get('engagement_score', '?'):<6} "
              f"{c['start']:>7.1f} {c['end']:>7.1f} {d:>4.0f}s  {c['title']}")
    print()

    # ── 4. Extract ───────────────────────────────────────────────────────────
    outputs = extract_clips(
        str(video),
        clips,
        output_dir=output_dir,
        max_workers=args.workers,
    )

    # Save metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = output_dir / "clips.json"
    meta.write_text(json.dumps(clips, indent=2, ensure_ascii=False))

    elapsed_total = time.time() - t_total
    print(f"\n{GREEN}{BOLD}✓ Done!{RESET} "
          f"{len(outputs)}/{len(clips)} clips extracted → {output_dir}/ "
          f"({elapsed_total:.0f}s total)")
    print(f"  Metadata → {meta}")


if __name__ == "__main__":
    main()
