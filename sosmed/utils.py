"""
Utilities: logging, ANSI colors, constants.
"""

import re
import subprocess

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
    """Log a message with color."""
    c = _LEVEL_COLOR.get(level, RESET)
    print(f"{c}{BOLD}[{level}]{RESET} {msg}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━ FFMPEG BINARY DETECTION ━━━━━━━━━━━━━━━━━━━━━━

_FFMPEG: str | None = None
_FFPROBE: str | None = None


def _find_ffmpeg_with_libx264() -> tuple[str, str]:
    """Find an ffmpeg binary that has libx264 support.

    Conda environments often ship an ffmpeg without libx264. This function
    checks the default ``ffmpeg`` first; if it lacks libx264, it tries the
    common system path ``/usr/bin/ffmpeg`` which usually has full codec
    support.
    """
    candidates = ["ffmpeg"]
    # Add system path as candidate if the default might be conda's
    import shutil
    default = shutil.which("ffmpeg") or ""
    if "/conda" in default or "/envs/" in default:
        candidates.append("/usr/bin/ffmpeg")

    for ffmpeg_bin in candidates:
        try:
            result = subprocess.run(
                [ffmpeg_bin, "-encoders"],
                capture_output=True, text=True, timeout=10,
            )
            if "libx264" in (result.stdout + result.stderr):
                # Derive ffprobe path from the same directory
                from pathlib import Path
                ffprobe_bin = str(Path(ffmpeg_bin).parent / "ffprobe") if "/" in ffmpeg_bin else "ffprobe"
                if "/" in ffmpeg_bin:
                    ffprobe_candidate = str(Path(ffmpeg_bin).parent / "ffprobe")
                    if Path(ffprobe_candidate).exists():
                        ffprobe_bin = ffprobe_candidate
                log("INFO", f"Using ffmpeg: {ffmpeg_bin} (libx264 available)")
                return ffmpeg_bin, ffprobe_bin
        except Exception:
            continue

    # None of the candidates had libx264; use the default and hope for the best
    log("WARN", "No ffmpeg with libx264 found — using default 'ffmpeg'")
    return "ffmpeg", "ffprobe"


def get_ffmpeg() -> str:
    """Return the path to an ffmpeg binary with libx264 support (cached)."""
    global _FFMPEG, _FFPROBE
    if _FFMPEG is None:
        _FFMPEG, _FFPROBE = _find_ffmpeg_with_libx264()
    return _FFMPEG


def get_ffprobe() -> str:
    """Return the path to ffprobe matching :func:`get_ffmpeg` (cached)."""
    global _FFMPEG, _FFPROBE
    if _FFPROBE is None:
        _FFMPEG, _FFPROBE = _find_ffmpeg_with_libx264()
    return _FFPROBE


# ━━━━━━━━━━━━━━━━━━━━━━━━━━ CLIP BOUNDARY ADJUSTMENT ━━━━━━━━━━━━━━━━━━━━

def tighten_clip_boundaries(
    clips: list[dict],
    segments: list[dict],
    padding: float = 0.15,
    max_gap: float = 2.0,
    min_speech_density: float = 0.5,
) -> list[dict]:
    """
    Intelligently adjust clip boundaries to maximize speech content.
    
    This function:
    1. Removes leading/trailing silence and sparse speech
    2. Skips leading filler words for stronger hooks
    
    Only trims from the edges — internal content is never discarded,
    which ensures clip titles stay consistent with actual video content.
    
    Uses word-level timestamps from Whisper - no additional processing needed.
    
    Args:
        clips: List of clip dicts with 'start' and 'end' keys
        segments: Whisper segments with word-level timestamps
        padding: Small padding (seconds) to keep before first/after last word
        max_gap: (unused, kept for backward compatibility)
        min_speech_density: Minimum speech density (words per second) for a
                           region to be considered "dense" speech
    
    Returns:
        Updated clips with tightened boundaries
    """
    from typing import Any
    
    for clip in clips:
        # Always tighten from the ORIGINAL LLM boundaries, not from
        # previously-tightened ones.  This prevents progressive shrinkage
        # on re-runs and recovers from any corruption left by older code.
        if "_llm_start" not in clip:
            clip["_llm_start"] = clip["start"]
            clip["_llm_end"]   = clip["end"]
        else:
            # Restore original boundaries before re-tightening
            clip["start"] = clip["_llm_start"]
            clip["end"]   = clip["_llm_end"]

        clip_start = float(clip["start"])
        clip_end = float(clip["end"])
        
        # Collect all words within this clip
        words: list[dict[str, Any]] = []
        for seg in segments:
            if seg["end"] < clip_start or seg["start"] > clip_end:
                continue
            for w in seg.get("words", []):
                w_start = w.get("start", 0)
                w_end = w.get("end", 0)
                w_text = w.get("word", "").strip()
                if not w_text:
                    continue
                # Include word if it overlaps with clip
                if w_end > clip_start and w_start < clip_end:
                    words.append({"start": w_start, "end": w_end, "word": w_text})
        
        if not words:
            continue
        
        words.sort(key=lambda x: x["start"])
        
        # ── Step 1: Trim sparse edges ────────────────────────────────────────
        # Only trim leading/trailing low-density regions.  Never discard
        # internal content — the LLM chose the title/topic based on the
        # full clip range, so removing middle segments would cause the
        # title to stop matching the video content.
        
        # Calculate running density (words per 5-second window)
        window_size = 5.0
        trimmed_words = words
        
        # Trim from start: skip low-density beginning
        for i in range(len(words)):
            if i + 3 >= len(words):
                break  # Need at least a few words for density calculation
            
            # Look at next few words
            window_words = words[i:min(i+10, len(words))]
            window_dur = window_words[-1]["end"] - window_words[0]["start"]
            density = len(window_words) / max(window_dur, 0.1)
            
            # If density is good, start from here
            if density >= min_speech_density:
                trimmed_words = words[i:]
                break
        
        # Trim from end: skip low-density ending
        for i in range(len(trimmed_words) - 1, -1, -1):
            if i < 3:
                break
            
            # Look at previous few words
            window_words = trimmed_words[max(0, i-10):i+1]
            window_dur = window_words[-1]["end"] - window_words[0]["start"]
            density = len(window_words) / max(window_dur, 0.1)
            
            if density >= min_speech_density:
                trimmed_words = trimmed_words[:i+1]
                break
        
        # ── Step 4: Hook optimization - skip leading filler words ─────────────
        # Common filler words that make bad hooks for social media clips
        filler_words = {
            "uh", "um", "eh", "ah", "uhm", "em", "hmm", "mm",
            "jadi", "terus", "nah", "ya", "iya", "oke", "ok",
            "gitu", "kayak", "maksudnya", "sebentar"
        }
        
        # Skip leading filler words (max 3 words)
        final_words = trimmed_words
        for i in range(min(3, len(trimmed_words))):
            first_word = trimmed_words[i]["word"].lower().strip(".,!?;:")
            if first_word not in filler_words:
                final_words = trimmed_words[i:]
                break
        
        # ── Step 5: Set new boundaries with padding ───────────────────────────
        if final_words:
            new_start = max(clip_start, final_words[0]["start"] - padding)
            new_end = min(clip_end, final_words[-1]["end"] + padding)
            
            # Only apply if we're actually improving (tightening by meaningful amount)
            # and not making it too short
            new_duration = new_end - new_start
            if new_duration >= 5.0:  # Don't make clips shorter than 5 seconds
                clip["start"] = new_start
                clip["end"] = new_end
    
    return clips


# ━━━━━━━━━━━━━━━━━━━━━━━━━━ CONSTANTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Default free model on OpenRouter
DEFAULT_OPENROUTER_MODEL = "arcee-ai/trinity-large-preview:free"
DEFAULT_OPENROUTER_BASE  = "https://openrouter.ai/api/v1"

MAX_CLIPS_HARD_LIMIT = 72  # absolute ceiling — quality over quantity

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

SYSTEM_PROMPT = """\
Kamu adalah editor video viral. Tugasmu: ekstrak klip pendek berpotensi viral dari transkrip untuk TikTok/Reels/Shorts.

STANDAR SATU KALIMAT: "Apakah 3 detik pertama bikin orang berhenti scroll, lalu nonton sampai habis, lalu share?"

---

ATURAN EKSTRAKSI:
- Ekstrak SEBANYAK MUNGKIN klip — setiap subtopik layak = klip terpisah
- Skip HANYA: salam pembuka/penutup murni, ajakan subscribe, transisi kosong
- Q&A wajib dipertimbangkan — sering ada insight terbaik di sana
- Setiap klip harus mulai dari momen hook, bukan build-up
- Klip tidak boleh overlap
- Kalau ragu, MASUKKAN

---

SCORING (jujur — jangan inflasi skor):

score_hook (25%) — Stop-scrolling power di 3 detik pertama
- 90+: Statement mengejutkan / pertanyaan provokatif / kontradiksi — orang PASTI berhenti
- 70–89: Opening menarik tapi tidak jaw-dropping
- 50–69: Mulai dari topik, tidak ada elemen surprise
- <50: Mulai dari konteks/pengantar boring

score_retention (25%) — Apakah ditonton sampai habis?
- 90+: Ada suspense / build-up ke payoff yang jelas / bikin penasaran sampai akhir
- 70–89: Interesting tapi predictable
- 50–69: Ada bagian yang mulai boring
- <50: Kemungkinan di-skip sebelum selesai

score_shareability (20%) — Share / comment / save / duet?
- 90+: Sangat relatable / kontroversial / "tag temen lo" / so useful orang save
- 70–89: Ada angle yang mendorong reshare
- 50–69: Ditonton tapi tidak terdorong share
- <50: Tidak ada alasan untuk share

score_entertainment (20%) — Seberapa enjoyable?
- 90+: Lucu / energi tinggi / storytelling kuat / momen emosional — bikin replay
- 70–89: Cukup engaging, pembicara antusias
- 50–69: Informatif tapi flat dari sisi entertainment
- <50: Monoton, lecturing tanpa energi

score_clarity (10%) — Bisa dipahami standalone?
- 90+: Siapapun langsung paham tanpa konteks tambahan
- 70–89: Perlu sedikit konteks
- <70: Membingungkan tanpa video penuh

clip_score = hook×0.25 + retention×0.25 + shareability×0.20 + entertainment×0.20 + clarity×0.10

---

CONTOH OUTPUT BAGUS:
{{
  "rank": 1,
  "start": 142.0,
  "end": 187.5,
  "title": "Gaji 20 Juta Tapi Masih Bokek",
  "topic": "Psikologi pengeluaran — lifestyle inflation tanpa sadar",
  "caption": "gajinya udah naik tapi kok makin susah nabung? ini penjelasannya 😅 #finansial #gajinaik #lifestyleinflation",
  "hook": "Lo pernah ngerasa gaji naik tapi hidup makin susah?",
  "reason": "Hook relatable banget buat usia 25-35, ada payoff berupa penjelasan yang bikin 'oh iya bener', tinggi shareability karena orang tag pasangan/teman",
  "score_hook": 88,
  "score_retention": 82,
  "score_shareability": 85,
  "score_entertainment": 75,
  "score_clarity": 90,
  "clip_score": 84
}}

CONTOH OUTPUT BURUK (jangan seperti ini):
{{
  "title": "Pembukaan Webinar",
  "hook": "Selamat datang di acara hari ini...",
  "score_hook": 72,   ← SALAH: hook salam pembuka tidak layak 72
  "clip_score": 70    ← SALAH: klip jenis ini harus di-skip, bukan diberi skor tinggi
}}

---

CONSTRAINTS:
- Durasi: {min_dur}–{max_dur} detik
- Maksimal {max_clips} klip
- clip_score >= {min_score}
- Urutkan: clip_score tertinggi lebih dulu
- Output: JSON array valid SAJA — tanpa penjelasan, tanpa markdown fence
"""
