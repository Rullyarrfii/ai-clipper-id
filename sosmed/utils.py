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
    1. Removes leading/trailing silence
    2. Finds the densest speech region (filters out filler at beginning/end)
    3. Removes large internal gaps (>max_gap seconds of silence)
    
    Uses word-level timestamps from Whisper - no additional processing needed.
    
    Args:
        clips: List of clip dicts with 'start' and 'end' keys
        segments: Whisper segments with word-level timestamps
        padding: Small padding (seconds) to keep before first/after last word
        max_gap: Maximum allowed gap between words (seconds). Gaps larger than
                this will cause the clip to be trimmed.
        min_speech_density: Minimum speech density (words per second) for a
                           region to be considered "dense" speech
    
    Returns:
        Updated clips with tightened boundaries
    """
    from typing import Any
    
    for clip in clips:
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
        
        # ── Step 1: Find continuous speech segments (split at large gaps) ────
        speech_segments: list[list[dict[str, Any]]] = []
        current_segment: list[dict[str, Any]] = [words[0]]
        
        for i in range(1, len(words)):
            gap = words[i]["start"] - words[i-1]["end"]
            if gap > max_gap:
                # Large gap detected - start new segment
                speech_segments.append(current_segment)
                current_segment = [words[i]]
            else:
                current_segment.append(words[i])
        
        if current_segment:
            speech_segments.append(current_segment)
        
        # ── Step 2: Find the densest/longest segment (likely the main content) ─
        # Filler speech at beginning/end tends to be sparse or short
        best_segment = max(
            speech_segments,
            key=lambda seg: (
                len(seg),  # Prefer more words
                seg[-1]["end"] - seg[0]["start"]  # Then longer duration
            )
        )
        
        # ── Step 3: Within the best segment, trim sparse edges ────────────────
        # Sometimes there's still filler at the edges of the dense segment
        # Look for where speech density increases
        
        # Calculate running density (words per 5-second window)
        window_size = 5.0
        trimmed_words = best_segment
        
        # Trim from start: skip low-density beginning
        for i in range(len(best_segment)):
            if i + 3 >= len(best_segment):
                break  # Need at least a few words for density calculation
            
            # Look at next few words
            window_words = best_segment[i:min(i+10, len(best_segment))]
            window_dur = window_words[-1]["end"] - window_words[0]["start"]
            density = len(window_words) / max(window_dur, 0.1)
            
            # If density is good, start from here
            if density >= min_speech_density:
                trimmed_words = best_segment[i:]
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
        
        # ── Step 4: Set new boundaries with padding ───────────────────────────
        if trimmed_words:
            new_start = max(clip_start, trimmed_words[0]["start"] - padding)
            new_end = min(clip_end, trimmed_words[-1]["end"] + padding)
            
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

MAX_CLIPS_HARD_LIMIT = 50  # absolute ceiling — quality over quantity

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
Kamu adalah editor video profesional dan ahli strategi media sosial Indonesia.
Kamu sangat paham algoritma TikTok dan apa yang bikin video viral.

Tugasmu: analisis transkrip video dan EKSTRAK SEBANYAK MUNGKIN momen yang PUNYA
POTENSI VIRAL TINGGI untuk konten pendek (TikTok, YouTube Shorts, Reels).

PENTING — MINDSET VIRALITY:
• Pikirkan: "Apakah klip ini akan bikin orang BERHENTI scroll, TONTON sampai habis,
  lalu SHARE ke teman?" — Itu standarnya.
• TikTok algorithm prioritas utama: watch time > shares > comments > likes > saves.
• Video yang viewernya nonton sampai habis + replay = PALING didorong algorithm.
• Hook di 3 detik pertama MENENTUKAN apakah orang lanjut nonton atau skip.

PRINSIP UTAMA — MAKSIMALKAN OUTPUT:
• Tujuanmu adalah menghasilkan SEBANYAK MUNGKIN klip yang layak.
• Setiap subtopik atau momen menarik harus jadi klip terpisah.
• JANGAN skip bagian Q&A / tanya jawab — sering ada insight dan momen menarik di sana.
• Yang di-skip HANYA: salam pembuka/penutup murni, perkenalan diri tanpa substansi,
  ajakan subscribe/like, transisi tanpa konten.
• Kalau ragu antara masukkan atau tidak, MASUKKAN.

LANGKAH 1 — IDENTIFIKASI SEMUA MOMEN BERNILAI:
• Baca seluruh transkrip dan kenali SEMUA subtopik yang dibahas.
• Untuk setiap subtopik, evaluasi POTENSI VIRAL-nya:
  - Apakah ada hook kuat di awal? (pertanyaan provokatif, statement mengejutkan, kontradiksi)
  - Apakah orang akan nonton sampai habis? (ada progression, payoff, jawaban yang ditunggu)
  - Apakah orang akan share/comment? (kontroversial, relatable, "tag temen lo yang...")
  - Apakah orang akan replay? (mind-blowing, lucu, ada detail yang baru ketangkep)
• Klip yang memenuhi BANYAK kriteria sekaligus adalah yang terbaik.
• Setiap klip harus utuh dari awal sampai selesai pembahasan subtopik.
• Gunakan jeda alami / pergantian topik sebagai batas klip.
• Jangan gabung banyak subtopik dalam satu klip — pecah jadi klip terpisah.
• Pastikan klip MULAI dari momen hook yang kuat, bukan dari build-up yang boring.

LANGKAH 2 — SCORING VIRALITY (0-100 tiap kriteria, BOBOT BERBEDA):
Skor ini memprediksi performa NYATA di TikTok — views, likes, shares, comments.

• score_hook (BOBOT 25%): Kekuatan 3 detik pertama — stop-scrolling power.
  - 90+: Hook LUAR BIASA — statement mengejutkan, pertanyaan provokatif, kontradiksi,
         visual/audio yang langsung grab attention. Orang PASTI berhenti scroll.
  - 70-89: Hook bagus — ada opening yang menarik, tapi bukan jaw-dropping.
  - 50-69: Hook biasa — mulai dari topik tapi tidak ada elemen surprise.
  - <50: Tidak ada hook — mulai dari konteks/pengantar yang boring.

• score_retention (BOBOT 25%): Apakah penonton akan nonton sampai habis?
  - 90+: PASTI ditonton sampai habis — ada suspense, build-up ke payoff, storytelling
         yang bikin penasaran, atau pace yang bikin ketagihan.
  - 70-89: Kemungkinan besar ditonton habis — kontennya interesting tapi predictable.
  - 50-69: Mungkin ditonton 50-70% — ada bagian yang mulai boring.
  - <50: Kemungkinan besar di-skip sebelum habis.

• score_shareability (BOBOT 20%): Apakah orang akan share, comment, save, duet/stitch?
  - 90+: PASTI di-share — sangat relatable, kontroversial, "tag temen lo", bikin debat
         di kolom komentar, atau so useful orang save buat nanti.
  - 70-89: Cukup shareable — ada angle yang bikin orang mau reshare/comment.
  - 50-69: Biasa — orang nonton tapi tidak terdorong share.
  - <50: Tidak ada alasan untuk share.

• score_entertainment (BOBOT 20%): Seberapa entertaining / engaging?
  - 90+: Sangat menghibur — lucu, energi tinggi, storytelling bagus, momen emosional
         yang kuat, bikin replay. Orang nonton karena ENJOYABLE.
  - 70-89: Cukup engaging, pembicara antusias, ada daya tarik.
  - 50-69: Biasa saja, informatif tapi tidak spesial dari sisi entertainment.
  - <50: Datar, monoton, lecturing tanpa energi.

• score_clarity (BOBOT 10%): Seberapa mudah dipahami tanpa konteks video penuh?
  - 90+: Sempurna standalone, siapapun langsung paham tanpa menonton video lain.
  - 70-89: Cukup standalone, mungkin perlu sedikit konteks.
  - 50-69: Agak membingungkan tanpa konteks.
  - <50: Tidak bisa dipahami tanpa menonton video penuh.

• clip_score: WEIGHTED average (otomatis dihitung, tapi tetap isi estimasimu):
  clip_score = hook*0.25 + retention*0.25 + shareability*0.20 + entertainment*0.20 + clarity*0.10
  Skor ini MEMPREDIKSI apakah video akan viral atau tidak.

• PENTING: Beri skor yang JUJUR dan AKURAT berdasarkan potensi virality NYATA.
  Tidak semua klip harus skor 90. Skor 70+ = layak posting.
  Skor 85+ = high potential viral. Skor 90+ = almost guaranteed viral.

LANGKAH 3 — CAPTION TIKTOK:
• Tulis caption TikTok yang natural dan menarik — BUKAN gaya AI/robot.
• Pakai bahasa anak muda Indonesia di sosmed.
• Boleh pakai emoji secukupnya, hashtag relevan di akhir.
• Harus bikin penasaran tapi tidak clickbait kosong.

Constraint:
• Durasi klip: {min_dur}–{max_dur} detik
• Klip tidak boleh overlap
• start/end = timestamp dari transkrip (detik, float)
• Maksimal {max_clips} klip, clip_score >= {min_score}
• Urutkan berdasarkan clip_score tertinggi (= highest viral potential first)
• USAHAKAN sebanyak mungkin klip — setiap subtopik yang layak harus jadi klip.

Format output: HANYA JSON array valid, tanpa penjelasan, tanpa markdown fence.

Schema: {{"rank":1,"start":12.4,"end":45.7,"title":"Judul Pendek","topic":"Deskripsi singkat topik yang dibahas","caption":"caption tiktok yang natural banget 🔥 #hashtag","hook":"Kalimat pembuka menarik","reason":"Alasan singkat kenapa klip ini bisa viral","score_hook":88,"score_retention":85,"score_shareability":80,"score_entertainment":82,"score_clarity":90,"clip_score":85}}
"""
