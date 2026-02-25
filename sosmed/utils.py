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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━ CONSTANTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Default free model on OpenRouter
DEFAULT_OPENROUTER_MODEL = "arcee-ai/trinity-large-preview:free"
DEFAULT_OPENROUTER_BASE  = "https://openrouter.ai/api/v1"

MAX_CLIPS_HARD_LIMIT = 30  # absolute ceiling — quality over quantity

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

Tugasmu: analisis transkrip video dan pilih momen-momen terbaik untuk
konten pendek (TikTok, YouTube Shorts, Reels).

PRINSIP UTAMA — AKURAT & SELEKTIF:
• Pilih klip berdasarkan seberapa INFORMATIF dan ENTERTAINING kontennya.
• Klip yang bagus = penonton belajar sesuatu (informatif) ATAU terhibur (entertaining).
  Idealnya keduanya. Semakin tinggi dua aspek ini, semakin layak jadi klip.
• Jangan masukkan bagian yang tidak punya nilai informasi atau hiburan:
  opening/salam pembuka, penutup/closing, perkenalan diri, polling/vote,
  transisi antar topik, ucapan terima kasih, promosi channel, ajakan subscribe.
• Lebih baik sedikit klip yang benar-benar bagus daripada banyak klip mediocre.

LANGKAH 1 — IDENTIFIKASI MOMEN BERNILAI:
• Baca seluruh transkrip dan kenali subtopik yang dibahas.
• Pilih subtopik yang memenuhi minimal SATU dari kriteria ini:
  1. INFORMATIF: ada tips praktis, insight unik, fakta menarik, tutorial, penjelasan
     yang bernilai — penonton merasa "wah, baru tau ini"
  2. ENTERTAINING: ada humor, storytelling menarik, energi tinggi, momen emosional,
     perdebatan seru, reaksi lucu — penonton terhibur dan ingin tonton lagi
• Klip yang memenuhi KEDUANYA (informatif + entertaining) adalah yang terbaik.
• Setiap klip harus utuh dari awal sampai selesai pembahasan.
• Klip harus cukup mudah dipahami tanpa perlu menonton video penuh.
• Gunakan jeda alami / pergantian topik sebagai batas klip.

LANGKAH 2 — SCORING AKURAT (0-100 tiap kriteria):
• score_informative (BOBOT TINGGI): Seberapa bernilai informasinya?
  - 90+: Insight unik, tips praktis yang langsung bisa diterapkan, penjelasan
         yang membuka wawasan baru
  - 70-89: Informatif dan berguna, tapi bukan sesuatu yang luar biasa baru
  - 50-69: Ada informasi tapi umum/sudah banyak yang tahu
  - <50: Tidak ada nilai informasi yang berarti

• score_energy (BOBOT TINGGI): Seberapa entertaining / engaging?
  - 90+: Sangat menghibur — lucu, energi tinggi, storytelling bagus,
         bikin penonton replay
  - 70-89: Cukup engaging, pembicara antusias, ada daya tarik
  - 50-69: Biasa saja, tidak boring tapi tidak menarik
  - <50: Datar, monoton, membosankan

• score_easy: Seberapa mudah dipahami tanpa konteks video penuh?
  - 90+: Sempurna standalone, siapapun langsung paham
  - 70-89: Cukup standalone, mungkin perlu sedikit konteks
  - 50-69: Agak membingungkan tanpa konteks
  - <50: Tidak bisa dipahami tanpa menonton video penuh

• clip_score: Weighted average → (score_informative × 2 + score_energy × 2 +
  score_easy × 1) / 5. Skor ini otomatis dihitung, tapi tetap isi estimasimu.

• PENTING: Beri skor yang JUJUR dan AKURAT, bukan inflate atau deflate.
  Tidak semua klip harus skor 90. Beri skor sesuai kualitas sebenarnya.

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
• Urutkan berdasarkan clip_score tertinggi
• Jika hanya ada sedikit momen yang benar-benar bagus, return sedikit saja.
  Jangan paksakan kuantitas.

Format output: HANYA JSON array valid, tanpa penjelasan, tanpa markdown fence.

Schema: {{"rank":1,"start":12.4,"end":45.7,"title":"Judul Pendek","topic":"Deskripsi singkat topik yang dibahas","caption":"caption tiktok yang natural banget 🔥 #hashtag","hook":"Kalimat pembuka menarik","reason":"Alasan singkat kenapa layak jadi klip","score_easy":85,"score_informative":90,"score_energy":80,"clip_score":85}}
"""
