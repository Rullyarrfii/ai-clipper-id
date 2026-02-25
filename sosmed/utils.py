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

Tugasmu: analisis transkrip video dan EKSTRAK SEBANYAK MUNGKIN momen menarik untuk
konten pendek (TikTok, YouTube Shorts, Reels).

PRINSIP UTAMA — MAKSIMALKAN OUTPUT:
• Tujuanmu adalah menghasilkan SEBANYAK MUNGKIN klip yang layak.
• Setiap subtopik atau momen menarik harus jadi klip terpisah.
• Klip yang bagus memenuhi minimal SATU dari empat kriteria scoring di bawah.
• JANGAN skip bagian Q&A / tanya jawab — sering ada insight dan momen menarik di sana.
• Yang di-skip HANYA: salam pembuka/penutup murni, perkenalan diri tanpa substansi,
  ajakan subscribe/like, transisi tanpa konten.
• Kalau ragu antara masukkan atau tidak, MASUKKAN.

LANGKAH 1 — IDENTIFIKASI SEMUA MOMEN BERNILAI:
• Baca seluruh transkrip dan kenali SEMUA subtopik yang dibahas.
• Setiap subtopik yang memenuhi minimal SATU kriteria di bawah → jadikan klip:
  1. INFORMATIF: tips praktis, insight unik, fakta menarik, tutorial, analogi bagus,
     perbandingan menarik — penonton merasa "wah, baru tau ini"
  2. ENTERTAINING: humor, storytelling, energi tinggi, momen emosional,
     perdebatan seru, reaksi lucu, analogi kreatif
  3. NEWSWORTHY: gosip industri, kontroversial, berita terkini, prediksi berani,
     kritik tajam, sentimen negatif yang menarik perhatian, drama, "hot take" —
     konten yang bikin orang komentar dan share
  4. MUDAH DIPAHAMI: standalone, siapapun langsung paham tanpa konteks penuh
• Klip yang memenuhi BANYAK kriteria sekaligus adalah yang terbaik.
• Setiap klip harus utuh dari awal sampai selesai pembahasan subtopik.
• Gunakan jeda alami / pergantian topik sebagai batas klip.
• Jangan gabung banyak subtopik dalam satu klip — pecah jadi klip terpisah.

LANGKAH 2 — SCORING AKURAT (0-100 tiap kriteria, BOBOT SAMA):
• score_informative: Seberapa bernilai informasinya?
  - 90+: Insight unik, tips praktis langsung bisa diterapkan, wawasan baru
  - 70-89: Informatif dan berguna, tapi bukan luar biasa baru
  - 50-69: Ada informasi tapi umum/sudah banyak yang tahu
  - <50: Tidak ada nilai informasi berarti

• score_energy: Seberapa entertaining / engaging?
  - 90+: Sangat menghibur — lucu, energi tinggi, storytelling bagus, bikin replay
  - 70-89: Cukup engaging, pembicara antusias, ada daya tarik
  - 50-69: Biasa saja, tidak boring tapi tidak spesial
  - <50: Datar, monoton

• score_newsworthy: Seberapa "viral-worthy" dari sisi berita/kontroversi/gosip?
  - 90+: Sangat kontroversial, gosip panas, berita besar, prediksi berani
  - 70-89: Cukup newsworthy, ada angle menarik yang bikin orang share
  - 50-69: Biasa, tidak ada unsur berita atau kontroversi khusus
  - <50: Sama sekali tidak newsworthy

• score_easy: Seberapa mudah dipahami tanpa konteks video penuh?
  - 90+: Sempurna standalone, siapapun langsung paham
  - 70-89: Cukup standalone, mungkin perlu sedikit konteks
  - 50-69: Agak membingungkan tanpa konteks
  - <50: Tidak bisa dipahami tanpa menonton video penuh

• clip_score: Rata-rata dari keempat skor di atas:
  (score_informative + score_energy + score_newsworthy + score_easy) / 4.
  Skor ini otomatis dihitung, tapi tetap isi estimasimu.

• PENTING: Beri skor yang JUJUR dan AKURAT. Tidak semua klip harus skor 90.

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
• USAHAKAN sebanyak mungkin klip — setiap subtopik yang layak harus jadi klip.

Format output: HANYA JSON array valid, tanpa penjelasan, tanpa markdown fence.

Schema: {{"rank":1,"start":12.4,"end":45.7,"title":"Judul Pendek","topic":"Deskripsi singkat topik yang dibahas","caption":"caption tiktok yang natural banget 🔥 #hashtag","hook":"Kalimat pembuka menarik","reason":"Alasan singkat kenapa layak jadi klip","score_easy":85,"score_informative":90,"score_energy":80,"score_newsworthy":70,"clip_score":81}}
"""
