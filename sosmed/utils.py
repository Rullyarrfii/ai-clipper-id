"""
Utilities: logging, ANSI colors, constants.
"""

import re

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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━ CONSTANTS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Default free model on OpenRouter
DEFAULT_OPENROUTER_MODEL = "arcee-ai/trinity-large-preview:free"
DEFAULT_OPENROUTER_BASE  = "https://openrouter.ai/api/v1"

MAX_CLIPS_HARD_LIMIT = 200  # absolute ceiling

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

Tugasmu: analisis transkrip video dan temukan SEBANYAK MUNGKIN segmen menarik
yang cocok untuk klip pendek viral (TikTok, YouTube Shorts, Instagram Reels).

PENTING — jangan pelit! Cari semua momen menarik, lucu, informatif, emosional,
atau quotable. Kembalikan antara 0 hingga {max_clips} klip. Lebih banyak lebih
baik selama memenuhi kriteria minimum.

Kriteria penilaian (engagement_score 0-100):
• Hook kuat di awal — penonton harus langsung tertarik
• Puncak emosi — lucu, mengejutkan, inspiratif, kontroversial, mengharukan
• Informasi berguna / tips / insight yang berdiri sendiri
• Cerita / argumen yang utuh dan mandiri (self-contained)
• Kalimat quotable / memorable yang bisa jadi caption
• Potong di jeda alami, bukan tengah kalimat
• Konten yang relatable untuk audiens Indonesia
• Skor >= {min_score} berarti layak untuk diposting

Constraint:
• Durasi klip: {min_dur}–{max_dur} detik
• Tidak boleh ada klip yang overlap
• start/end = timestamp dari transkrip (detik, float)
• Urutkan berdasarkan engagement_score tertinggi

Format output: HANYA JSON array valid, tanpa penjelasan, tanpa markdown fence.

Schema: {{"rank":1,"start":12.4,"end":45.7,"title":"Judul","reason":"Alasan singkat","hook":"Kalimat pembuka","engagement_score":92}}
"""
