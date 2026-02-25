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

Tugasmu: analisis transkrip video, identifikasi SEMUA SUBTOPIK yang dibahas,
dan jadikan SETIAP subtopik sebagai klip pendek (TikTok, YouTube Shorts, Reels).

PRINSIP UTAMA: KUALITAS di atas kuantitas. Pilih HANYA bagian yang benar-benar
menonjol — yang energinya tinggi, informatif, dan mudah dipahami tanpa konteks.
Jangan paksakan klip dari bagian yang biasa-biasa saja.

LANGKAH 1 — IDENTIFIKASI SEMUA SUBTOPIK:
• Baca seluruh transkrip dan kenali SETIAP subtopik / pembahasan yang berbeda.
• Hanya ambil subtopik yang punya energi tinggi, nilai informasi kuat, atau
  sangat mudah dipahami — skip bagian yang datar/membosankan.
• Setiap subtopik harus utuh dari awal sampai selesai (JANGAN potong di tengah
  pembahasan). Satu subtopik = satu klip.
• Gunakan jeda alami / pergantian topik sebagai batas — BUKAN tengah kalimat.
• Mulai dari awal pembahasan subtopik, akhiri setelah subtopik benar-benar selesai.
• Jika ada subtopik panjang, coba pecah jadi beberapa klip yang masing-masing
  bisa berdiri sendiri.

LANGKAH 2 — SCORING (0-100 tiap kriteria):
• score_easy: Seberapa mudah dipahami tanpa konteks video penuh?
  (standalone, bahasa sederhana, tidak butuh pengetahuan khusus)
• score_informative: Seberapa bernilai informasinya?
  (tips praktis, insight, fakta menarik, bisa langsung diterapkan)
• score_energy: Seberapa tinggi energi / daya tariknya?
  (antusias, lucu, emosional, ada hook kuat, penonton akan stay)
• clip_score: Rata-rata dari ketiga skor di atas
• Beri skor yang KETAT dan realistis. Hanya klip dengan skor tinggi yang layak.
• Prioritaskan: score_energy tinggi, score_informative tinggi, score_easy tinggi.

LANGKAH 3 — CAPTION TIKTOK:
• Tulis caption TikTok yang natural dan menarik — BUKAN gaya AI/robot.
• Pakai bahasa yang dipakai anak muda Indonesia di sosmed.
• Boleh pakai emoji secukupnya, hashtag relevan di akhir.
• Harus bikin penasaran tapi tidak clickbait kosong.

Constraint:
• Durasi klip: {min_dur}–{max_dur} detik
• Klip tidak boleh overlap secara signifikan (sedikit overlap tidak apa-apa)
• start/end = timestamp dari transkrip (detik, float) — HARUS mencakup
  seluruh subtopik dari awal sampai selesai
• Kembalikan HANYA klip terbaik, maksimal {max_clips} klip, clip_score >= {min_score}
• Urutkan berdasarkan clip_score tertinggi
• PENTING: lebih baik sedikit klip berkualitas tinggi daripada banyak klip biasa.
  Pilih yang benar-benar high energy, informatif, dan mudah dipahami.

Format output: HANYA JSON array valid, tanpa penjelasan, tanpa markdown fence.

Schema: {{"rank":1,"start":12.4,"end":45.7,"title":"Judul Pendek","topic":"Deskripsi singkat topik yang dibahas","caption":"caption tiktok yang natural banget 🔥 #hashtag","hook":"Kalimat pembuka menarik","reason":"Alasan singkat kenapa layak jadi klip","score_easy":85,"score_informative":90,"score_energy":80,"clip_score":85}}
"""
