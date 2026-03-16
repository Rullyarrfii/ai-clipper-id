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
DEFAULT_OPENROUTER_MODEL = "openrouter/hunter-alpha"
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


# ━━━━━━━━━━━━━━━━━━━━━━━━━━ INTERNAL FIELDS CACHE ━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_clips_cache_dir(output_dir):
    """Get cache directory for storing internal clip fields."""
    from pathlib import Path
    output = Path(output_dir)
    cache_dir = output.parent / ".cache" / output.name
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def strip_internal_fields(clips):
    """Return a copy of clips with all _ -prefixed fields removed."""
    cleaned = []
    for clip in clips:
        clean_clip = {k: v for k, v in clip.items() if not k.startswith("_")}
        cleaned.append(clean_clip)
    return cleaned


def get_internal_fields(clips):
    """Extract only internal (underscore-prefixed) fields from clips, keyed by filename."""
    internal_by_filename = {}
    for clip in clips:
        filename = clip.get("filename")
        if filename:
            internal = {k: v for k, v in clip.items() if k.startswith("_")}
            if internal:
                internal_by_filename[filename] = internal
    return internal_by_filename


def save_clips_to_disk(clips, output_dir):
    """Save clips cleanly: public fields to clips.json, internal fields to .cache."""
    import json
    from pathlib import Path
    
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    # Strip and save public version
    public_clips = strip_internal_fields(clips)
    clips_json = output / "clips.json"
    clips_json.write_text(json.dumps(public_clips, indent=2, ensure_ascii=False))
    
    # Save internal fields to cache
    internal_fields = get_internal_fields(clips)
    if internal_fields:
        cache_dir = get_clips_cache_dir(output)
        cache_file = cache_dir / "clips_internal.json"
        cache_file.write_text(json.dumps(internal_fields, indent=2, ensure_ascii=False))
    
    return clips_json


def load_clips_with_internal_fields(output_dir):
    """Load clips.json and merge in internal fields from cache."""
    import json
    from pathlib import Path
    
    output = Path(output_dir)
    clips_json = output / "clips.json"
    
    if not clips_json.exists():
        return []
    
    clips = json.loads(clips_json.read_text())
    
    # Try to load internal fields from cache
    cache_dir = get_clips_cache_dir(output)
    cache_file = cache_dir / "clips_internal.json"
    
    if cache_file.exists():
        try:
            internal_fields = json.loads(cache_file.read_text())
            for clip in clips:
                filename = clip.get("filename")
                if filename and filename in internal_fields:
                    clip.update(internal_fields[filename])
        except Exception as e:
            log("WARN", f"Could not load internal fields from cache: {e}")
    
    return clips

FILLER_RE = re.compile(rf"^({_ID_FILLERS})\W*$", re.IGNORECASE)

SYSTEM_PROMPT = """\
GOAL
You are a viral content strategist. Your only goal is to extract ONLY clips guaranteed to perform exceptionally well on short-form video (TikTok, Instagram Reels, YouTube Shorts).
Curate ruthlessly. Better to miss a borderline clip than to include anything that won't hit. Only output clips with genuine viral potential.

Clip duration: {min_dur}–{max_dur} seconds. Maximum {max_clips} clips. Output JSON array only — no explanation, no markdown fence.

---

STEP 1 — FILTER OUT IMMEDIATELY (do not score these)
Skip any clip that matches at least one of the following:
- Contains only greetings, closings, or housekeeping ("sapa peserta", "see you next week", "thanks for joining")
- Is a pure teaser for future content with nothing standalone to watch
- Audio starts or ends mid-sentence with no payoff

Everything else moves to Step 2.

---

STEP 2 — SCORE EACH CLIP
Assign a number 0–100 for each dimension using the anchors below.

score_hook — Stop-scroll power in the first 2 seconds
90–100 | Shocking statement, direct provocation, strong contradiction, instantly recognizable meme/brand reference
70–89  | Clear question or bold claim, minor surprise, light humor opener
50–69  | Neutral but relevant opener, no filler
30–49  | Slow buildup, context-setting before the point
0–29   | Filler words, technical jargon opener, greeting

score_retention — Will viewers watch to the end?
90–100 | <45s clip, clear beginning and end, suspense or curiosity built toward a payoff
70–89  | Clean arc, moderate length, ends with resolution
50–69  | Decent pacing, minor dead air, still resolves
30–49  | Trails off, missing context, or too long without payoff
0–29   | No resolution, pure ramble, or clearly cut mid-thought

score_shareability — Would someone tag a friend or repost?
90–100 | Controversial take, "send this to your developer friend" energy, immediately useful insight, counterintuitive reveal
70–89  | Broadly relatable frustration or win, mildly useful tip
50–69  | Interesting but niche, shareable only within a specific community
30–49  | Only makes sense mid-stream, requires prior context
0–29   | No standalone value

score_entertainment — Like trigger: humor, surprise, or triumph
90–100 | Unexpected punchline, triumphant "it works!" moment, laugh-out-loud situation
70–89  | Funny-ish, relatable win or fail, light emotional hit
50–69  | Mildly amusing or satisfying, no strong emotion
30–49  | Flat, informational, no emotional payoff
0–29   | Dry tutorial steps only

score_clarity — Does it work without watching the full stream?
90–100 | Fully self-contained, anyone can watch cold
70–89  | Mostly understandable, minor context gap
50–69  | Understandable if viewer knows the general topic
30–49  | Confusing without the stream, but funny/interesting anyway
0–29   | Completely unintelligible without prior context

---

STEP 3 — CALCULATE clip_score
Use this exact formula:

clip_score = (score_hook × 0.30) + (score_shareability × 0.25) + (score_entertainment × 0.25) + (score_retention × 0.15) + (score_clarity × 0.05)

---

STEP 4 — APPLY STRICT SELECTION RULES

INCLUDE the clip only if ALL are true:
- clip_score ≥ {min_score}
- AND at least THREE individual scores ≥ 80 (hook, shareability, retention, or entertainment must dominate)
- AND score_hook ≥ 75 (weak hooks don't stop scrolls)
- AND score_clarity ≥ 60 (standalone must work without context)

DEDUPLICATE: If two clips cover the exact same moment or insight, keep only the one with the higher clip_score. Similar topics from different angles are NOT duplicates — keep both only if both pass the threshold above.

---

STEP 5 — GENERATE FIELDS IN THIS EXACT SEQUENCE FOR EVERY INCLUDED CLIP

For each clip, reason through the fields in this order before writing any JSON:

(1) topic
- One sentence: the core insight or emotional moment, not just the subject category
- Think: What is actually happening or being revealed in this clip?

(2) reason
- Name the specific viral signal(s) driving this clip (hook / shareability / entertainment / retention / clarity)
- Explain in 1–2 sentences why that signal applies to this specific clip
- Think: What makes this clip work? Which scoring dimensions drove its acceptance?

(3) hook
- Use the single most provocative or emotionally charged line in the clip
- If the original hook is weak, escalate it — use the clip's best internal line as the hook
- Do NOT start with "In this clip..." or any description
- Think: Based on the topic and reason above, what 1–2 words will stop scrolling?

(4) caption
- Write like a native creator posting their own content
- Must accurately reflect what actually happens in the clip
- Include relevant hashtags at the end
- Think: How would someone naturally describe this clip if sharing it?

(5) title
- Max 8 words
- Must create a curiosity gap — the viewer should want to know the answer
- Think: Based on the hook, topic, and reason, what question or claim makes sense?

---

STEP 6 — OUTPUT FORMAT

Return a JSON array:
- All original fields preserved
- Rewritten fields replace original values entirely (topic, reason, hook, caption, title generated in that order)
- Sorted by clip_score descending
- No new fields added, no fields removed
"""
