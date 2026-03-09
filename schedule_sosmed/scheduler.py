#!/usr/bin/env python3
# ============================================================
#  scheduler.py  –  Cross-platform video uploader  [v2]
#  Platforms : Instagram Reels | YouTube Shorts | TikTok
#
#  Posting strategy:
#    • Times = statistically best slots for Indonesian audience
#    • Each slot has a tier (1=best engagement)
#    • Highest clip_score goes to best-tier slot
#    • clips.json re-read fresh every job (edit anytime)
#    • Filename read from "filename" field in clips.json
#    • No FFmpeg — raw files uploaded as-is
# ============================================================

import os
import json
import time
import pickle
import random
import shutil
import logging
import argparse
import schedule
import pytz
import subprocess
from datetime import datetime, timezone

from instagrapi import Client as IGClient
from instagrapi.exceptions import LoginRequired
import googleapiclient.discovery
import google_auth_oauthlib.flow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload
from tiktok_uploader.upload import TikTokUploader
import atexit

from config import (
    CLIPS_FOLDER,
    INSTAGRAM_CREDENTIAL_FILE, INSTAGRAM_SESSION_FILE,
    YOUTUBE_CLIENT_SECRETS, YOUTUBE_TOKEN_FILE, YOUTUBE_CATEGORY_ID,
    YOUTUBE_CHANNEL_HANDLE,
    YOUTUBE_PRIVACY_STATUS,
    YOUTUBE_DEFAULT_LANGUAGE,
    YOUTUBE_DEFAULT_AUDIO_LANGUAGE,
    YOUTUBE_LICENSE,
    YOUTUBE_EMBEDDABLE,
    YOUTUBE_PUBLIC_STATS_VIEWABLE,
    YOUTUBE_NOTIFY_SUBSCRIBERS,
    YOUTUBE_SELF_DECLARED_MADE_FOR_KIDS,
    TIKTOK_COOKIES_FILE,
    TIKTOK_BROWSER_DATA_DIR,
    FFMPEG_BIN,
    FFPROBE_BIN,
    ENABLE_INSTAGRAM, ENABLE_YOUTUBE, ENABLE_TIKTOK,
)

# ── Dynamic daily upload target ──────────────────────────────────────
# Each day a fresh target is drawn from a weighted distribution.
#
# Research basis (as of 2024-2025):
#   • TikTok Creator Academy: 3–5/day optimal for growth accounts
#   • Meta Business Blog: 3–5 Reels/day for short-form clip accounts
#   • YouTube Creator Insider: 2–4 Shorts/day to avoid cannibalisation
#
# Weights → [3,4,5,6] = [15%, 35%, 35%, 15%]
# Peak at 4–5: highest algorithmic reach without triggering spam filters.
_DAILY_UPLOAD_POOL   = [3, 4, 5, 6]
_DAILY_UPLOAD_WEIGHTS = [3, 7, 7, 3]

# Drawn fresh every midnight; shared across all platforms since one slot
# = one cross-platform post.
_daily_target: int = 4  # will be overwritten on first reset

# Track daily upload counts (reset at midnight WIB)
_daily_counts = {
    "youtube": 0,
    "instagram": 0,
    "tiktok": 0,
    "last_reset_date": None,
}

# Track last upload timestamp per platform (minimum gap enforcement)
_last_upload_ts: dict[str, float] = {
    "youtube": 0.0,
    "instagram": 0.0,
    "tiktok": 0.0,
}

# Minimum seconds between consecutive uploads to the same platform.
# 90 minutes keeps us under spam-detection thresholds while safely
# accommodating up to 15-min jitter on 2-hour-spaced slots
# (worst case: 2 h − 15 min jitter = 1 h 45 min > 90 min ✓).
MIN_GAP_BETWEEN_UPLOADS_SECS = 90 * 60  # 90 minutes

def reset_daily_counts_if_needed():
    """Reset upload counters and draw a fresh daily target at midnight WIB."""
    global _daily_target
    today = datetime.now(pytz.timezone("Asia/Jakarta")).date()
    if _daily_counts["last_reset_date"] != today:
        _daily_counts["youtube"] = 0
        _daily_counts["instagram"] = 0
        _daily_counts["tiktok"] = 0
        _daily_counts["last_reset_date"] = today
        _daily_target = random.choices(_DAILY_UPLOAD_POOL, weights=_DAILY_UPLOAD_WEIGHTS, k=1)[0]
        log.info(f"📊 Daily counters reset for {today}. Today's upload target: {_daily_target}/platform")

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("uploader.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

CLIPS_JSON = os.path.join(CLIPS_FOLDER, "clips.json")

# folder where we keep a simple record of every clip that was posted
# - a JSON array is stored in logs/clips.json and a line-oriented journal
#   is kept in logs/uploads.jsonl for easy tailing/debugging
LOGS_FOLDER = os.path.join(os.path.dirname(CLIPS_FOLDER), "logs")

# request read access as well so that we can call channels.list (confirm auth)
# and future features that read channel data.  If the saved token only had
# the upload scope we'll drop it and re-authorize automatically.
YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
    # alternatively use the single broad scope:
    # "https://www.googleapis.com/auth/youtube",
]

FFMPEG_EXECUTABLE = FFMPEG_BIN
FFPROBE_EXECUTABLE = FFPROBE_BIN


def resolve_executable(executable: str) -> str | None:
    executable = (executable or "").strip()
    if not executable:
        return None

    if os.path.isabs(executable) or os.path.dirname(executable):
        if os.path.isfile(executable):
            return executable
        return None

    return shutil.which(executable)


def ensure_media_tools_ready() -> bool:
    global FFMPEG_EXECUTABLE, FFPROBE_EXECUTABLE

    ffmpeg_resolved = resolve_executable(FFMPEG_BIN)
    ffprobe_resolved = resolve_executable(FFPROBE_BIN)

    if not ffmpeg_resolved:
        log.error(
            "❌ ffmpeg executable not found. "
            f"Set FFMPEG_BIN in config.py (current: '{FFMPEG_BIN}')."
        )
    if not ffprobe_resolved:
        log.error(
            "❌ ffprobe executable not found. "
            f"Set FFPROBE_BIN in config.py (current: '{FFPROBE_BIN}')."
        )

    if not ffmpeg_resolved or not ffprobe_resolved:
        log.error("   Install FFmpeg and point both settings to valid executables.")
        return False

    FFMPEG_EXECUTABLE = ffmpeg_resolved
    FFPROBE_EXECUTABLE = ffprobe_resolved
    log.info(f"🎬 ffmpeg : {FFMPEG_EXECUTABLE}")
    log.info(f"🎬 ffprobe: {FFPROBE_EXECUTABLE}")
    return True

# ═══════════════════════════════════════════════════════════
#  Statistically best posting times — Indonesian audience WIB
#
#  Based on: Sprout Social, Later, HubSpot, SocialBee research
#  + Indonesian social media usage patterns (APJII, We Are Social)
#
#  Tier 1 ★★★ — highest engagement windows
#  Tier 2 ★★  — strong engagement
#  Tier 3 ★   — moderate engagement
#
#  21:00  Peak prime time. Highest daily screen time.
#         Everyone is home, relaxed, long scroll sessions.
#  12:00  Lunch break. Commuters + office workers peak.
#         2nd highest engagement across all 3 platforms.
#  19:00  After dinner / Maghrib. Family screen time begins.
#         Reels/Shorts perform especially well here.
#  17:00  End of school/work. Commute scroll.
#         Strong for TikTok and Reels.
#  09:00  Mid-morning work break. Consistent but lower.
#  07:00  Morning commute. Decent reach, lower engagement rate.
# ═══════════════════════════════════════════════════════════

SCHEDULE_SLOTS = [
    # (time_WIB, tier, label)
    ("21:00", 1, "Peak prime time"),
    ("12:00", 1, "Lunch peak"),
    ("19:00", 2, "After dinner"),
    ("17:00", 2, "End of work/school"),
    ("09:00", 3, "Morning work break"),
    ("07:00", 3, "Morning commute"),
]

# ── Day-of-week slot selection ──────────────────────────────
# Not every slot fires every day.  This keeps the posting pattern
# looking organic (humans don't post at exact same 6 times daily).
# Each day randomly picks a subset of slots from the full list.
# Tier-1 slots are always included; lower tiers are included
# probabilistically.  The result is 2–4 posts on a typical day
# instead of a rigid 6.

# Keep track of which slots are active today (regenerated at midnight)
DAY_ENGAGEMENT = {
    "Monday":    0.85,
    "Tuesday":   0.90,
    "Wednesday": 0.95,
    "Thursday":  1.00,  # Best engagement
    "Friday":    0.92,
    "Saturday":  0.88,
    "Sunday":    0.80,
}

# Keep track of which slots are active today (regenerated at midnight)
_active_slots: set[str] = set()
_active_slots_date = None

def refresh_active_slots():
    """Select today's active slots so the count matches ``_daily_target``.

    Slots are filled by tier priority (Tier 1 first, then 2, then 3).
    Within each tier the order is shuffled so the specific times chosen
    vary day-to-day, preserving organic-looking behaviour.
    Called once per day (or on first run).
    """
    global _active_slots, _active_slots_date
    today = datetime.now(pytz.timezone("Asia/Jakarta")).date()
    if _active_slots_date == today:
        return

    # Ensure we have a fresh daily target for today
    reset_daily_counts_if_needed()

    # Group available slots by tier
    by_tier: dict[int, list[str]] = {}
    for slot_time, tier, _label in SCHEDULE_SLOTS:
        by_tier.setdefault(tier, []).append(slot_time)

    # Fill slots in tier order until the daily target is reached
    selected: list[str] = []
    remaining = _daily_target
    for tier in sorted(by_tier.keys()):
        tier_slots = by_tier[tier][:]
        random.shuffle(tier_slots)
        take = min(remaining, len(tier_slots))
        selected.extend(tier_slots[:take])
        remaining -= take
        if remaining == 0:
            break

    _active_slots = set(selected)
    _active_slots_date = today
    log.info(
        f"📅 Active slots for {today} (target={_daily_target}): "
        f"{sorted(_active_slots)}  ({len(_active_slots)}/{len(SCHEDULE_SLOTS)})"
    )


# ═══════════════════════════════════════════════════════════
#  Clip queue helpers
# ═══════════════════════════════════════════════════════════

def dedupe_clips(clips: list) -> list:
    """Remove duplicate entries based on *filename*.

    Clips are considered identical if they share the same ``filename``.  This
    function keeps the first occurrence and logs a warning for each removed
    duplicate.  Returns a new list in the original order with duplicates
    stripped.  If an entry is missing ``filename`` we leave it in place; the
    caller will warn separately.
    """
    seen = set()
    out = []
    for clip in clips:
        fn = clip.get("filename")
        if fn in seen:
            log.warning(f"duplicate clip entry removed (filename already seen): {fn}")
            continue
        seen.add(fn)
        out.append(clip)
    return out


def load_clips() -> list:
    """Read the queue JSON, dedupe entries and validate fields.

    The queue may be edited by hand or generated by external scripts, so we
    perform a little sanitisation here.  We intentionally *do not* use the
    old ``rank`` field as an identity – only ``filename`` matters.
    """
    with open(CLIPS_JSON, "r", encoding="utf-8") as f:
        clips = json.load(f)

    orig_len = len(clips)
    clips = dedupe_clips(clips)
    if len(clips) != orig_len:
        log.info(f"🧹 removed {orig_len - len(clips)} duplicate clip(s) from queue")

    # warn about missing filenames
    for clip in clips:
        if not clip.get("filename"):
            log.warning(f"clip missing 'filename' field: rank {clip.get('rank')} title '{clip.get('title')}'")
    return clips


def get_posted_filenames() -> set:
    """Return a set of filenames that have already been uploaded.

    We prefer the structured log ``logs/clips.json`` if it exists because it
    contains the full metadata.  As a fallback we scan ``uploader.log`` for
    lines that mention a filename; this is less reliable but better than
    nothing.
    """
    posted = set()
    json_log = os.path.join(LOGS_FOLDER, "clips.json")
    if os.path.exists(json_log):
        try:
            with open(json_log, "r", encoding="utf-8") as f:
                data = json.load(f)
            for entry in data:
                clip = entry.get("clip", {})
                fn = clip.get("filename")
                if fn:
                    posted.add(fn)
        except Exception:
            log.warning("Could not read structured upload log; falling back to text scan")
    if not posted and os.path.exists("uploader.log"):
        import re
        with open("uploader.log", "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                m = re.search(r"Filename\s*:\s*(\S+\.mp4)", line)
                if m:
                    posted.add(m.group(1))
    return posted


def clean_orphan_files(clips: list) -> int:
    """Remove any video files in ``CLIPS_FOLDER`` that aren't in clips.

    Returns the number of files deleted.  Only ``.mp4`` files are considered;
    padded derivatives (``_padded.mp4``) are also swept if their base file is
    missing.
    """
    valid = {c.get("filename") for c in clips if c.get("filename")}
    deleted = 0
    for fname in os.listdir(CLIPS_FOLDER):
        if not fname.lower().endswith(".mp4"):
            continue
        if fname not in valid:
            path = os.path.join(CLIPS_FOLDER, fname)
            try:
                os.remove(path)
                log.info(f"🧹 removed orphan video file: {path}")
                deleted += 1
            except Exception as e:
                log.warning(f"failed to delete orphan {path}: {e}")
    return deleted



def save_clips(clips: list):
    """Write clips to disk atomically to prevent corruption on crash."""
    tmp = CLIPS_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(clips, f, indent=2, ensure_ascii=False)
    os.replace(tmp, CLIPS_JSON)



# Enhanced: pick best clip for slot and day, using engagement statistics
def get_clip_for_slot_and_day(slot_time: str, day_name: str) -> tuple:
    """
    Pick the best clip for a given time slot and day, using engagement statistics.
    Matching logic:
      - For the current day and slot, multiply engagement score by clip_score (or class_score if present)
      - Sort all available clips by this combined score
      - Return the highest scoring clip
    """
    clips = load_clips()
    if not clips:
        return None, None

    slot_info = next((s for s in SCHEDULE_SLOTS if s[0] == slot_time), None)
    tier = slot_info[1] if slot_info else 3
    engagement = DAY_ENGAGEMENT.get(day_name, 1.0)

    available = []
    for clip in clips:
        filename = clip.get("filename")
        if not filename:
            log.warning(f"Clip rank {clip.get('rank', '?')} missing 'filename' field — skipping")
            continue
        path = os.path.join(CLIPS_FOLDER, filename)
        if os.path.exists(path):
            # Use class_score if present, else clip_score
            score = clip.get("class_score", clip.get("clip_score", 0))
            combined_score = score * engagement
            available.append((clip, path, combined_score))
        else:
            log.warning(f"File not found for rank {clip.get('rank', '?')}: {filename}")

    if not available:
        return None, None

    # Sort by combined score descending
    available.sort(key=lambda x: x[2], reverse=True)
    # Always pick the highest scoring available clip
    return available[0][0], available[0][1]


def get_clip_by_filename(filename: str) -> tuple:
    clips = load_clips()
    for clip in clips:
        if clip.get("filename") == filename:
            path = os.path.join(CLIPS_FOLDER, filename)
            if os.path.exists(path):
                return clip, path
            log.error(f"File for test post not found: {path}")
            return None, None
    
    # If not found in clips.json, check logs/clips.json
    json_log = os.path.join(LOGS_FOLDER, "clips.json")
    if os.path.exists(json_log):
        try:
            with open(json_log, "r", encoding="utf-8") as f:
                log_data = json.load(f)
            for entry in log_data:
                clip = entry.get("clip", {})
                if clip.get("filename") == filename:
                    path = os.path.join(CLIPS_FOLDER, filename)
                    if os.path.exists(path):
                        return clip, path
                    log.error(f"File for test post not found: {path}")
                    return None, None
        except Exception as e:
            log.warning(f"Could not read logs/clips.json: {e}")
    
    log.error(f"No clip entry found in clips.json or logs/clips.json for filename: {filename}")
    return None, None


def log_upload(clip: dict, video_path: str, results: dict) -> None:
    """Write upload results to structured logs.

    Appends to both logs/clips.json (structured array) and logs/uploads.jsonl
    (newline-delimited journal for easy tailing).
    """
    os.makedirs(LOGS_FOLDER, exist_ok=True)
    
    timestamp = datetime.now(pytz.timezone("Asia/Jakarta")).isoformat()
    
    entry = {
        "timestamp": timestamp,
        "clip": clip,
        "video_path": video_path,
        "results": results,
        "success_count": sum(bool(v) for v in results.values()),
        "total_platforms": len(results),
    }
    
    # Append to newline-delimited journal (easy to tail/grep)
    jsonl_path = os.path.join(LOGS_FOLDER, "uploads.jsonl")
    try:
        with open(jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        log.warning(f"Could not write to upload journal: {e}")
    
    # Update structured log array
    json_log = os.path.join(LOGS_FOLDER, "clips.json")
    try:
        if os.path.exists(json_log):
            with open(json_log, "r", encoding="utf-8") as f:
                log_data = json.load(f)
        else:
            log_data = []
        
        log_data.append(entry)
        
        with open(json_log, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.warning(f"Could not update structured upload log: {e}")


def mark_done(clip: dict, delete_files: bool = True) -> bool:
    """Remove a clipped entry from the queue.

    Identification is solely by ``filename``.  If multiple entries happen to
    share the same filename (which should never happen after deduplication)
    they all are removed to avoid orphaned duplicates.
    
    Args:
        clip: The clip dict to remove
        delete_files: If True, also delete the video file(s). If False, only
                      remove from clips.json (useful for partial uploads)
    
    Returns ``True`` if anything was removed from the queue, ``False`` otherwise.
    """
    fn = clip.get("filename")
    if not fn:
        log.error("mark_done called with clip lacking filename")
        return False

    clips = load_clips()
    remaining = [c for c in clips if c.get("filename") != fn]
    removed = len(clips) - len(remaining)
    if removed:
        save_clips(remaining)
        log.info(f"📋 Removed {removed} clip(s) with filename '{fn}' from clips.json ({len(remaining)} remaining)")
        
        if delete_files:
            # try deleting the file(s)
            path = os.path.join(CLIPS_FOLDER, fn)
            for candidate in (path, f"{os.path.splitext(path)[0]}_padded.mp4"):
                try:
                    if os.path.exists(candidate):
                        os.remove(candidate)
                        log.info(f"🗑️  Deleted source: {candidate}")
                except Exception as e:
                    log.warning(f"Could not delete {candidate}: {e}")
        else:
            log.info(f"💾 Kept video file for manual review: {fn}")
        
        return True
    else:
        log.warning(f"mark_done: clip '{fn}' not found in queue")
        return False


# ═══════════════════════════════════════════════════════════
#  Auth helpers
# ═══════════════════════════════════════════════════════════

def get_ig_client() -> IGClient:
    cl = IGClient()
    if os.path.exists(INSTAGRAM_SESSION_FILE):
        cl.load_settings(INSTAGRAM_SESSION_FILE)
        try:
            cl.get_timeline_feed()
            return cl
        except LoginRequired:
            log.warning("Instagram session expired, re-logging in ...")
            # Create fresh client instance for re-login
            cl = IGClient()
    
    # Load credentials from JSON file
    with open(INSTAGRAM_CREDENTIAL_FILE, "r", encoding="utf-8") as f:
        creds = json.load(f)
    
    try:
        cl.login(creds["username"], creds["password"])
        cl.dump_settings(INSTAGRAM_SESSION_FILE)
    except Exception as e:
        log.error(f"Instagram login failed: {e}")
        if "403" in str(e) or "login_required" in str(e):
            log.error("   Instagram may be blocking automated logins. Try:")
            log.error("   1. Delete ig_session.json and try again")
            log.error("   2. Verify credentials in ig_cred.json are correct")
            log.error("   3. Login manually to Instagram from your IP address first")
            log.error("   4. Wait a few hours before retrying (rate limit)")
        raise
    return cl


def get_youtube_client():
    credentials = None
    if os.path.exists(YOUTUBE_TOKEN_FILE):
        with open(YOUTUBE_TOKEN_FILE, "rb") as f:
            credentials = pickle.load(f)
        # ensure scopes match what we currently ask for
        if credentials and hasattr(credentials, "scopes"):
            existing = set(credentials.scopes or [])
            needed = set(YOUTUBE_SCOPES)
            if not needed.issubset(existing):
                log.warning("Existing YouTube token lacking required scopes; deleting and re-authenticating.")
                try:
                    os.remove(YOUTUBE_TOKEN_FILE)
                except OSError:
                    pass
                credentials = None
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                YOUTUBE_CLIENT_SECRETS, YOUTUBE_SCOPES
            )
            credentials = flow.run_local_server(port=0)
        with open(YOUTUBE_TOKEN_FILE, "wb") as f:
            pickle.dump(credentials, f)
    return googleapiclient.discovery.build("youtube", "v3", credentials=credentials)


def validate_youtube_channel(yt) -> bool:
    response = yt.channels().list(part="id,snippet", mine=True).execute()
    items = response.get("items", [])
    if not items:
        log.error("❌ YouTube failed: no authenticated channel found for current token.")
        return False

    channel = items[0]
    channel_id = channel.get("id", "")
    channel_title = channel.get("snippet", {}).get("title", "")
    channel_custom_url = channel.get("snippet", {}).get("customUrl", "")
    log.info(
        f"📺 YouTube auth channel: title='{channel_title}', id='{channel_id}', customUrl='{channel_custom_url}'"
    )

    expected = (YOUTUBE_CHANNEL_HANDLE or "").strip().lower().lstrip("@")
    actual = (channel_custom_url or "").strip().lower().lstrip("@")
    if expected and actual != expected:
        log.error(
            "❌ YouTube failed: authenticated channel does not match YOUTUBE_CHANNEL_HANDLE. "
            f"expected='@{expected}', actual='@{actual or 'unknown'}'"
        )
        log.error("   Delete yt_token.pickle, re-run auth, and login with the correct YouTube channel account.")
        return False

    return True


def probe_video_info(video_path: str) -> tuple[int, int, float] | tuple[None, None, None]:
    cmd = [
        FFPROBE_EXECUTABLE,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height:format=duration",
        "-of",
        "json",
        video_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        stream = (data.get("streams") or [{}])[0]
        fmt = data.get("format") or {}
        raw_w = stream.get("width")
        raw_h = stream.get("height")
        raw_dur = fmt.get("duration")
        if raw_w is None or raw_h is None or raw_dur is None:
            log.error("❌ ffprobe returned incomplete stream info (missing width/height/duration)")
            return None, None, None
        width = int(raw_w)
        height = int(raw_h)
        duration = float(raw_dur)
        return width, height, duration
    except Exception as e:
        log.error(f"❌ Could not inspect video with ffprobe: {e}")
        return None, None, None


def pad_video_to_vertical(video_path: str) -> str | None:
    """Pad a landscape clip with black bars to make it taller than it is
    wide.

    The output file is created alongside the original with a
    ``_padded`` suffix.  We choose a target height equal to twice the
    original width (a 1:2 aspect) which is safely taller than the
    original.  If FFmpeg fails or the resulting video still isn't vertical
    we return ``None``.
    """
    base, ext = os.path.splitext(video_path)
    out_path = f"{base}_padded{ext}"

    # get original dimensions
    w, h, _ = probe_video_info(video_path)
    if w is None or h is None:
        return None

    target_h = int(w * 2)
    cmd = [
        FFMPEG_EXECUTABLE,
        "-y",
        "-i",
        video_path,
        "-vf",
        f"pad={w}:{target_h}:(ow-iw)/2:(oh-ih)/2:black",
        "-c:a",
        "copy",
        out_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except Exception as e:
        log.error(f"❌ Could not pad video to vertical: {e}")
        return None

    # verify result is vertical
    w2, h2, _ = probe_video_info(out_path)
    if w2 is None or h2 is None or h2 <= w2:
        log.error(
            "❌ Padding did not produce a vertical video (still "
            f"{w2}x{h2})."
        )
        try:
            os.remove(out_path)
        except Exception:
            pass
        return None

    log.info(f"🔲 Padded {video_path} -> {out_path} with black bars to vertical")
    return out_path


def ensure_shorts_eligible(video_path: str) -> str | None:
    """Verify (and if possible fix) a clip so it can be uploaded as a
    YouTube Short.

    Returns the path that should be used for the upload.  ``None`` means
    the clip cannot be used.
    """
    width, height, duration = probe_video_info(video_path)
    if width is None or height is None or duration is None:
        return None

    if duration > 180:
        log.error(
            f"❌ YouTube Shorts max duration is 180s. Current duration is {duration:.2f}s."
        )
        return None

    # if clip is landscape, pad with black bars instead of rotating
    if height <= width:
        log.warning(
            f"📐 Video is not vertical ({width}x{height}); padding with black bars"
        )
        padded = pad_video_to_vertical(video_path)
        if not padded:
            # helper already logged the error
            return None
        return ensure_shorts_eligible(padded)

    # at this point we know height > width and duration is OK
    return video_path


def unique_ify_video(video_path: str, platform_tag: str = "generic") -> str | None:
    """Create a unique version of the video to defeat perceptual hash matching.

    Each call produces a different output because every parameter is
    randomised.  Use a different ``platform_tag`` for each platform so
    that Instagram, YouTube, and TikTok each receive a distinct file.

    Techniques applied:
    1. Strip all metadata.
    2. Random brightness (±0.03) and contrast (0.97–1.03) — wide enough
       to change the perceptual hash, still imperceptible to viewers.
    3. Random saturation tweak (0.97–1.03).
    4. Tiny random crop (1–3 px per side) then scale back — changes the
       pixel grid so DCT-based hashes differ.
    5. Re-encode video *and* audio (slight pitch-preserving tempo shift
       of ±0.5 %) so the audio fingerprint also changes.
    """
    base, ext = os.path.splitext(video_path)
    out_path = f"{base}_unique_{platform_tag}{ext}"

    # --- Video tweaks ---
    brightness = random.uniform(-0.03, 0.03)
    contrast = random.uniform(0.97, 1.03)
    saturation = random.uniform(0.97, 1.03)

    # Tiny random crop (1-3 px per side) → scale back to original dims
    w, h, _ = probe_video_info(video_path)
    crop_px = random.randint(1, 3)
    if w and h:
        cw, ch = w - 2 * crop_px, h - 2 * crop_px
        crop_filter = f"crop={cw}:{ch}:{crop_px}:{crop_px},scale={w}:{h}"
    else:
        crop_filter = None

    vf_parts = [f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}"]
    if crop_filter:
        vf_parts.append(crop_filter)
    vf_str = ",".join(vf_parts)

    # --- Audio tweak: tiny tempo shift (±0.5 %) to change audio fingerprint ---
    tempo = random.uniform(0.995, 1.005)
    af_str = f"atempo={tempo}"

    cmd = [
        FFMPEG_EXECUTABLE,
        "-y",
        "-i", video_path,
        "-map_metadata", "-1",        # Strip all metadata
        "-vf", vf_str,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "18",                 # High quality
        "-af", af_str,
        "-c:a", "aac", "-b:a", "192k",  # Re-encode audio
        out_path,
    ]

    try:
        log.info(
            f"✨ Uniquifying video [{platform_tag}]: {video_path} "
            f"(bright={brightness:+.3f}, contrast={contrast:.3f}, "
            f"sat={saturation:.3f}, crop={crop_px}px, tempo={tempo:.4f})"
        )
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return out_path
    except Exception as e:
        log.error(f"❌ Could not uniquify video [{platform_tag}]: {e}")
        return None


def generate_thumbnail(video_path: str, width: int, height: int, suffix: str) -> str | None:
    """Create a still image from ``video_path`` at the requested size.

    Seeks to the first frame of the video.
    The thumbnail is scaled to fit within ``width``x``height`` while
    preserving the original aspect ratio, padded with black bars to
    exactly match the target dimensions.
    """
    thumbs_dir = os.path.join(CLIPS_FOLDER, "_thumbnails")
    os.makedirs(thumbs_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(video_path))[0]
    thumbnail_path = os.path.join(thumbs_dir, f"{base}_{suffix}.jpg")

    # Seek to the first frame
    _w, _h, duration = probe_video_info(video_path)
    seek_secs = 0.0

    cmd = [
        FFMPEG_EXECUTABLE,
        "-y",
        "-ss", f"{seek_secs:.2f}",
        "-i", video_path,
        "-frames:v", "1",
        "-vf",
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black",
        thumbnail_path,
    ]
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        return thumbnail_path
    except Exception as e:
        log.error(f"❌ Could not generate thumbnail with ffmpeg: {e}")
        return None

# ═══════════════════════════════════════════════════════════
#  Upload functions — raw file, no conversion
# ═══════════════════════════════════════════════════════════

def unique_ify_caption(caption: str) -> str:
    """Randomize caption slightly to avoid exact-duplicate detection.

    Uses only invisible Unicode variation selectors — these are
    imperceptible to readers but alter the byte signature of the string.
    Cyrillic homoglyphs are intentionally avoided: modern spam classifiers
    flag mixed-script text and can trigger shadowbans.
    """
    # Append a random invisible Unicode character from the Variation Selector
    # block (U+FE00–FE0F) — zero visual impact, changes string hash.
    variation_selectors = [chr(c) for c in range(0xFE00, 0xFE10)]
    suffix = random.choice(variation_selectors)

    # Also insert a zero-width non-joiner at a random interior position
    # to further differentiate the byte sequence.
    if len(caption) > 10:
        pos = random.randint(5, len(caption) - 5)
        caption = caption[:pos] + "\u200C" + caption[pos:]

    return caption + suffix


def upload_instagram(video_path: str, clip: dict) -> bool:
    reset_daily_counts_if_needed()
    if _daily_counts["instagram"] >= _daily_target:
        log.warning(f"⚠️  Instagram daily limit reached ({_daily_target}/day) — skipping")
        return False
    
    try:
        thumbnail_path = generate_thumbnail(video_path, 1080, 1920, "instagram")
        if not thumbnail_path:
            return False

        ig = get_ig_client()
        unique_caption = unique_ify_caption(clip.get("caption", ""))
        ig.clip_upload(path=video_path, caption=unique_caption, thumbnail=thumbnail_path)
        _daily_counts["instagram"] += 1
        log.info(f"✅ Instagram: {clip.get('title', '?')} [{_daily_counts['instagram']}/{_daily_target} today]")
        log.info(f"🖼️ Instagram thumbnail uploaded: {thumbnail_path}")
        return True
    except Exception as e:
        log.error(f"❌ Instagram failed: {e}")
        return False


def upload_youtube(video_path: str, clip: dict) -> bool:
    reset_daily_counts_if_needed()
    if _daily_counts["youtube"] >= _daily_target:
        log.warning(f"⚠️  YouTube daily quota limit reached ({_daily_target}/day) — skipping")
        return False
    
    try:
        yt = get_youtube_client()
        if not validate_youtube_channel(yt):
            return False

        new_path = ensure_shorts_eligible(video_path)
        if not new_path:
            return False
        if new_path != video_path:
            log.debug(f"Using converted video file for upload: {new_path}")
            video_path = new_path

        # Landscape thumbnail (1280×720) — standard for YouTube Shorts
        thumbnail_path = generate_thumbnail(video_path, 1280, 720, "youtube")
        if not thumbnail_path:
            return False

        title_base = clip.get("title", "").strip()
        title = f"{title_base} #Shorts"
        if len(title) > 100:
            title = f"{title_base[:91].rstrip()} #Shorts"

        desc = unique_ify_caption(clip.get("caption", "").strip())
        if "#shorts" not in desc.lower():
            desc = f"{desc}\n\n#Shorts"
        desc = desc[:5000]

        tags = ["Shorts", "shorts", "AI", "Indonesia", "fyp", "viral"]

        req = yt.videos().insert(
            part="snippet,status,recordingDetails",
            body={
                "snippet": {
                    "title": title,
                    "description": desc,
                    "tags": tags,
                    "categoryId": YOUTUBE_CATEGORY_ID,
                    "defaultLanguage": YOUTUBE_DEFAULT_LANGUAGE,
                    "defaultAudioLanguage": YOUTUBE_DEFAULT_AUDIO_LANGUAGE,
                },
                "status": {
                    "privacyStatus": YOUTUBE_PRIVACY_STATUS,
                    "selfDeclaredMadeForKids": YOUTUBE_SELF_DECLARED_MADE_FOR_KIDS,
                    "license": YOUTUBE_LICENSE,
                    "embeddable": YOUTUBE_EMBEDDABLE,
                    "publicStatsViewable": YOUTUBE_PUBLIC_STATS_VIEWABLE,
                },
                "recordingDetails": {
                    "recordingDate": datetime.now(timezone.utc).replace(microsecond=0).isoformat()
                },
            },
            media_body=MediaFileUpload(
                video_path,
                mimetype="video/mp4",
                chunksize=8 * 1024 * 1024,
                resumable=True,
            ),
            notifySubscribers=YOUTUBE_NOTIFY_SUBSCRIBERS,
        )
        resp = req.execute()

        yt.thumbnails().set(
            videoId=resp["id"],
            media_body=MediaFileUpload(thumbnail_path, mimetype="image/jpeg"),
        ).execute()

        _daily_counts["youtube"] += 1
        log.info(f"✅ YouTube: https://youtube.com/shorts/{resp['id']} [{_daily_counts['youtube']}/{_daily_target} today]")
        log.info(f"🖼️ YouTube thumbnail uploaded: {thumbnail_path}")
        return True
    except Exception as e:
        error_str = str(e)
        if "quotaExceeded" in error_str or "403" in error_str:
            log.error("❌ YouTube failed: Daily API quota exceeded")
            log.error("   YouTube API has a daily quota limit (10,000 units/day)")
            log.error("   Video uploads cost 1,600 units each (~6 uploads/day max)")
            log.error("   Quota resets at midnight Pacific Time (PT)")
            log.error("   Solutions:")
            log.error("   1. Wait until quota resets (check: https://console.cloud.google.com/apis/api/youtube.googleapis.com/quotas)")
            log.error("   2. Request quota increase via Google Cloud Console")
            log.error("   3. Disable YouTube uploads temporarily in config.py")
        else:
            log.error(f"❌ YouTube failed: {e}")
        return False



def ensure_vertical(video_path: str) -> str | None:
    """Pad a landscape video with black bars until it is taller than it is wide.

    TikTok favors vertical uploads; if the source clip is wider than it is
    tall we call ``pad_video_to_vertical`` (which is already used for Instagram
    thumbnails) and recurse until the result is vertical.  ``None`` is
    returned on failure.
    """
    w, h, _ = probe_video_info(video_path)
    if w is None or h is None:
        return None
    if w > h:
        log.warning(
            f"📐 Video is not vertical ({w}x{h}); padding with black bars"
        )
        padded = pad_video_to_vertical(video_path)
        if not padded:
            return None
        return ensure_vertical(padded)
    return video_path


def upload_tiktok(video_path: str, clip: dict) -> bool:
    """Upload a clip using the tiktok-uploader library with TikTokUploader class.

    This uses the class-based API which provides better control over the upload
    process and proper context management.
    
    Each upload gets a fresh browser instance to avoid asyncio event loop corruption.
    """
    reset_daily_counts_if_needed()
    if _daily_counts["tiktok"] >= _daily_target:
        log.warning(f"⚠️  TikTok daily limit reached ({_daily_target}/day) — skipping")
        return False
    
    try:
        video_path = ensure_vertical(video_path)
        if not video_path:
            return False

        if not os.path.exists(TIKTOK_COOKIES_FILE):
            log.error(f"❌ TikTok failed: cookies file not found: {TIKTOK_COOKIES_FILE}")
            log.error("   Re-export cookies from tiktok.com and save to that exact path.")
            return False

        def cookies_expired(path: str) -> bool:
            now = time.time()
            expiring_soon = now + (7 * 24 * 60 * 60)  # Warn if expires within 7 days
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split("\t")
                        if len(parts) >= 7:
                            name = parts[5]
                            try:
                                expires = int(parts[4])
                            except ValueError:
                                continue
                            if name in ("sessionid", "ttwid"):
                                if expires < now:
                                    return True
                                elif expires < expiring_soon:
                                    days_left = (expires - now) / (24 * 60 * 60)
                                    log.warning(f"⚠️  TikTok cookie '{name}' expires in {days_left:.1f} days - refresh soon!")
            except Exception:
                return True
            return False

        if cookies_expired(TIKTOK_COOKIES_FILE):
            log.error("❌ TikTok failed: cookies file appears expired.")
            log.error("   Re-login to tiktok.com, export fresh cookies (Netscape), overwrite tiktok_cookies.txt, then retry.")
            return False

        with open(TIKTOK_COOKIES_FILE, "r", encoding="utf-8") as f:
            cookie_content = f.read()
        missing = [cookie for cookie in ("sessionid", "ttwid") if cookie not in cookie_content]
        if missing:
            log.error(f"❌ TikTok failed: cookies file missing required keys: {missing}")
            log.error("   Open tiktok.com in browser, login, then export cookies again (Netscape format).")
            return False

        thumbnail_path = generate_thumbnail(video_path, 1080, 1920, "tiktok")
        if not thumbnail_path:
            return False

        unique_caption = unique_ify_caption(clip.get("caption", ""))
        video_dict = {
            "path": video_path,
            "description": unique_caption[:2200],
            "cover": thumbnail_path,
        }

        # Create a fresh uploader instance for each upload and use context manager
        # to guarantee proper cleanup. This prevents asyncio event loop corruption
        # which was causing "Playwright Sync API inside asyncio loop" errors.
        with TikTokUploader(
            cookies=TIKTOK_COOKIES_FILE,
            headless=True,
            user_data_dir=TIKTOK_BROWSER_DATA_DIR,
        ) as uploader:
            # allow a couple of overall retries to handle flaky navigation or
            # temporary TikTok hiccups (also covers the video-set retry).
            failed = uploader.upload_videos([video_dict], num_retries=2, skip_interactivity=True)

            if failed:
                log.error(f"❌ TikTok upload failed: {failed}")
                return False

            _daily_counts["tiktok"] += 1
            log.info(f"✅ TikTok: {clip.get('title', '?')} [{_daily_counts['tiktok']}/{_daily_target} today]")
            log.info(f"🖼️ TikTok cover uploaded: {thumbnail_path}")
            return True
    except Exception as e:
        log.error(f"❌ TikTok failed: {e}")
        if "No valid authentication source found" in str(e):
            log.error("   TikTok cookies likely expired or no longer valid for upload.")
            log.error("   Re-login to tiktok.com, export fresh cookies (Netscape), overwrite tiktok_cookies.txt, then retry.")
        return False


# ═══════════════════════════════════════════════════════════
#  Job factory — one closure per time slot
# ═══════════════════════════════════════════════════════════


# Post ONE clip per time slot (prevents quota exhaustion)
def make_post_job(slot_time: str, tier: int, label: str):
    def post_job():
        # --- Day-of-week slot filtering ---
        refresh_active_slots()
        if slot_time not in _active_slots:
            log.info(f"⏭️  Slot {slot_time} not active today — skipping (organic variation)")
            return

        # --- Shadowban prevention: Jitter ---
        # Instead of posting exactly at the minute, wait 0-15 minutes
        jitter_mins = random.randint(0, 15)
        jitter_secs = random.randint(0, 59)
        total_wait = (jitter_mins * 60) + jitter_secs
        
        log.info(f"🎲 Shadowban protection: delaying post for {jitter_mins}m {jitter_secs}s...")
        time.sleep(total_wait)
        # ------------------------------------

        now_dt = datetime.now(pytz.timezone("Asia/Jakarta"))
        now = now_dt.strftime("%Y-%m-%d %H:%M:%S WIB")
        day_name = now_dt.strftime("%A")
        log.info("─" * 60)
        log.info(f"⏰  {slot_time} WIB  —  {label}  (tier {tier})  |  {now} | {day_name}")
        
        reset_daily_counts_if_needed()

        # Post ONE clip per slot (not all clips!)
        clip, video_path = get_clip_for_slot_and_day(slot_time, day_name)
        if clip is None:
            log.warning("📭 Queue empty — nothing to post.")
            return

        log.info(f"📹 Posting   : [score={clip.get('class_score', clip.get('clip_score', '?'))}]  rank {clip.get('rank', '?')}  — {clip.get('title', '?')}")
        log.info(f"   File      : {clip.get('filename')}")

        # --- Build the list of enabled platforms in RANDOM order ---
        # Randomising the order prevents a fixed IG→YT→TT fingerprint.
        platform_fns = []
        if ENABLE_INSTAGRAM:
            platform_fns.append(("instagram", upload_instagram))
        if ENABLE_YOUTUBE:
            platform_fns.append(("youtube", upload_youtube))
        if ENABLE_TIKTOK:
            platform_fns.append(("tiktok", upload_tiktok))
        random.shuffle(platform_fns)
        log.info(f"📤 Platform order this slot: {[p[0] for p in platform_fns]}")

        results = {}
        unique_files: list[str] = []  # track all generated files for cleanup

        for idx, (platform_name, upload_fn) in enumerate(platform_fns):
            # --- Minimum-gap enforcement ---
            elapsed = time.time() - _last_upload_ts[platform_name]
            if elapsed < MIN_GAP_BETWEEN_UPLOADS_SECS:
                wait_remaining = MIN_GAP_BETWEEN_UPLOADS_SECS - elapsed
                log.warning(
                    f"⏳ {platform_name}: last upload was {elapsed/60:.0f}m ago "
                    f"(min gap {MIN_GAP_BETWEEN_UPLOADS_SECS/60:.0f}m) — skipping this slot"
                )
                results[platform_name] = False
                continue

            # --- Per-platform unique video ---
            unique_path = unique_ify_video(video_path, platform_tag=platform_name)
            upload_path = unique_path if unique_path else video_path
            if unique_path:
                unique_files.append(unique_path)

            try:
                success = upload_fn(upload_path, clip)
                results[platform_name] = success
                if success:
                    _last_upload_ts[platform_name] = time.time()
            except Exception as e:
                log.error(f"Exception during {platform_name} upload: {e}")
                results[platform_name] = False

            # Random pause between platforms (60-180 s) — wider than before
            if idx < len(platform_fns) - 1:
                pause = random.randint(60, 180)
                log.info(f"💤 Pausing {pause}s before next platform...")
                time.sleep(pause)

        ok = sum(bool(v) for v in results.values())
        total = len(results)
        log.info(f"📊 {ok}/{total} platforms succeeded — {results}")
        
        # Log the upload attempt (success or failure)
        log_upload(clip, video_path, results)

        # Cleanup all unique-ified video files
        for uf in unique_files:
            if os.path.exists(uf):
                try:
                    os.remove(uf)
                    log.info(f"🗑️  Removed temporary unique-ified file: {uf}")
                except Exception as e:
                    log.warning(f"⚠️  Could not cleanup temporary file {uf}: {e}")

        if ok == total and total > 0:
            # All enabled platforms succeeded - remove from queue AND delete files
            log.info("✅ All platforms succeeded - removing from queue and deleting files")
            mark_done(clip, delete_files=True)
        elif ok > 0:
            # Partial success - remove from queue but KEEP files for manual review/retry
            log.warning(f"⚠️  Only {ok}/{total} platforms succeeded - removing from queue but keeping files")
            mark_done(clip, delete_files=False)
        else:
            # Total failure - keep everything for automatic retry
            log.error("❌ All enabled platforms failed - keeping entry and file for retry next slot")

    post_job.__name__ = f"post_{slot_time.replace(':', '')}"
    return post_job


def run_test_post(filename: str, platform: str | None = None):
    now = datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S WIB")
    log.info("=" * 60)
    log.info("🧪 TEST MODE — single file post")
    log.info(f"🕒 Triggered at: {now}")
    log.info(f"🎯 Filename    : {filename}")
    if platform:
        log.info(f"🎯 Platform    : {platform}")
    log.info("=" * 60)

    clip, video_path = get_clip_by_filename(filename)
    if clip is None:
        return

    results = {}
    # If platform is specified, only post to that platform
    if platform:
        if platform.lower() == "tiktok" and ENABLE_TIKTOK:
            results["tiktok"] = upload_tiktok(video_path, clip)
        elif platform.lower() == "instagram" and ENABLE_INSTAGRAM:
            results["instagram"] = upload_instagram(video_path, clip)
        elif platform.lower() == "youtube" and ENABLE_YOUTUBE:
            results["youtube"] = upload_youtube(video_path, clip)
        else:
            log.error(f"Platform '{platform}' not enabled or not recognized")
            return
    else:
        # If no platform specified, post to all enabled platforms
        if ENABLE_TIKTOK:
            results["tiktok"] = upload_tiktok(video_path, clip)
        if ENABLE_INSTAGRAM:
            results["instagram"] = upload_instagram(video_path, clip)
        if ENABLE_YOUTUBE:
            results["youtube"] = upload_youtube(video_path, clip)

    ok = sum(results.values())
    log.info(f"📊 Test result: {ok}/{len(results)} platforms succeeded — {results}")
    
    # Log the test upload
    log_upload(clip, video_path, results)

    if ok > 0:
        posted_at = datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S WIB")
        log.info(f"✅ Test post completed at {posted_at}")
        log.warning("🧪 TEST MODE: clips.json entry and source file were NOT deleted.")
        log.warning(f"🧹 Please delete manually if needed: {video_path}")
    else:
        log.error("❌ Test post failed on all enabled platforms.")


# ═══════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════

def rebuild_queue(master_path: str, dry_run: bool = False) -> None:
    """Regenerate ``CLIPS_JSON`` from a master file by removing already-posted clips.

    If ``dry_run`` is True the new queue is printed but not written.
    """
    if not os.path.exists(master_path):
        log.error(f"master file not found: {master_path}")
        return
    with open(master_path, "r", encoding="utf-8") as f:
        master = json.load(f)
    master = dedupe_clips(master)
    posted = get_posted_filenames()
    new_queue = [c for c in master if c.get("filename") not in posted]
    log.info(f"master contains {len(master)} clips, {len(posted)} already posted")
    log.info(f"resulting queue would have {len(new_queue)} clips")
    if dry_run:
        for c in new_queue:
            log.info(f"  {c.get('filename')} – {c.get('title')}")
        return
    save_clips(new_queue)
    log.info(f"wrote rebuilt queue to {CLIPS_JSON}")


def main(test_file: str | None = None,
         rebuild: bool = False,
         master: str | None = None,
         clean_orphans_flag: bool = False,
         prune_posted: bool = False,
         dry_run: bool = False,
         platform: str | None = None):
    log.info("=" * 60)
    log.info("  Cross-Platform Video Scheduler")
    log.info(f"  Timezone  : Asia/Jakarta (WIB, UTC+7)")
    log.info("  Platforms : " + ", ".join(
        p for p, en in [
            ("Instagram", ENABLE_INSTAGRAM),
            ("YouTube",   ENABLE_YOUTUBE),
            ("TikTok",    ENABLE_TIKTOK),
        ] if en
    ))
    log.info(f"  Clips dir : {CLIPS_FOLDER}")
    log.info("=" * 60)

    if not ensure_media_tools_ready():
        return

    # maintenance commands
    if rebuild:
        rebuild_queue(master or CLIPS_JSON, dry_run=dry_run)
        return

    if not os.path.exists(CLIPS_JSON):
        log.error(f"clips.json not found: {CLIPS_JSON}")
        return

    clips = load_clips()

    if prune_posted:
        posted = get_posted_filenames()
        before = len(clips)
        clips = [c for c in clips if c.get("filename") not in posted]
        removed = before - len(clips)
        if removed:
            log.info(f"🧹 pruned {removed} already-posted clip(s) from queue")
            if not dry_run:
                save_clips(clips)
        if dry_run:
            return

    if clean_orphans_flag:
        deleted = clean_orphan_files(clips)
        log.info(f"🧹 cleaned {deleted} orphan file(s)")
        return

    if test_file:
        run_test_post(test_file, platform=platform)
        return

    log.info(f"\n  {'Time':<8}  {'Tier':<8}  Description")
    log.info(f"  {'─'*8}  {'─'*8}  {'─'*30}")
    for t, tier, label in sorted(SCHEDULE_SLOTS, key=lambda s: s[0]):
        stars = "★" * (4 - tier)
        log.info(f"  {t:<8}  {stars:<8}  {label}")

    log.info(f"\n📋 {len(clips)} clips in queue at startup")
    log.info("   (clips.json is re-read fresh before every upload)")
    log.info("")
    log.info(f"📊 Daily uploads: {_DAILY_UPLOAD_POOL[0]}–{_DAILY_UPLOAD_POOL[-1]}/platform (drawn fresh each midnight, weighted toward {_DAILY_UPLOAD_POOL[1]}–{_DAILY_UPLOAD_POOL[2]})")
    log.info("")

    for slot_time, tier, label in SCHEDULE_SLOTS:
        job_fn = make_post_job(slot_time, tier, label)
        schedule.every().day.at(slot_time, "Asia/Jakarta").do(job_fn)

    log.info("🚀 Running — Ctrl+C to stop.\n")
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-platform video scheduler")
    parser.add_argument(
        "--test-file",
        help="Run one immediate test post using filename from clips.json; no deletion is performed.",
    )
    parser.add_argument(
        "--rebuild-queue",
        action="store_true",
        help="Regenerate the queue by removing clips that have already been posted. Requires --master if you want to use a different source."
    )
    parser.add_argument(
        "--master",
        help="Path to master clips.json to rebuild from (defaults to clips.json).",
    )
    parser.add_argument(
        "--clean-orphans",
        action="store_true",
        help="Delete any mp4 files in the clips folder that are not referenced in the current queue.",
    )
    parser.add_argument(
        "--prune-posted",
        action="store_true",
        help="Remove already-posted clips from the queue (based on upload logs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="When used with other maintenance flags, do not write changes; only show what would happen.",
    )
    parser.add_argument(
        "--platform",
        help="Specify a platform to post to (instagram, youtube, or tiktok). Used with --test-file to post to a single platform.",
    )
    args = parser.parse_args()
    main(
        test_file=args.test_file,
        rebuild=args.rebuild_queue,
        master=args.master,
        clean_orphans_flag=args.clean_orphans,
        prune_posted=args.prune_posted,
        dry_run=args.dry_run,
        platform=args.platform,
    )