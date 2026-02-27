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
import shutil
import logging
import argparse
import schedule
import pytz
import subprocess
from datetime import datetime

from instagrapi import Client as IGClient
from instagrapi.exceptions import LoginRequired
import googleapiclient.discovery
import google_auth_oauthlib.flow
from google.auth.transport.requests import Request
from googleapiclient.http import MediaFileUpload
from tiktok_uploader.upload import upload_video as tiktok_upload_video

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
    FFMPEG_BIN,
    FFPROBE_BIN,
    ENABLE_INSTAGRAM, ENABLE_YOUTUBE, ENABLE_TIKTOK,
)

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


# ═══════════════════════════════════════════════════════════
#  Clip queue helpers
# ═══════════════════════════════════════════════════════════

def load_clips() -> list:
    with open(CLIPS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def save_clips(clips: list):
    with open(CLIPS_JSON, "w", encoding="utf-8") as f:
        json.dump(clips, f, indent=2, ensure_ascii=False)


def get_clip_for_slot(slot_time: str) -> tuple:
    """
    Pick the best clip for a given time slot.

    Matching logic:
      - Tier 1 slot → highest clip_score available
      - Tier 2 slot → 2nd highest
      - Tier 3 slot → 3rd highest (or best remaining)

    This ensures your best content always hits the best windows.
    clips.json is re-read fresh here every single call.
    """
    clips = load_clips()
    if not clips:
        return None, None

    slot_info = next((s for s in SCHEDULE_SLOTS if s[0] == slot_time), None)
    tier = slot_info[1] if slot_info else 3

    # Only consider clips whose file actually exists
    available = []
    for clip in clips:
        filename = clip.get("filename")
        if not filename:
            log.warning(f"Clip rank {clip['rank']} missing 'filename' field — skipping")
            continue
        path = os.path.join(CLIPS_FOLDER, filename)
        if os.path.exists(path):
            available.append((clip, path))
        else:
            log.warning(f"File not found for rank {clip['rank']}: {filename}")

    if not available:
        return None, None

    # Sort by clip_score descending
    available.sort(key=lambda x: x[0].get("clip_score", 0), reverse=True)

    # Tier 1 → index 0, Tier 2 → index 1, Tier 3 → index 2
    # Clamp to list size so it always returns something
    index = min(tier - 1, len(available) - 1)
    return available[index]


def get_clip_by_filename(filename: str) -> tuple:
    clips = load_clips()
    for clip in clips:
        if clip.get("filename") == filename:
            path = os.path.join(CLIPS_FOLDER, filename)
            if os.path.exists(path):
                return clip, path
            log.error(f"File for test post not found: {path}")
            return None, None
    log.error(f"No clip entry found in clips.json for filename: {filename}")
    return None, None


def mark_done(clip: dict):
    clips = load_clips()
    clips = [c for c in clips if c["rank"] != clip["rank"]]
    save_clips(clips)
    log.info(f"📋 Removed rank {clip['rank']} from clips.json  ({len(clips)} remaining)")


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
    
    # Load credentials from JSON file
    with open(INSTAGRAM_CREDENTIAL_FILE, "r", encoding="utf-8") as f:
        creds = json.load(f)
    
    cl.login(creds["username"], creds["password"])
    cl.dump_settings(INSTAGRAM_SESSION_FILE)
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
        width = int(stream.get("width"))
        height = int(stream.get("height"))
        duration = float(fmt.get("duration"))
        return width, height, duration
    except Exception as e:
        log.error(f"❌ Could not inspect video with ffprobe: {e}")
        return None, None, None


def ensure_shorts_eligible(video_path: str) -> bool:
    width, height, duration = probe_video_info(video_path)
    if width is None or height is None or duration is None:
        return False

    if height <= width:
        log.error(
            f"❌ YouTube Shorts requires vertical video. Current size is {width}x{height}."
        )
        return False

    if duration > 180:
        log.error(
            f"❌ YouTube Shorts max duration is 180s. Current duration is {duration:.2f}s."
        )
        return False

    return True


def generate_thumbnail(video_path: str, width: int, height: int, suffix: str) -> str | None:
    thumbs_dir = os.path.join(CLIPS_FOLDER, "_thumbnails")
    os.makedirs(thumbs_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(video_path))[0]
    thumbnail_path = os.path.join(thumbs_dir, f"{base}_{suffix}.jpg")
    cmd = [
        FFMPEG_EXECUTABLE,
        "-y",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-vf",
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black",
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

def upload_instagram(video_path: str, clip: dict) -> bool:
    try:
        thumbnail_path = generate_thumbnail(video_path, 1080, 1920, "instagram")
        if not thumbnail_path:
            return False

        ig = get_ig_client()
        ig.clip_upload(path=video_path, caption=clip["caption"], thumbnail=thumbnail_path)
        log.info(f"✅ Instagram: {clip['title']}")
        log.info(f"🖼️ Instagram thumbnail uploaded: {thumbnail_path}")
        return True
    except Exception as e:
        log.error(f"❌ Instagram failed: {e}")
        return False


def upload_youtube(video_path: str, clip: dict) -> bool:
    try:
        yt = get_youtube_client()
        if not validate_youtube_channel(yt):
            return False

        if not ensure_shorts_eligible(video_path):
            return False

        thumbnail_path = generate_thumbnail(video_path, 1280, 720, "youtube")
        if not thumbnail_path:
            return False

        title_base = clip["title"].strip()
        title = f"{title_base} #Shorts"
        if len(title) > 100:
            title = f"{title_base[:91].rstrip()} #Shorts"

        desc = clip["caption"].strip()
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
                    "recordingDate": datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
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

        log.info(f"✅ YouTube: https://youtube.com/shorts/{resp['id']}")
        log.info(f"🖼️ YouTube thumbnail uploaded: {thumbnail_path}")
        return True
    except Exception as e:
        log.error(f"❌ YouTube failed: {e}")
        return False


def upload_tiktok(video_path: str, clip: dict) -> bool:
    try:
        thumbnail_path = generate_thumbnail(video_path, 1080, 1920, "tiktok")
        if not thumbnail_path:
            return False

        if not os.path.exists(TIKTOK_COOKIES_FILE):
            log.error(f"❌ TikTok failed: cookies file not found: {TIKTOK_COOKIES_FILE}")
            log.error("   Re-export cookies from tiktok.com and save to that exact path.")
            return False

        with open(TIKTOK_COOKIES_FILE, "r", encoding="utf-8") as f:
            cookie_content = f.read()
        missing = [cookie for cookie in ("sessionid", "ttwid") if cookie not in cookie_content]
        if missing:
            log.error(f"❌ TikTok failed: cookies file missing required keys: {missing}")
            log.error("   Open tiktok.com in browser, login, then export cookies again (Netscape format).")
            return False

        tiktok_upload_video(
            filename=video_path,
            description=clip["caption"][:2200],
            cookies=TIKTOK_COOKIES_FILE,
            cover=thumbnail_path,
        )
        log.info(f"✅ TikTok: {clip['title']}")
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

def make_post_job(slot_time: str, tier: int, label: str):
    def post_job():
        now = datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S WIB")
        log.info("─" * 60)
        log.info(f"⏰  {slot_time} WIB  —  {label}  (tier {tier})  |  {now}")

        # clips.json re-read fresh every call
        clip, video_path = get_clip_for_slot(slot_time)

        if clip is None:
            log.warning("📭 Queue empty — nothing to post.")
            return

        log.info(f"📹 Posting   : [{clip.get('clip_score', '?')} score]  rank {clip['rank']}  — {clip['title']}")
        log.info(f"   File      : {clip.get('filename')}")

        results = {}
        if ENABLE_INSTAGRAM:
            results["instagram"] = upload_instagram(video_path, clip)
        if ENABLE_YOUTUBE:
            results["youtube"]   = upload_youtube(video_path, clip)
        if ENABLE_TIKTOK:
            results["tiktok"]    = upload_tiktok(video_path, clip)

        ok = sum(results.values())
        log.info(f"📊 {ok}/{len(results)} platforms succeeded — {results}")

        if ok > 0:
            mark_done(clip)
            try:
                os.remove(video_path)
                log.info(f"🗑️  Deleted source: {video_path}")
            except FileNotFoundError:
                log.warning(f"Source already missing during cleanup: {video_path}")
            except Exception as e:
                log.error(f"Could not delete {video_path}: {e}")
        else:
            log.error("⚠️  All platforms failed — keeping file for retry next slot.")

    post_job.__name__ = f"post_{slot_time.replace(':', '')}"
    return post_job


def run_test_post(filename: str):
    now = datetime.now(pytz.timezone("Asia/Jakarta")).strftime("%Y-%m-%d %H:%M:%S WIB")
    log.info("=" * 60)
    log.info("🧪 TEST MODE — single file post")
    log.info(f"🕒 Triggered at: {now}")
    log.info(f"🎯 Filename    : {filename}")
    log.info("=" * 60)

    clip, video_path = get_clip_by_filename(filename)
    if clip is None:
        return

    results = {}
    if ENABLE_INSTAGRAM:
        results["instagram"] = upload_instagram(video_path, clip)
    if ENABLE_YOUTUBE:
        results["youtube"] = upload_youtube(video_path, clip)
    if ENABLE_TIKTOK:
        results["tiktok"] = upload_tiktok(video_path, clip)

    ok = sum(results.values())
    log.info(f"📊 Test result: {ok}/{len(results)} platforms succeeded — {results}")

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

def main(test_file: str | None = None):
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

    if not os.path.exists(CLIPS_JSON):
        log.error(f"clips.json not found: {CLIPS_JSON}")
        return

    if test_file:
        run_test_post(test_file)
        return

    log.info(f"\n  {'Time':<8}  {'Tier':<8}  Description")
    log.info(f"  {'─'*8}  {'─'*8}  {'─'*30}")
    for t, tier, label in sorted(SCHEDULE_SLOTS, key=lambda s: s[0]):
        stars = "★" * (4 - tier)
        log.info(f"  {t:<8}  {stars:<8}  {label}")

    log.info(f"\n📋 {len(load_clips())} clips in queue at startup")
    log.info("   (clips.json is re-read fresh before every upload)")
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
    args = parser.parse_args()
    main(test_file=args.test_file)
