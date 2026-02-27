# ============================================================
#  config.py  –  Edit this file with your credentials & paths
# ============================================================

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Folder that contains clips.json + video files ───────────
CLIPS_FOLDER = os.path.join(BASE_DIR, "clips")   # <-- CHANGE THIS

# ── Upload times (24-h, server local time) ──────────────────
SCHEDULE_TIMES = ["06:00", "10:00", "14:00", "18:00", "20:00", "23:00"]

# ── Instagram ────────────────────────────────────────────────
INSTAGRAM_USERNAME = "samuelmat19@gmail.com"
INSTAGRAM_PASSWORD = "!Wonderfoo4258"
INSTAGRAM_SESSION_FILE = os.path.join(BASE_DIR, "ig_session.json")    # saved after first login

# ── YouTube ──────────────────────────────────────────────────
YOUTUBE_CLIENT_SECRETS = os.path.join(BASE_DIR, "client_secrets.json")  # downloaded from Google Cloud
YOUTUBE_TOKEN_FILE     = os.path.join(BASE_DIR, "yt_token.pickle")      # saved after first auth
YOUTUBE_CATEGORY_ID    = "28"                   # 22 = People & Blogs
YOUTUBE_CHANNEL_HANDLE = "@samuel.koesnadi"     # required: scheduler validates authenticated channel
YOUTUBE_PRIVACY_STATUS = "public"               # public | private | unlisted
YOUTUBE_DEFAULT_LANGUAGE = "id"
YOUTUBE_DEFAULT_AUDIO_LANGUAGE = "id"
YOUTUBE_LICENSE = "youtube"                     # youtube | creativeCommon
YOUTUBE_EMBEDDABLE = True
YOUTUBE_PUBLIC_STATS_VIEWABLE = True
YOUTUBE_NOTIFY_SUBSCRIBERS = True
YOUTUBE_SELF_DECLARED_MADE_FOR_KIDS = False

# ── TikTok ───────────────────────────────────────────────────
TIKTOK_COOKIES_FILE = os.path.join(BASE_DIR, "tiktok_cookies.txt")      # exported from browser

# ── FFmpeg video output settings ────────────────────────────
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")
VIDEO_RESOLUTION = "1080:1920"   # vertical 9:16
VIDEO_FPS        = 30
MAX_DURATION_YT  = 180           # YouTube Shorts max = 3 min
MAX_DURATION_IG  = 90            # Instagram Reels max = 90 sec
MAX_DURATION_TT  = 180           # TikTok max (3 min; up to 10 min allowed)

# ── Which platforms to post to ───────────────────────────────
ENABLE_INSTAGRAM = True
ENABLE_YOUTUBE   = True
ENABLE_TIKTOK    = True

# ── Delete video file after successful upload to ALL platforms?
DELETE_AFTER_UPLOAD = True
