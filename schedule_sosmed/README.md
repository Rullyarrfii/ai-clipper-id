# Cross-Platform Video Scheduler

Automatically uploads videos to **Instagram Reels**, **YouTube Shorts**, and **TikTok**
6 times a day, reading from your ranked clips folder.

---

## File Structure

```
uploader/
├── config.py                  ← Edit this first!
├── scheduler.py               ← Main script (run this)
├── auth_test_instagram.py     ← Run once to verify IG auth
├── auth_test_youtube.py       ← Run once to verify YT auth
├── auth_test_tiktok.py        ← Run once to verify TT auth
├── requirements.txt
└── client_secrets.json        ← Download from Google Cloud Console

your-clips-folder/
├── clips.json
├── rank01_Title_final.mp4
├── rank02_Title_final.mp4
└── ...
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Edit config.py
Open `config.py` and set:
- `CLIPS_FOLDER` → path to your folder with clips.json + videos
- `FFMPEG_BIN` and `FFPROBE_BIN` → executable name/path (for Windows usually `ffmpeg.exe` and `ffprobe.exe` full path if not in PATH)
- Instagram username & password
- YouTube client_secrets.json path
- TikTok cookies file path
- Toggle `ENABLE_INSTAGRAM / ENABLE_YOUTUBE / ENABLE_TIKTOK`

### 3. Auth setup — run each once

#### Instagram
```bash
python auth_test_instagram.py
```
Logs in and saves a session file. Re-run if you get login errors.

#### YouTube
```bash
python auth_test_youtube.py
```
Opens your browser for OAuth2. After approving, a token is saved.

> ⚠️ **Scope warning:** the script now requests both `youtube.upload` and
> `youtube.readonly` (or the broader `youtube` scope) because the
> `channels().list(mine=True)` call requires read access to channel data.
> If you previously ran the auth flow with only the upload scope you'll see
> 403 errors. In that case delete the token file (`yt_token.pickle`) and
> run the command above again; the browser will open and you can re‑authorize
> with the correct permissions.


**Prerequisites:**
1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create a project → Enable **YouTube Data API v3**
3. Credentials → Create → **OAuth 2.0 Client ID** (Desktop app)
4. Download JSON → save as `client_secrets.json` in this folder

#### TikTok
```bash
python auth_test_tiktok.py
```
Checks your cookies file is valid.

**Getting TikTok cookies:**
1. Install **EditThisCookie** (Chrome) or **Cookie-Editor** (Firefox)
2. Log into tiktok.com
3. Export cookies → Netscape format
4. Save as `tiktok_cookies.txt`

> TikTok cookies expire every ~30 days. Re-export when uploads fail.

---

## Run the Scheduler

```bash
python scheduler.py
```

To keep it running in the background:
```bash
# Linux/macOS
nohup python scheduler.py > /dev/null 2>&1 &

# Or with screen
screen -S uploader
python scheduler.py
# Ctrl+A then D to detach
```

---

## How It Works

1. At each scheduled time, the script reads `clips.json`
2. Picks the **lowest rank** clip that has a matching video file
3. Converts it to platform-specific specs using FFmpeg:
   - Instagram: max 90s, 1080×1920
   - YouTube: max 180s, 1080×1920 (landscape clips are automatically padded with black bars)
   - TikTok: max 180s, 1080×1920

   **Note:** if a video is horizontal the scheduler will automatically
   add black bars to make it vertical; the original file remains untouched.
   TikTok cookies are also inspected for expiry so you get a clear message
   prompting a re-export instead of a generic auth failure.
4. Uploads to all enabled platforms
5. **Deletes the source `.mp4`** and removes the entry from `clips.json`

---

## Logs

All activity is logged to `uploader.log` and printed to console.

---

## Platform Limits

| Platform       | Max Duration | Notes                              |
|----------------|-------------|-------------------------------------|
| Instagram Reels| 90 seconds  | Longer clips are trimmed            |
| YouTube Shorts | 3 minutes   | Must have `#Shorts` in title/desc   |
| TikTok         | 3 minutes   | Up to 10 min with creator account   |

## API Quota (YouTube)
- Free quota: **10,000 units/day**
- Each upload costs: **~1,600 units**
- 6 uploads/day = ~9,600 units (just within limit)
- Request increase at [console.cloud.google.com](https://console.cloud.google.com) if needed
