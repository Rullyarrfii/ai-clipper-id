#!/usr/bin/env python3
# ============================================================
#  auth_test_youtube.py
#  Run this ONCE to do the OAuth2 browser flow.
#  A token.pickle is saved so future runs are automatic.
#
#  Prerequisites:
#    1. Go to console.cloud.google.com
#    2. Create a project → Enable "YouTube Data API v3"
#    3. Create OAuth 2.0 credentials (Desktop app)
#    4. Download JSON → save as client_secrets.json
# ============================================================

import os, pickle
import googleapiclient.discovery
import google_auth_oauthlib.flow
from google.auth.transport.requests import Request
from config import YOUTUBE_CLIENT_SECRETS, YOUTUBE_TOKEN_FILE

# request broader scopes so we can call the channels endpoint during
# auth test.  If the saved token already exists with narrower permissions we
# explicitly remove it and force a re‑login (see notes below).
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",  # needed for channels/list
    # you can also use the single broad scope below if you control the app:
    # "https://www.googleapis.com/auth/youtube",
]


def get_youtube_client():
    credentials = None

    if os.path.exists(YOUTUBE_TOKEN_FILE):
        print(f"📂 Loading saved token from {YOUTUBE_TOKEN_FILE} ...")
        with open(YOUTUBE_TOKEN_FILE, "rb") as f:
            credentials = pickle.load(f)
        # check that the saved credentials actually include at least the
        # scopes we now require; older tokens might only have upload-rights and
        # will fail when we hit channels().list().
        if credentials and hasattr(credentials, "scopes"):
            existing = set(credentials.scopes or [])
            needed = set(SCOPES)
            if not needed.issubset(existing):
                print("⚠️  Existing token missing required scopes, deleting...")
                try:
                    os.remove(YOUTUBE_TOKEN_FILE)
                except OSError:
                    pass
                credentials = None

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            print("🔄 Token expired — refreshing ...")
            credentials.refresh(Request())
        else:
            print("🌐 Opening browser for OAuth2 login ...")
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                YOUTUBE_CLIENT_SECRETS, SCOPES
            )
            credentials = flow.run_local_server(port=0)

        with open(YOUTUBE_TOKEN_FILE, "wb") as f:
            pickle.dump(credentials, f)
        print(f"✅ Token saved to {YOUTUBE_TOKEN_FILE}")

    return googleapiclient.discovery.build("youtube", "v3", credentials=credentials)


if __name__ == "__main__":
    print("=" * 50)
    print("  YouTube Auth Test")
    print("=" * 50)

    try:
        youtube = get_youtube_client()

        # Fetch channel info to confirm auth works
        response = youtube.channels().list(part="snippet,statistics", mine=True).execute()
        channel = response["items"][0]
        snippet = channel["snippet"]
        stats   = channel["statistics"]

        print(f"\n📺 Channel : {snippet['title']}")
        print(f"   Subs    : {int(stats.get('subscriberCount', 0)):,}")
        print(f"   Videos  : {stats.get('videoCount', 0)}")
        print(f"\n✅ YouTube auth is working correctly!")
        print(f"\n⚠️  API Quota reminder:")
        print(f"   Each upload costs ~1,600 units.")
        print(f"   Daily free quota = 10,000 units (~6 uploads/day).")
    except Exception as e:
        print(f"\n❌ Auth failed: {e}")
