#!/usr/bin/env python3
# ============================================================
#  auth_test_instagram.py
#  Run this ONCE to verify Instagram login works.
#  A session file will be saved so the main scheduler
#  doesn't need to re-login every run.
# ============================================================

from instagrapi import Client
from instagrapi.exceptions import LoginRequired
import os
import json

INSTAGRAM_CREDENTIAL_FILE="ig_credentials.json"
INSTAGRAM_SESSION_FILE="ig_session.json"


def get_ig_client() -> Client:
    cl = Client()
 
    if os.path.exists(INSTAGRAM_SESSION_FILE):
        print(f"📂 Loading saved session from {INSTAGRAM_SESSION_FILE} ...")
        cl.load_settings(INSTAGRAM_SESSION_FILE)
        try:
            cl.get_timeline_feed()          # lightweight call to test session
            print("✅ Session still valid — no re-login needed.")
            return cl
        except LoginRequired:
            print("⚠️  Session expired, re-logging in ...")

    with open(INSTAGRAM_CREDENTIAL_FILE, "r") as f:
        creds = json.load(f)
    cl.login(creds["username"], creds["password"])
    cl.dump_settings(INSTAGRAM_SESSION_FILE)
    print(f"✅ Logged in & session saved to {INSTAGRAM_SESSION_FILE}")
    return cl


if __name__ == "__main__":
    print("=" * 50)
    print("  Instagram Auth Test")
    print("=" * 50)

    try:
        cl = get_ig_client()
        me = cl.account_info()
        print(f"\n👤 Logged in as : @{me.username}")
        print(f"   Full name     : {me.full_name}")
        print("\n✅ Instagram auth is working correctly!")
    except Exception as e:
        print(f"\n❌ Auth failed: {e}")
