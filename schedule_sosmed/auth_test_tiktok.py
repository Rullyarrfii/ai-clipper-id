#!/usr/bin/env python3
# ============================================================
#  auth_test_tiktok.py
#  Tests that your TikTok cookies file is valid.
#
#  How to get cookies:
#    1. Install "EditThisCookie" or "Cookie-Editor" in Chrome
#    2. Log into tiktok.com
#    3. Export cookies → save as tiktok_cookies.txt
#       (Netscape/Mozilla format — most exporters do this by default)
#
#  Cookies expire after ~30 days — re-export when uploads fail.
# ============================================================

import os
from config import TIKTOK_COOKIES_FILE


def check_cookies_file(path: str) -> bool:
    """Basic sanity check on the cookies file."""
    if not os.path.exists(path):
        print(f"❌ Cookies file not found: {path}")
        return False

    with open(path, "r") as f:
        content = f.read()

    # Key TikTok auth cookies
    required = ["sessionid", "ttwid"]
    missing  = [c for c in required if c not in content]

    if missing:
        print(f"⚠️  Missing cookies: {missing}")
        print("   Try re-exporting your cookies after logging in to tiktok.com")
        return False

    return True


def test_tiktok_upload(dry_run=True):
    """
    Attempts a test upload with tiktok-uploader.
    Set dry_run=True to only validate without actually uploading.
    """
    from tiktok_uploader.upload import upload_video

    if dry_run:
        print("ℹ️  dry_run=True — skipping actual upload, only checking library import.")
        print("✅ tiktok-uploader imported successfully.")
        print("\n   To do a real test upload, call:")
        print("   test_tiktok_upload(dry_run=False)")
        return

    # Upload a tiny test video
    test_video = "test_1sec.mp4"
    if not os.path.exists(test_video):
        print(f"❌ Test video not found: {test_video}")
        print("   Create a short test video or set dry_run=True")
        return

    result = upload_video(
        filename=test_video,
        description="Test upload — will delete #test",
        cookies=TIKTOK_COOKIES_FILE,
    )
    print(f"✅ Test upload result: {result}")


if __name__ == "__main__":
    print("=" * 50)
    print("  TikTok Auth Test")
    print("=" * 50)

    print(f"\n📂 Checking cookies file: {TIKTOK_COOKIES_FILE}")

    if check_cookies_file(TIKTOK_COOKIES_FILE):
        print("✅ Cookies file looks good!")

        print("\n📦 Checking tiktok-uploader installation ...")
        try:
            test_tiktok_upload(dry_run=True)
        except ImportError:
            print("❌ tiktok-uploader not installed.")
            print("   Run: pip install tiktok-uploader")
    else:
        print("\n📋 Steps to fix:")
        print("  1. Open Chrome and go to tiktok.com")
        print("  2. Log in to your account")
        print("  3. Open EditThisCookie / Cookie-Editor extension")
        print("  4. Export → Netscape format")
        print(f"  5. Save as: {TIKTOK_COOKIES_FILE}")
