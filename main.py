#!/usr/bin/env python3
"""
AI Video Clipper — Indonesian-optimized
────────────────────────────────────────
1. Transcribes video with faster-whisper (GPU/CPU auto, VAD, batched)
2. Pre-filters noise, fillers, duplicates (tuned for Bahasa Indonesia)
3. LLM automatically decides how many engaging clips to extract (max 100)
4. Extracts clips in parallel with ffmpeg

LLM priority: OpenRouter (free default) → Anthropic → OpenAI → Ollama

Usage:
  python main.py video.mp4
  python main.py video.mp4 --model large-v3
  python main.py video.mp4 --min 15 --max 90 --max-clips 50
  OPENROUTER_API_KEY=sk-... python main.py video.mp4
"""

from sosmed.cli import main

if __name__ == "__main__":
    main()
