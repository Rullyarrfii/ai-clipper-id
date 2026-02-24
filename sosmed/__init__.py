"""
AI Video Clipper — Indonesian-optimized video clip extraction.

Transcribes video with faster-whisper, filters noise/fillers, uses LLM to
identify engaging clips, then extracts them in parallel with ffmpeg.

Supports OpenRouter (free default), Anthropic, OpenAI, and local Ollama.
"""

from .transcription import transcribe
from .prefilter import prefilter_segments
from .llm import find_clips
from .extraction import extract_clips
from .utils import log

__version__ = "0.1.0"
__all__ = [
    "transcribe",
    "prefilter_segments",
    "find_clips",
    "extract_clips",
    "log",
]
