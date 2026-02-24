# AI Video Clipper — Indonesian Edition

An intelligent video-to-clips converter that uses AI transcription and LLM analysis to automatically extract engaging short clips from longer videos. Optimized for Indonesian-language content.

## Features

- 🎬 **Automatic Transcription** — Uses `faster-whisper` for accurate, fast speech-to-text (GPU/CPU auto-detection)
- 🧠 **AI-Powered Clip Extraction** — LLM automatically decides which segments are most engaging (max 100 clips)
- 🇮🇩 **Indonesian Optimized** — Pre-filters noise, filler words, and duplicates tuned for Bahasa Indonesia
- ⚡ **Parallel Processing** — Extracts multiple clips simultaneously using FFmpeg
- 🔌 **Multi-LLM Support** — Works with OpenRouter, Anthropic, OpenAI, or Ollama
- 📊 **Smart Ranking** — Clips are ranked by engagement score with compelling hooks extracted

## Project Structure

```
sosmed/
├── main.py              # Main video processing script
├── clips/
│   └── clips.json       # Generated clip metadata and timestamps
└── videos/              # Input video directory
```

## Setup

### Requirements

- Python 3.10+
- FFmpeg
- CUDA/cuDNN (optional, for GPU acceleration)

### Installation

```bash
# Clone or navigate to the project
cd sosmed

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

**LLM Priority:** The script automatically uses available LLM providers in this order:
1. **OpenRouter** (free default) — Set `OPENROUTER_API_KEY` environment variable
2. **Anthropic** — Set `ANTHROPIC_API_KEY` environment variable
3. **OpenAI** — Set `OPENAI_API_KEY` environment variable
4. **Ollama** — Run locally (no API key needed)

Example:
```bash
export OPENROUTER_API_KEY=sk-...
python main.py video.mp4
```

## Usage

### Basic Usage

```bash
python main.py videos/your-video.mp4
```

### Advanced Options

```bash
# Specify whisper model size
python main.py video.mp4 --model large-v3

# Set custom clip duration range (in seconds)
python main.py video.mp4 --min 15 --max 90

# Limit number of clips to extract
python main.py video.mp4 --max-clips 50

# Specify language for transcription
python main.py video.mp4 --lang id
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | auto | Whisper model size (tiny, small, base, medium, large) |
| `--min` | 30 | Minimum clip duration (seconds) |
| `--max` | 120 | Maximum clip duration (seconds) |
| `--max-clips` | 100 | Maximum number of clips to extract |
| `--lang` | auto | Language code (e.g., `id` for Indonesian) |

## Output

The script generates a `clips/clips.json` file with extracted clips:

```json
[
  {
    "rank": 1,
    "start": 2078.1,
    "end": 2138.5,
    "title": "Perkembangan AI dari Tahun ke Tahun",
    "reason": "Visualisasi menarik tentang kemajuan AI yang dramatis dalam waktu singkat",
    "hook": "2023, bentukannya masih seperti ini. Sepagetti sama mulut nggak ada yang tahu.",
    "engagement_score": 88
  }
]
```

**Fields:**
- `rank` — Priority ranking (1 = highest engagement)
- `start` / `end` — Clip timestamps in seconds
- `title` — Generated title for the clip
- `reason` — Why the LLM found this segment engaging
- `hook` — Opening quote to grab attention
- `engagement_score` — Score 0-100 indicating potential viral appeal

## Example Workflow

```bash
# Process a webinar about AI in industry
python main.py "videos/AI di Dunia Industri [AI Webinar Series - Eps 1].mp4" --lang id

# Results appear in clips/clips.json with timestamps, titles, and hooks
# Use these to:
# - Create short-form content for social media (TikTok, Reels, Shorts)
# - Identify key moments in long-form content
# - Generate video summaries automatically
```

## Performance Tips

- **GPU Acceleration**: Install CUDA for ~10x faster transcription
- **Batch Processing**: Process multiple videos with a loop
- **Model Size**: Use `--model base` for faster processing, `--model large-v3` for better accuracy

## Troubleshooting

**Issue**: Slow transcription  
→ Install CUDA or use a smaller model with `--model base`

**Issue**: No LLM API key error  
→ Set one of the LLM provider variables (see Configuration)

**Issue**: FFmpeg not found  
→ Install FFmpeg: `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Ubuntu)

## License

MIT License

Copyright (c) 2026 Samuel Koesnadi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Author

Samuel Koesnadi

---

**Questions or improvements?** Feel free to open an issue or submit a pull request!
