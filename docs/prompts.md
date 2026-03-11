# Clip Processing Prompts

## Fix

```
Given a list of video clips in JSON format, do the following:

1. Deduplicate topics that overlap or are too similar
2. Rescore all scores (score_hook, score_retention, score_shareability, score_entertainment, score_clarity, clip_score) based on social media engagement probability (stop-scroll power, shareability, relatability, hook strength) — replace the original values, do not add new columns
3. Pick only the good ones, no hard limit on count
4. Improve title, topic, caption, and hook where possible — fix any mismatched caption/topic instead of skipping
5. Keep all original columns per entry (score_hook, score_retention, score_shareability, score_entertainment, score_clarity, clip_score, filename, start, end, _llm_start, _llm_end, reason)
6. Skip only: webinar filler clips (opening, closing, Q&A, sapa peserta)
7. Output as JSON
```
