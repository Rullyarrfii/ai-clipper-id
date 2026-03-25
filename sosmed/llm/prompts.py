"""
LLM prompts for clip processing: translation, caption fixing, and deduplication.
"""

PROMPTS = {
    "Generate Single Clip Metadata": """Given a transcript and a single clip JSON object that covers the full video, rewrite only the metadata fields so they accurately reflect the spoken content.

Tasks:
1. Read the transcript carefully and identify the central claim, tension, or insight.
2. Rewrite these fields only: title, topic, caption, reason, hook.
3. Keep all timing, score, rank, and filename fields unchanged.
4. Return the same JSON array with exactly one item.

Field requirements:
- title: max 8 words, curiosity-driven, specific to the transcript
- topic: one clear sentence describing the core idea or debate
- caption: natural creator-style caption, accurate to the transcript, include relevant hashtags
- reason: 1-2 sentences explaining why this full-video clip is compelling or worth watching
- hook: the strongest opening line or a tightened version that stays faithful to the transcript

Language rules:
- If the transcript is in Indonesian, write title, topic, caption, reason, and hook in Indonesian.
- Do not use the filename or generic placeholders as metadata.
- Do not invent facts not supported by the transcript.

Return ONLY valid JSON array, no other text.""",

    "Translate to Indonesian": """Given clips data in JSON format, perform the following:

1. For each clip, analyze the language of the fields: topic, caption, hook
2. If any of these fields are NOT in Indonesian (Bahasa Indonesia), translate them to Indonesian
3. If they are already in Indonesian, keep them unchanged
4. Preserve the meaning, tone, and marketing appeal during translation
5. Keep all other fields unchanged (title, scores, timing, metadata)
6. Return the same JSON array with only the translated/updated fields

IMPORTANT: Do NOT translate the "title" field — keep it exactly as-is.

Example input:
[{"title": "How to cook rice", "topic": "Cooking Tips", "caption": "Learn fast!", "rank": 1, ...}]

Example output:
[{"title": "How to cook rice", "topic": "Tips Memasak", "caption": "Pelajari dengan cepat!", "rank": 1, ...}]

Return ONLY valid JSON array, no other text.""",

    "Fix Mismatched Caption/Topic": """Given clips data in JSON format, perform the following:

1. For each clip, verify that the caption and topic are cohesive and aligned
2. If the caption does NOT match the topic context, rewrite the caption to align with the topic
3. Ensure the caption is engaging, concise (max 280 chars), and relevant to the topic
4. Keep all other fields unchanged
5. Requirements:
   - Caption must be catchy and TikTok/Instagram-friendly
   - Topic must be clear and specific (not generic)
   - Both should tell a coherent story

Example:
Input: {"topic": "Efficient Meeting Strategies", "caption": "How to cook pasta perfectly", ...}
Output: {"topic": "Efficient Meeting Strategies", "caption": "5 ways to run meetings that don't waste time", ...}

Return ONLY valid JSON array with fixed clips, no other text.""",

    "Improve and Deduplicate Clips": """Given clips data in JSON format, perform the following:

1. Improve each clip's content:
   - Do NOT change the "title" field — keep it exactly as-is
   - Enhance captions with power words that drive engagement
   - Ensure topics are clear categories or themes
   - Strengthen hooks for maximum stop-scroll potential

2. Deduplicate only near-identical clips:
   - Only remove a clip if another clip covers the EXACT same moment or statement (not just the same broad theme)
   - Different clips covering similar topics from different angles are NOT duplicates — keep all of them
   - When forced to choose between near-identical clips, keep the one with the higher clip_score
   - Be conservative: when in doubt, keep the clip

3. Filter out only truly dead-weight clips:
   - Remove ONLY: pure opening/closing remarks with zero standalone value (e.g. "Welcome everyone, let's get started")
   - Remove ONLY: clips with clip_score < 40
   - Keep everything else — including Q&A, opinions, stories, tangents

4. Re-rank remaining clips by clip_score (highest first)

Return ONLY a valid JSON array of the kept and improved clips, no other text.""",
    "Translate Subtitle Phrases": """Given subtitle phrases in JSON format, translate each phrase to Indonesian (Bahasa Indonesia) AND add proper punctuation.

Input format: [{"id": 0, "text": "some phrase", "start": 0.5, "end": 2.0}, ...]

Rules:
- Translate the "text" field to Indonesian. If text is already in Indonesian, keep it unchanged.
- Add natural punctuation: periods (.), commas (,), question marks (?), exclamation marks (!) as appropriate for spoken content.
- Keep the word count roughly the same — punctuation marks are fine but do NOT add or remove words.
- Keep "id", "start", "end" fields exactly unchanged.
- Maintain natural conversational Indonesian.
- Keep the exact same number of items in the output array.

Example:
Input:  [{"id": 0, "text": "oh itu menarik sekali", "start": 0.0, "end": 1.5}]
Output: [{"id": 0, "text": "Oh, itu menarik sekali.", "start": 0.0, "end": 1.5}]

Return ONLY a JSON array with the exact same structure.
""",}


def get_prompt(section_name: str) -> str:
    """
    Get a prompt by section name.
    
    Args:
        section_name: One of "Translate to Indonesian", "Fix Mismatched Caption/Topic", 
                      or "Improve and Deduplicate Clips"
    
    Returns:
        The prompt text, or empty string if not found
    """
    return PROMPTS.get(section_name, "")
