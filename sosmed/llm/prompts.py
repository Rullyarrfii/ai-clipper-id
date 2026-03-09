"""
LLM prompts for clip processing: translation, caption fixing, and deduplication.
"""

PROMPTS = {
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
   - Refine titles to be engaging and specific
   - Enhance captions with power words that drive engagement
   - Ensure topics are clear categories or themes
   - Strengthen hooks for maximum stop-scroll potential

2. Deduplicate by topic:
   - Group clips by similar topics
   - For each group, keep ONLY the highest-scoring clip (by clip_score)
   - Discard duplicates with similar topics but lower scores

3. Filter out undesirable clips:
   - Remove: webinar filler (opening statements, closing remarks, Q&A preambles, generic welcomes)
   - Remove: low-value clips with clip_score < 50
   - Keep: substantive content clips

4. Re-rank remaining clips by clip_score (highest first)

Requirements:
- Each topic should appear only once in final output
- For kept clips, consolidate topic names to be clear and marketable
- Return clips with refined titles, captions, and topics

Return ONLY a valid JSON array of deduplicated and improved clips, no other text.""",
    "Translate Subtitle Phrases": """Given subtitle phrases in JSON format, translate each phrase to Indonesian (Bahasa Indonesia).

Input format: [{"id": 0, "text": "some phrase", "start": 0.5, "end": 2.0}, ...]

Rules:
- Translate ONLY the "text" field to Indonesian
- Keep "id", "start", "end" fields exactly unchanged
- If text is already in Indonesian, keep it unchanged
- Maintain natural conversational Indonesian
- Keep the exact same number of items in the output array

Return ONLY valid JSON array with translated phrases, no other text.""",
}


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
