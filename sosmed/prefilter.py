"""
Segment pre-filtering: removes noise, duplicates, and low-value segments.
"""

from typing import Any

from .utils import FILLER_RE


def _wps(seg: dict[str, Any]) -> float:
    """Calculate words per second."""
    d = seg["end"] - seg["start"]
    return len(seg["text"].split()) / d if d > 0 else 0.0


def _jaccard(a: str, b: str) -> float:
    """Calculate Jaccard similarity between two strings."""
    sa, sb = set(a.lower().split()), set(b.lower().split())
    union = sa | sb
    return len(sa & sb) / len(union) if union else 0.0


def prefilter_segments(
    segments: list[dict[str, Any]],
    *,
    min_words: int = 3,
    min_duration: float = 0.5,
    max_no_speech: float = 0.7,
    min_wps: float = 0.3,
    max_wps: float = 12.0,
    dup_threshold: float = 0.85,
    merge_gap: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Filter low-value segments before LLM analysis.

    Filters: too-short, too-few-words, high no-speech, pure filler,
    abnormal speech rate, near-duplicate. Then merges close neighbours.
    """
    kept: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    seen: list[str] = []
    n_dropped = 0

    def _drop(reason: str) -> None:
        nonlocal n_dropped
        n_dropped += 1
        reasons[reason] = reasons.get(reason, 0) + 1

    for seg in segments:
        txt = seg["text"].strip()
        dur = seg["end"] - seg["start"]
        wps = _wps(seg)
        nsp = seg.get("no_speech_prob", 0.0)

        if dur < min_duration:          _drop("short_dur"); continue
        if len(txt.split()) < min_words:_drop("few_words"); continue
        if nsp > max_no_speech:         _drop("no_speech"); continue
        if FILLER_RE.match(txt):        _drop("filler"); continue
        if wps < min_wps:               _drop("slow_speech"); continue
        if wps > max_wps:               _drop("hallucination"); continue
        if any(_jaccard(txt, p) >= dup_threshold for p in seen[-12:]):
            _drop("duplicate"); continue

        seen.append(txt)
        kept.append(seg)

    # Merge adjacent segments separated by tiny gaps
    merged: list[dict[str, Any]] = []
    for seg in kept:
        if merged and (seg["start"] - merged[-1]["end"]) <= merge_gap:
            prev = merged[-1]
            prev["end"] = seg["end"]
            prev["text"] += " " + seg["text"]
            prev["words"] = prev.get("words", []) + seg.get("words", [])
        else:
            merged.append(dict(seg))

    stats: dict[str, Any] = {
        "original": len(segments),
        "kept": len(merged),
        "dropped": n_dropped,
        "drop_pct": f"{n_dropped / len(segments) * 100:.1f}%" if segments else "0%",
        "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }
    return merged, stats
