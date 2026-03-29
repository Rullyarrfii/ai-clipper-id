"""
Person detection and close-up cropping using YOLO.

Detects persons in video frames and generates crop coordinates
to create close-up shots — essential for converting horizontal
video to vertical shorts/reels format.
"""

import subprocess
import json
from pathlib import Path
from typing import Any

from .utils import get_ffmpeg, get_ffprobe, log


def _get_video_dimensions(video_path: str) -> tuple[int, int, float]:
    """Get video width, height, and fps."""
    try:
        result = subprocess.run(
            [
                get_ffprobe(), "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,r_frame_rate",
                "-of", "json",
                video_path,
            ],
            capture_output=True, text=True, check=True,
        )
        data = json.loads(result.stdout)
        stream = data["streams"][0]
        w = int(stream["width"])
        h = int(stream["height"])
        fps_str = stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den) if float(den) > 0 else 30.0
        else:
            fps = float(fps_str)
        return w, h, fps
    except Exception as e:
        log("WARN", f"Could not get video dimensions: {e}")
        return 0, 0, 30.0


def detect_persons_in_clip(
    video_path: str,
    sample_interval: float = 1.0,
    confidence_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Detect persons in video frames using YOLO.

    Samples frames at regular intervals and returns bounding boxes
    for detected persons.

    Args:
        video_path: Path to the video clip.
        sample_interval: Seconds between sampled frames.
        confidence_threshold: Minimum YOLO confidence to accept detection.

    Returns:
        List of detections: [{"time": float, "boxes": [{"x1","y1","x2","y2","conf"}]}]
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        log("WARN", "ultralytics not installed. pip install ultralytics — skipping person detection")
        return []

    import cv2
    import numpy as np

    w, h, fps = _get_video_dimensions(video_path)
    if w == 0 or h == 0:
        return []

    # Load YOLO model (cached after first load)
    model = YOLO("yolov8n.pt")  # nano model — fast, good enough for person detection

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log("WARN", f"Could not open video: {video_path}")
        return []

    frame_interval = int(fps * sample_interval)
    detections: list[dict[str, Any]] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            time_sec = frame_idx / fps
            # Run YOLO — class 0 = person
            results = model(frame, verbose=False, classes=[0])

            boxes = []
            for result in results:
                for box in result.boxes:
                    conf = float(box.conf[0])
                    if conf >= confidence_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        boxes.append({
                            "x1": int(x1), "y1": int(y1),
                            "x2": int(x2), "y2": int(y2),
                            "conf": round(conf, 3),
                            "area": int((x2 - x1) * (y2 - y1)),
                        })

            if boxes:
                # Sort by area descending (largest person = likely speaker)
                boxes.sort(key=lambda b: -b["area"])
                detections.append({"time": round(time_sec, 2), "boxes": boxes})

        frame_idx += 1

    cap.release()
    return detections


def compute_crop_region(
    detections: list[dict[str, Any]],
    src_w: int,
    src_h: int,
    target_aspect: float = 9 / 16,
    padding_pct: float = 0.25,
    smoothing_window: int = 5,
) -> dict[str, int] | None:
    """Compute a stable crop region that keeps the main person centered.

    Algorithm:
    1. For each frame, find the largest person (likely speaker)
    2. Compute the center of that person across all frames
    3. Smooth the center position to avoid jittery panning
    4. Calculate crop box that centers on the person with padding

    Args:
        detections: Output from detect_persons_in_clip
        src_w: Source video width
        src_h: Source video height
        target_aspect: Target width/height ratio (9/16 for vertical)
        padding_pct: Extra padding around person (fraction of crop size)
        smoothing_window: Frames to average for smooth tracking

    Returns:
        {"x": int, "y": int, "w": int, "h": int} or None if no persons found
    """
    if not detections:
        return None

    # Collect center-x positions of the largest person per frame
    centers_x: list[float] = []
    centers_y: list[float] = []
    for det in detections:
        box = det["boxes"][0]  # largest person
        cx = (box["x1"] + box["x2"]) / 2.0
        cy = (box["y1"] + box["y2"]) / 2.0
        centers_x.append(cx)
        centers_y.append(cy)

    # Smooth center positions
    def _smooth(values: list[float], window: int) -> list[float]:
        if len(values) <= window:
            return values
        smoothed = []
        for i in range(len(values)):
            start = max(0, i - window // 2)
            end = min(len(values), i + window // 2 + 1)
            smoothed.append(sum(values[start:end]) / (end - start))
        return smoothed

    smooth_cx = _smooth(centers_x, smoothing_window)
    smooth_cy = _smooth(centers_y, smoothing_window)

    # Use median center as the stable crop point
    avg_cx = sorted(smooth_cx)[len(smooth_cx) // 2]
    avg_cy = sorted(smooth_cy)[len(smooth_cy) // 2]

    # Calculate crop dimensions
    # For horizontal→vertical: crop width is narrow, height is full
    src_aspect = src_w / src_h
    if target_aspect < src_aspect:
        # Source is wider than target → crop width (most common: landscape→portrait)
        crop_h = src_h
        crop_w = int(crop_h * target_aspect)
    else:
        # Source is taller than target → crop height
        crop_w = src_w
        crop_h = int(crop_w / target_aspect)

    # Ensure even dimensions (required by libx264)
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)

    # Center crop on the person
    crop_x = int(avg_cx - crop_w / 2)
    crop_y = int(avg_cy - crop_h / 2)

    # Clamp to video bounds
    crop_x = max(0, min(crop_x, src_w - crop_w))
    crop_y = max(0, min(crop_y, src_h - crop_h))

    return {"x": crop_x, "y": crop_y, "w": crop_w, "h": crop_h}


def compute_dynamic_crop_regions(
    detections: list[dict[str, Any]],
    src_w: int,
    src_h: int,
    target_aspect: float = 9 / 16,
    segment_duration: float = 3.0,
    smoothing_window: int = 5,
) -> list[dict[str, Any]]:
    """Compute per-segment crop regions for smooth person tracking.

    Instead of a single static crop, divides the clip into segments
    and computes crop position for each, enabling smooth panning
    to follow the active speaker.

    Returns:
        [{"time": float, "x": int, "y": int, "w": int, "h": int}, ...]
    """
    if not detections:
        return []

    # Group detections by time segment
    max_time = max(d["time"] for d in detections)
    n_segments = max(1, int(max_time / segment_duration) + 1)

    # Calculate crop dimensions (constant for all segments)
    src_aspect = src_w / src_h
    if target_aspect < src_aspect:
        crop_h = src_h
        crop_w = int(crop_h * target_aspect)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_aspect)
    crop_w = crop_w - (crop_w % 2)
    crop_h = crop_h - (crop_h % 2)

    segments: list[dict[str, Any]] = []
    prev_cx, prev_cy = src_w / 2, src_h / 2  # default center

    for seg_idx in range(n_segments):
        seg_start = seg_idx * segment_duration
        seg_end = seg_start + segment_duration

        # Find detections in this segment
        seg_dets = [d for d in detections if seg_start <= d["time"] < seg_end]

        if seg_dets:
            # Use median center of largest person
            cxs = [(d["boxes"][0]["x1"] + d["boxes"][0]["x2"]) / 2 for d in seg_dets]
            cys = [(d["boxes"][0]["y1"] + d["boxes"][0]["y2"]) / 2 for d in seg_dets]
            cx = sorted(cxs)[len(cxs) // 2]
            cy = sorted(cys)[len(cys) // 2]
            # Smooth transition from previous segment
            cx = prev_cx * 0.3 + cx * 0.7
            cy = prev_cy * 0.3 + cy * 0.7
            prev_cx, prev_cy = cx, cy
        else:
            cx, cy = prev_cx, prev_cy

        crop_x = int(cx - crop_w / 2)
        crop_y = int(cy - crop_h / 2)
        crop_x = max(0, min(crop_x, src_w - crop_w))
        crop_y = max(0, min(crop_y, src_h - crop_h))

        segments.append({
            "time": round(seg_start, 2),
            "x": crop_x, "y": crop_y,
            "w": crop_w, "h": crop_h,
        })

    return segments


def build_crop_filter(
    crop_region: dict[str, int] | None,
    target_w: int = 1080,
    target_h: int = 1920,
) -> str | None:
    """Build FFmpeg crop+scale filter string for static crop.

    Args:
        crop_region: Output from compute_crop_region
        target_w: Output width
        target_h: Output height

    Returns:
        FFmpeg filter string like "crop=608:1080:336:0,scale=1080:1920"
        or None if no crop needed
    """
    if not crop_region:
        return None

    x, y, w, h = crop_region["x"], crop_region["y"], crop_region["w"], crop_region["h"]
    return f"crop={w}:{h}:{x}:{y},scale={target_w}:{target_h}:flags=lanczos"


def needs_crop(src_w: int, src_h: int, target_aspect: str = "vertical") -> bool:
    """Check if the source video needs cropping for the target format.

    Args:
        src_w: Source width
        src_h: Source height
        target_aspect: "vertical" (9:16), "horizontal" (16:9), or "square" (1:1)
    """
    src_aspect = src_w / src_h if src_h > 0 else 1.0

    if target_aspect == "vertical":
        return src_aspect > 0.7  # source is landscape or square → needs vertical crop
    elif target_aspect == "horizontal":
        return src_aspect < 1.3  # source is portrait or square → needs horizontal crop
    elif target_aspect == "square":
        return abs(src_aspect - 1.0) > 0.15
    return False
