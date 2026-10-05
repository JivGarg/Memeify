"""MediaPipe pose + object detection, turned into per-arm observations for the throw detector.

Pose gives the shoulders, elbows, wrists and a rough hand outline (wrist, thumb,
index and pinky knuckles). An object detector looks for everyday objects around
the hands; an object whose box touches a hand's box counts as being held.

Uses the MediaPipe Tasks API. Its model files aren't bundled with the pip package,
so they are downloaded once into models/ on first run.
"""

from __future__ import annotations

import logging
import math
import shutil
import ssl
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import ConfigError, MemeifyError, TrackingConfig
from .gesture import SIDES, ArmObservation, Observation, Point

log = logging.getLogger("memeify")

Box = tuple[float, float, float, float]  # x0, y0, x1, y1 in frame pixels

_MODEL_ROOT = "https://storage.googleapis.com/mediapipe-models"
MODELS = {
    "pose_landmarker_lite.task": f"{_MODEL_ROOT}/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "pose_landmarker_full.task": f"{_MODEL_ROOT}/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "efficientdet_lite0.tflite": f"{_MODEL_ROOT}/object_detector/efficientdet_lite0/int8/latest/efficientdet_lite0.tflite",
    "efficientdet_lite2.tflite": f"{_MODEL_ROOT}/object_detector/efficientdet_lite2/int8/latest/efficientdet_lite2.tflite",
    "selfie_segmenter.tflite": f"{_MODEL_ROOT}/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
}

# Pose landmark indices per arm, from the subject's own point of view:
# shoulder, elbow, wrist, and the hand points (pinky, index, thumb knuckles).
POSE_ARM = {
    "left": (11, 13, 15, (17, 19, 21)),
    "right": (12, 14, 16, (18, 20, 22)),
}

# Detection crop: this many shoulder widths of padding around the hands.
CROP_PAD = 0.8
# Smallest hand box half-size, in shoulder widths, when only the wrist is visible.
MIN_HAND_RADIUS = 0.15
# Detections below tracking.object_score but above this are kept as near-misses for the debug view.
CANDIDATE_SCORE = 0.1


def ensure_model(name: str, models_dir: Path) -> Path:
    """Return the path to a model file, downloading it the first time."""
    path = models_dir / name
    if path.exists():
        return path
    models_dir.mkdir(parents=True, exist_ok=True)
    url = MODELS[name]
    log.info("downloading %s (first run only)...", name)
    try:
        import certifi  # installed with mediapipe; avoids missing-certificate errors on macOS
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    partial = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60, context=context) as resp, open(partial, "wb") as out:
            shutil.copyfileobj(resp, out)
        partial.replace(path)
    except OSError as e:
        partial.unlink(missing_ok=True)
        raise MemeifyError(f"could not download {name} from {url}: {e}. "
                             f"Check your connection, or download it by hand into {models_dir}.") from e
    return path


# --------------------------------------------------------------------------- geometry (pure, tested)

def hand_box(points: list[Point], scale: float, margin: float) -> Box:
    """Box around a hand from its pose points (wrist first), grown by `margin` shoulder widths.

    `scale` is the shoulder width in px. Pose only marks the knuckles, so the margin
    also has to cover the fingers and whatever they're holding.
    """
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    half = max(max(xs) - min(xs), max(ys) - min(ys)) / 2
    half = max(half, MIN_HAND_RADIUS * scale) + margin * scale
    return (cx - half, cy - half, cx + half, cy + half)


def overlap(a: Box, b: Box) -> float:
    """Intersection area of two boxes (0 if they don't touch)."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def detection_crop(boxes: list[Box], pad: float, frame_size: tuple[int, int], min_side: float) -> Box | None:
    """A square region around the given boxes, padded by `pad` px and kept inside the frame."""
    if not boxes:
        return None
    w, h = frame_size
    x0, y0 = min(b[0] for b in boxes) - pad, min(b[1] for b in boxes) - pad
    x1, y1 = max(b[2] for b in boxes) + pad, max(b[3] for b in boxes) + pad
    side = min(max(x1 - x0, y1 - y0, min_side), w, h)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    left = min(max(cx - side / 2, 0), w - side)
    top = min(max(cy - side / 2, 0), h - side)
    return (left, top, left + side, top + side)


@dataclass(frozen=True)
class DetectedObject:
    label: str
    score: float
    box: Box


def objects_in_hands(hands: dict[str, Box], objects: list[DetectedObject]) -> dict[str, DetectedObject | None]:
    """For each hand, the detected object overlapping it most (None if nothing touches it)."""
    result: dict[str, DetectedObject | None] = {}
    for side, hbox in hands.items():
        touching = [(overlap(hbox, o.box), o) for o in objects]
        touching = [(a, o) for a, o in touching if a > 0]
        result[side] = max(touching, key=lambda pair: pair[0])[1] if touching else None
    return result


# --------------------------------------------------------------------------- tracker

@dataclass
class TrackResult:
    observation: Observation
    hand_boxes: dict[str, Box] = field(default_factory=dict)
    objects: list[DetectedObject] = field(default_factory=list)
    candidates: list[DetectedObject] = field(default_factory=list)  # near-misses, debug view only
    crop: Box | None = None  # where the object detector looked (for --debug)


class _Timestamps:
    """VIDEO-mode MediaPipe tasks need strictly increasing integer millisecond timestamps."""

    def __init__(self):
        self._last = -1

    def __call__(self, t: float) -> int:
        ms = max(int(t * 1000), self._last + 1)
        self._last = ms
        return ms


class Tracker:
    def __init__(self, cfg: TrackingConfig, models_dir: Path, segmentation: bool):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self.cfg = cfg
        if cfg.pose_model not in ("lite", "full"):
            raise ConfigError(f"tracking.pose_model must be 'lite' or 'full', not '{cfg.pose_model}'")
        if cfg.object_model not in ("lite0", "lite2"):
            raise ConfigError(f"tracking.object_model must be 'lite0' or 'lite2', not '{cfg.object_model}'")
        if not cfg.object_classes:
            raise ConfigError("tracking.object_classes is empty, so nothing could ever be thrown")
        mode = vision.RunningMode.VIDEO
        conf = cfg.min_detection_confidence

        pose_path = ensure_model(f"pose_landmarker_{cfg.pose_model}.task", models_dir)
        det_path = ensure_model(f"efficientdet_{cfg.object_model}.tflite", models_dir)
        self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(pose_path)), running_mode=mode, num_poses=1,
            min_pose_detection_confidence=conf, min_pose_presence_confidence=conf, min_tracking_confidence=conf))
        self._detector = vision.ObjectDetector.create_from_options(vision.ObjectDetectorOptions(
            base_options=BaseOptions(model_asset_path=str(det_path)), running_mode=mode,
            score_threshold=min(cfg.object_score, CANDIDATE_SCORE), max_results=10,
            category_allowlist=list(cfg.object_classes)))
        self._pose_ts, self._det_ts, self._seg_ts = _Timestamps(), _Timestamps(), _Timestamps()

        self._segmenter = None
        if segmentation:
            seg_path = ensure_model("selfie_segmenter.tflite", models_dir)
            self._segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
                base_options=BaseOptions(model_asset_path=str(seg_path)), running_mode=mode,
                output_confidence_masks=True, output_category_mask=False))
        self._mask_cache: tuple[float, np.ndarray] | None = None
        self._last_image = None  # (t, mp.Image) from process(), reused by person_mask()

    def _mp_image(self, frame: np.ndarray, max_width: int | None = None):
        h, w = frame.shape[:2]
        if max_width and w > max_width:
            frame = cv2.resize(frame, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

    def process(self, frame: np.ndarray, t: float, detect_objects: bool = True) -> TrackResult:
        """Track one frame. With detect_objects=False (e.g. during the cooldown) only pose runs."""
        h, w = frame.shape[:2]
        image = self._mp_image(frame, self.cfg.process_width)
        self._last_image = (t, image)
        pose = self._pose.detect_for_video(image, self._pose_ts(t))

        def px(lm) -> Point:
            return (lm.x * w, lm.y * h)

        def visible(lm) -> bool:
            vis = lm.visibility if lm.visibility is not None else 1.0
            return vis >= self.cfg.min_visibility

        arms = {side: ArmObservation() for side in SIDES}
        hand_points: dict[str, list[Point]] = {}
        shoulder_width = None
        if pose.pose_landmarks:
            lms = pose.pose_landmarks[0]
            for side, (s, e, wr, fingers) in POSE_ARM.items():
                arm = arms[side]
                arm.shoulder = px(lms[s]) if visible(lms[s]) else None
                arm.elbow = px(lms[e]) if visible(lms[e]) else None
                arm.wrist = px(lms[wr]) if visible(lms[wr]) else None
                if arm.wrist is not None:
                    hand_points[side] = [arm.wrist] + [px(lms[i]) for i in fingers if visible(lms[i])]
            if arms["left"].shoulder and arms["right"].shoulder:
                shoulder_width = math.dist(arms["left"].shoulder, arms["right"].shoulder)

        scale = shoulder_width or 0.2 * h
        hands = {side: hand_box(pts, scale, self.cfg.touch_margin) for side, pts in hand_points.items()}
        crop = detection_crop(list(hands.values()), CROP_PAD * scale, (w, h), min_side=0.35 * h)
        if not (crop and detect_objects):
            return TrackResult(Observation(t, arms, shoulder_width), hands)
        found = self._detect(frame, crop, t)
        objects = [o for o in found if o.score >= self.cfg.object_score]
        candidates = [o for o in found if o.score < self.cfg.object_score]
        for side, obj in objects_in_hands(hands, objects).items():
            arms[side].holding = obj is not None
            arms[side].object_label = obj.label if obj else None
        return TrackResult(Observation(t, arms, shoulder_width), hands, objects, candidates, crop)

    def _detect(self, frame: np.ndarray, crop: Box, t: float) -> list[DetectedObject]:
        x0, y0, x1, y1 = (int(round(v)) for v in crop)
        result = self._detector.detect_for_video(self._mp_image(frame[y0:y1, x0:x1]), self._det_ts(t))
        objects = []
        for d in result.detections:
            b, c = d.bounding_box, d.categories[0]
            box = (x0 + b.origin_x, y0 + b.origin_y, x0 + b.origin_x + b.width, y0 + b.origin_y + b.height)
            objects.append(DetectedObject(c.category_name, c.score, box))
        return objects

    def person_mask(self, frame: np.ndarray, t: float) -> np.ndarray | None:
        """Person probability (0..1, float32) at full frame size, or None without a segmenter."""
        if self._segmenter is None:
            return None
        if self._mask_cache and self._mask_cache[0] == t:
            return self._mask_cache[1]
        if self._last_image and self._last_image[0] == t:
            image = self._last_image[1]
        else:
            image = self._mp_image(frame, self.cfg.process_width)
        result = self._segmenter.segment_for_video(image, self._seg_ts(t))
        mask = np.squeeze(result.confidence_masks[0].numpy_view())  # selfie_segmenter: one mask, person = 1
        mask = cv2.GaussianBlur(np.ascontiguousarray(mask, dtype=np.float32), (0, 0), 1.5)  # soften the cut-out edge
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
        self._mask_cache = (t, mask)
        return mask

    def close(self) -> None:
        for task in (self._pose, self._detector, self._segmenter):
            if task is not None:
                task.close()
