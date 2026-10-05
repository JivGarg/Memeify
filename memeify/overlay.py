"""Preview-only drawing: the paused banner, and with --debug the tracking and trigger state.

Everything here draws on the scaled-down preview image, never on the virtual camera output.
"""

from __future__ import annotations

import cv2
import numpy as np

from .gesture import SIDES, ArmObservation, Phase, Point, ThrowDetector, ThrowEvent
from .tracking import Box, TrackResult

ARM_COLOR = {"left": (255, 170, 40), "right": (40, 190, 255)}  # BGR
PHASE_COLOR = {Phase.IDLE: (200, 200, 200), Phase.HOLDING: (0, 215, 255), Phase.SWUNG: (0, 120, 255)}
HELD_COLOR, EMPTY_COLOR, OBJECT_COLOR = (80, 230, 80), (235, 235, 235), (255, 90, 255)
NEAR_MISS_COLOR = (160, 70, 160)  # dim pink: seen, but not confidently enough to count
THROW_MARKER_SECONDS = 1.5


def _pt(p: Point, s: float) -> tuple[int, int]:
    return int(p[0] * s), int(p[1] * s)


def _rect(img: np.ndarray, box: Box, s: float, color, thickness: int = 1) -> None:
    cv2.rectangle(img, _pt(box[:2], s), _pt(box[2:], s), color, thickness, cv2.LINE_AA)


def text(img: np.ndarray, s: str, org: tuple[int, int], color=(255, 255, 255), size: float = 0.5) -> None:
    # Drop shadow rather than a thick outline: OpenCV 5 spaces thick text wider than thin text.
    cv2.putText(img, s, (org[0] + 1, org[1] + 1), cv2.FONT_HERSHEY_SIMPLEX, size, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)


def draw_paused(img: np.ndarray, hint: str) -> None:
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 30), (0, 0, 160), -1)
    text(img, f"DETECTION PAUSED  ({hint} to resume)", (10, 20))


def draw_debug(img: np.ndarray, scale: float, track: TrackResult | None, detector: ThrowDetector,
               t: float, fps: float, last_throw: tuple[ThrowEvent, Point] | None) -> None:
    p = detector.params
    if track is not None:
        obs = track.observation
        if track.crop:
            _rect(img, track.crop, scale, (110, 110, 110))  # where the object detector looked
        for obj in track.candidates:
            _rect(img, obj.box, scale, NEAR_MISS_COLOR, 1)
            text(img, f"{obj.label}? {obj.score:.2f}", _pt((obj.box[0], obj.box[1] - 6), scale), NEAR_MISS_COLOR, 0.4)
        for obj in track.objects:
            _rect(img, obj.box, scale, OBJECT_COLOR, 2)
            text(img, f"{obj.label} {obj.score:.2f}", _pt((obj.box[0], obj.box[1] - 6), scale), OBJECT_COLOR, 0.45)

        for side in SIDES:
            arm, color = obs.arms.get(side, ArmObservation()), ARM_COLOR[side]
            joints = [j for j in (arm.shoulder, arm.elbow, arm.wrist) if j is not None]
            for a, b in zip(joints, joints[1:]):
                cv2.line(img, _pt(a, scale), _pt(b, scale), color, 2, cv2.LINE_AA)
            if arm.shoulder is not None and detector.scale:
                # The shoulder line the wrist has to cross.
                sx, sy = arm.shoulder
                half = 0.7 * detector.scale
                cv2.line(img, _pt((sx - half, sy), scale), _pt((sx + half, sy), scale), color, 1, cv2.LINE_AA)
            if side in track.hand_boxes:
                box = track.hand_boxes[side]
                _rect(img, box, scale, HELD_COLOR if arm.holding else EMPTY_COLOR, 3 if arm.holding else 2)
                text(img, "holding" if arm.holding else "hand", _pt((box[0] + 4, box[3] - 6), scale),
                     HELD_COLOR if arm.holding else EMPTY_COLOR, 0.4)
            if arm.wrist is not None:
                cv2.circle(img, _pt(arm.wrist, scale), 6, PHASE_COLOR[detector.arms[side].phase], -1)
                vx, vy = detector.arms[side].velocity
                tip = (arm.wrist[0] + vx * 0.1, arm.wrist[1] + vy * 0.1)  # where the wrist will be in 0.1 s
                cv2.arrowedLine(img, _pt(arm.wrist, scale), _pt(tip, scale), (255, 0, 255), 2, cv2.LINE_AA, 0, 0.25)

    if last_throw and t - last_throw[0].t < THROW_MARKER_SECONDS:
        event, landing = last_throw
        cv2.line(img, _pt(event.position, scale), _pt(landing, scale), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.drawMarker(img, _pt(landing, scale), (0, 0, 255), cv2.MARKER_CROSS, 24, 2)

    # State panel.
    lines = [(f"fps {fps:4.1f}   need: speed {p.wrist_speed_threshold:.1f} sw/s, "
              f"held {p.hold_frames}f, empty {p.release_frames}f after swing", (255, 255, 255))]
    for side in SIDES:
        arm = detector.arms[side]
        seen = track.observation.arms.get(side) if track else None
        hand = ("--" if seen is None or seen.holding is None
                else f"holding {seen.object_label}" if seen.holding else "empty")
        lines.append((f"{side:5s} {arm.phase.value.upper():8s} {hand:22s} up {arm.up_speed:5.1f} sw/s",
                      PHASE_COLOR[arm.phase]))
    cooldown = detector.cooldown_remaining(t)
    if cooldown > 0:
        lines.append((f"COOLDOWN {cooldown:.1f}s", (0, 0, 255)))
    if detector.scale is None:
        lines.append(("no body found yet: get your shoulders in view", (0, 0, 255)))
    y = img.shape[0] - 12 - 20 * (len(lines) - 1)
    panel = img[max(0, y - 18):]
    panel[:] = panel // 3  # dim the background behind the text
    for line, color in lines:
        text(img, line, (10, y), color)
        y += 20
