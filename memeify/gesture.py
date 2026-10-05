"""Throw detection: object in hand -> fast upward swing past the shoulder -> object leaves the hand.

This module is plain Python on purpose (no OpenCV / MediaPipe), so the tests can
drive it with synthetic landmark sequences. Coordinates are pixels in the video
frame (x right, y DOWN), times are seconds.

The tracker reports, per arm and per frame, whether an object is touching the hand:
True (object on the hand), False (hand visible, nothing on it) or None (can't tell).
One state machine runs per arm; the cooldown is shared so a single throw can't
fire twice, even if both hands happen to match.

        object seen on the hand for N frames
    IDLE ----------------------------------> HOLDING
     ^  <---- object not seen for a while -----'  |
     |                                            | wrist crosses above the shoulder line
     |                                            | while moving up faster than threshold
     |  <---- hand never seen empty in time --- SWUNG
     |                                            | hand seen empty for M frames
     '------- (shared cooldown) <------------- TRIGGER -> ThrowEvent
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

Point = tuple[float, float]
SIDES = ("left", "right")


def landing_point(position: Point, velocity: Point, lead_time: float,
                  frame_size: tuple[int, int], margin: float = 0.0) -> Point:
    """Where the thrown object would be `lead_time` seconds after release.

    Straight-line projection along the release velocity, clamped inside the
    frame. `margin` is a fraction of the frame's shorter side kept clear of the edges.
    """
    w, h = frame_size
    m = margin * min(w, h)
    x = position[0] + velocity[0] * lead_time
    y = position[1] + velocity[1] * lead_time
    return (min(max(x, m), w - m), min(max(y, m), h - m))


@dataclass
class GestureParams:
    wrist_speed_threshold: float = 2.5  # upward wrist speed, shoulder-widths per second
    hold_frames: int = 2                # frames the object must be seen on the hand before it counts
    hold_memory: float = 0.6            # s the object is remembered when the detector loses it (motion blur)
    release_frames: int = 3             # frames the hand must be seen empty after the swing
    release_window: float = 0.7         # s after the swing within which the hand must be seen empty
    cooldown: float = 3.0               # s after a trigger during which nothing fires
    velocity_window: float = 0.1        # s of wrist history used to estimate velocity

    def validate(self) -> None:
        if self.hold_frames < 1 or self.release_frames < 1:
            raise ValueError("hold_frames and release_frames must be at least 1")
        for name in ("wrist_speed_threshold", "hold_memory", "release_window", "velocity_window"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.cooldown < 0:
            raise ValueError("cooldown can't be negative")


@dataclass
class ArmObservation:
    """What the trackers saw of one arm in one frame. None = not visible / unknown."""
    wrist: Point | None = None
    shoulder: Point | None = None
    elbow: Point | None = None          # only used for debug drawing
    holding: bool | None = None         # object touching the hand? None = couldn't check
    object_label: str | None = None     # what's on the hand, when holding


@dataclass
class Observation:
    t: float
    arms: dict[str, ArmObservation] = field(default_factory=dict)
    shoulder_width: float | None = None  # px; the body-size yardstick for speed


@dataclass(frozen=True)
class ThrowEvent:
    t: float
    side: str
    position: Point            # wrist at release, px
    velocity: Point            # px/s, used to project the landing spot
    label: str | None = None   # what was thrown, if the detector named it


class Phase(Enum):
    IDLE = "idle"
    HOLDING = "holding"
    SWUNG = "swung"


_NOT_SEEN = ArmObservation()


class ArmStateMachine:
    """Hold -> swing -> release for a single arm."""

    def __init__(self, side: str, params: GestureParams):
        self.side = side
        self.p = params
        self._history: deque[tuple[float, float, float]] = deque()
        self.velocity: Point = (0.0, 0.0)
        self.up_speed = 0.0              # shoulder-widths/s, positive = moving up
        self.holding: bool | None = None
        self.reset()

    def reset(self) -> None:
        """Forget everything, including wrist history (used on pause)."""
        self._history.clear()
        self.to_idle()

    def to_idle(self) -> None:
        self.phase = Phase.IDLE
        self._held_count = 0      # consecutive frames with the object on the hand
        self._empty_count = 0     # consecutive frames with the hand seen empty
        self._last_hold_t = -math.inf
        self._label: str | None = None
        self._was_below = False
        self._swing_t = -math.inf
        self._peak_velocity: Point = (0.0, 0.0)
        self._peak_up = 0.0
        self._last_wrist: Point | None = None

    def measure(self, t: float, arm: ArmObservation, scale: float) -> None:
        """Update wrist velocity and the held/empty counters. Runs every frame, even during cooldown."""
        if arm.wrist is not None:
            if self._history and t - self._history[-1][0] > 3 * self.p.velocity_window:
                self._history.clear()  # stale samples across a gap would fake a huge speed
            self._history.append((t, *arm.wrist))
            # Keep one sample at or before the window start so the span covers the full window.
            while len(self._history) > 2 and self._history[1][0] <= t - self.p.velocity_window:
                self._history.popleft()
            self.velocity = self._estimate_velocity()
        else:
            self.velocity = (0.0, 0.0)
        self.up_speed = -self.velocity[1] / scale
        self.holding = arm.holding
        # Unknown frames (None) neither count nor break a streak: the detector just didn't get a look.
        if arm.holding is True:
            self._held_count += 1
            self._empty_count = 0
            self._label = arm.object_label or self._label
        elif arm.holding is False:
            self._empty_count += 1
            self._held_count = 0

    def step(self, t: float, arm: ArmObservation) -> ThrowEvent | None:
        """Advance the state machine. Call after measure()."""
        p = self.p

        if self.phase is Phase.IDLE:
            if self._held_count >= p.hold_frames:
                self.phase = Phase.HOLDING
                self._last_hold_t = t
                self._was_below = self._below_shoulder(arm)
            return None

        if self.phase is Phase.HOLDING:
            # A fast swing blurs the object, so it's normal for the detector to lose it
            # mid-throw; the hold is only dropped once it's been gone for hold_memory.
            if arm.holding is True:
                self._last_hold_t = t
            if t - self._last_hold_t > p.hold_memory:
                self.to_idle()  # put down, or out of sight for too long
                return None
            if self._below_shoulder(arm):
                self._was_below = True
            elif (self._was_below and self._above_shoulder(arm)
                  and self.up_speed >= p.wrist_speed_threshold):
                self.phase = Phase.SWUNG
                self._swing_t = t
                self._peak_velocity, self._peak_up = self.velocity, self.up_speed
                self._empty_count = 0  # only empty frames after the swing count as the release
                self._last_wrist = arm.wrist
            return None

        # SWUNG: waiting to see the hand without the object.
        if arm.wrist is not None:
            self._last_wrist = arm.wrist
        if self.up_speed > self._peak_up:
            self._peak_velocity, self._peak_up = self.velocity, self.up_speed
        if self._empty_count >= p.release_frames:
            event = ThrowEvent(t, self.side, self._last_wrist, self._release_velocity(), self._label)
            self.to_idle()
            return event
        if t - self._swing_t > p.release_window:
            self.to_idle()
        return None

    def _estimate_velocity(self) -> Point:
        if len(self._history) < 2:
            return (0.0, 0.0)
        t0, x0, y0 = self._history[0]
        t1, x1, y1 = self._history[-1]
        dt = t1 - t0
        if dt <= 0:
            return (0.0, 0.0)
        return ((x1 - x0) / dt, (y1 - y0) / dt)

    def _release_velocity(self) -> Point:
        # The hand usually brakes as it lets go; if it slowed a lot, the swing's
        # peak velocity says more about where the object was going.
        if math.hypot(*self.velocity) < 0.5 * math.hypot(*self._peak_velocity):
            return self._peak_velocity
        return self.velocity

    @staticmethod
    def _below_shoulder(arm: ArmObservation) -> bool:
        return arm.wrist is not None and arm.shoulder is not None and arm.wrist[1] > arm.shoulder[1]

    @staticmethod
    def _above_shoulder(arm: ArmObservation) -> bool:
        return arm.wrist is not None and arm.shoulder is not None and arm.wrist[1] < arm.shoulder[1]


class ThrowDetector:
    """Runs one ArmStateMachine per arm with a shared cooldown."""

    def __init__(self, params: GestureParams | None = None):
        self.params = params or GestureParams()
        self.params.validate()
        self.arms = {side: ArmStateMachine(side, self.params) for side in SIDES}
        self.cooldown_until = -math.inf
        self.scale: float | None = None  # smoothed shoulder width, px

    def reset(self) -> None:
        for arm in self.arms.values():
            arm.reset()
        self.cooldown_until = -math.inf

    def cooldown_remaining(self, t: float) -> float:
        return max(0.0, self.cooldown_until - t)

    def update(self, obs: Observation) -> ThrowEvent | None:
        if obs.shoulder_width:
            self.scale = (obs.shoulder_width if self.scale is None
                          else 0.8 * self.scale + 0.2 * obs.shoulder_width)
        if self.scale is None:
            return None  # no body seen yet, so no yardstick for "fast"

        for side, arm in self.arms.items():
            arm.measure(obs.t, obs.arms.get(side, _NOT_SEEN), self.scale)
        if obs.t < self.cooldown_until:
            return None

        for side, arm in self.arms.items():
            event = arm.step(obs.t, obs.arms.get(side, _NOT_SEEN))
            if event:
                self.cooldown_until = obs.t + self.params.cooldown
                for other in self.arms.values():
                    other.to_idle()
                return event
        return None
