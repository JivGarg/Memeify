"""Synthetic landmark sequences for driving the throw detector without a camera."""

from __future__ import annotations

from dataclasses import dataclass, field

from memeify.gesture import ArmObservation, Observation, Point

FRAME_W = 1280
SHOULDER_WIDTH = 200.0
# Unmirrored webcam image: the subject's right shoulder appears on the image's left.
SHOULDERS = {"right": (540.0, 300.0), "left": (740.0, 300.0)}
RESTING_WRIST = {"right": (520.0, 620.0), "left": (760.0, 620.0)}

# What the tracker reports about the hand each frame.
HOLD = {
    "held": True,    # the detector found an object touching the hand
    "empty": False,  # hand visible, nothing touching it (thrown, or blurred past recognition)
    None: None,      # couldn't check (hand out of view)
}


def mirror(p: Point) -> Point:
    """Reflect a right-arm position to the corresponding left-arm position."""
    return (FRAME_W - p[0], p[1])


@dataclass
class ArmScript:
    """Keyframed motion of one arm. Wrist positions are interpolated linearly; the hand
    state ("held" / "empty" / None) holds from its keyframe until the next one.
    A wrist keyframe of None means the pose model lost the wrist until the next keyframe."""
    path: list[tuple[float, Point | None]]
    hold: list[tuple[float, str | None]] = field(default_factory=lambda: [(0.0, "empty")])
    label: str = "broccoli"

    def wrist_at(self, t: float) -> Point | None:
        if t <= self.path[0][0]:
            return self.path[0][1]
        for (t0, p0), (t1, p1) in zip(self.path, self.path[1:]):
            if t0 <= t < t1:
                if p0 is None or p1 is None:
                    return p0  # lost until the next keyframe, or holding still before being lost
                k = (t - t0) / (t1 - t0)
                return (p0[0] + (p1[0] - p0[0]) * k, p0[1] + (p1[1] - p0[1]) * k)
        return self.path[-1][1]

    def holding_at(self, t: float) -> bool | None:
        state = None
        for tk, s in self.hold:
            if t + 1e-9 >= tk:
                state = s
        return HOLD[state]

    def shifted(self, dt: float) -> ArmScript:
        return ArmScript([(t + dt, p) for t, p in self.path], [(t + dt, s) for t, s in self.hold], self.label)

    def mirrored(self) -> ArmScript:
        return ArmScript([(t, p and mirror(p)) for t, p in self.path], list(self.hold), self.label)

    @property
    def end(self) -> float:
        return max(self.path[-1][0], self.hold[-1][0])


def resting(side: str, until: float) -> ArmScript:
    return ArmScript([(0.0, RESTING_WRIST[side]), (until, RESTING_WRIST[side])])


def chain(*scripts: ArmScript) -> ArmScript:
    """Play several scripts of the same arm back to back (each already shifted to its start time)."""
    return ArmScript([kf for s in scripts for kf in s.path], [kf for s in scripts for kf in s.hold])


# The canonical throw with the right arm: broccoli held at chest height, whipped up past the
# shoulder (the detector loses the blurred object mid-swing), and gone from the hand afterwards.
THROW = ArmScript(
    path=[(0.0, (560.0, 460.0)), (0.4, (560.0, 460.0)), (0.6, (600.0, 180.0)),
          (0.7, (610.0, 160.0)), (1.2, (610.0, 160.0))],
    hold=[(0.0, "held"), (0.45, "empty")],
)


def observations(right: ArmScript | None = None, left: ArmScript | None = None, fps: float = 30.0,
                 shoulder_width: float | None = SHOULDER_WIDTH, end: float | None = None):
    """Yield one Observation per frame for the given arm scripts."""
    scripts = {"right": right, "left": left}
    end = end if end is not None else max(s.end for s in scripts.values() if s)
    for i in range(int(round(end * fps)) + 1):
        t = i / fps
        arms = {}
        for side, script in scripts.items():
            script = script or resting(side, end)
            holding = script.holding_at(t)
            arms[side] = ArmObservation(wrist=script.wrist_at(t), shoulder=SHOULDERS[side], holding=holding,
                                        object_label=script.label if holding else None)
        yield Observation(t, arms, shoulder_width)


def run(detector, frames) -> list:
    return [e for e in (detector.update(o) for o in frames) if e]
