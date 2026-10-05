"""Throw state machine, driven by synthetic landmark sequences (see synthetic.py)."""

import math

import pytest

from synthetic import SHOULDERS, THROW, ArmScript, chain, observations, run
from memeify.gesture import GestureParams, Phase, ThrowDetector, landing_point

P = GestureParams()


def swing_and_events(detector, frames):
    """Run the frames; return (time the right arm entered SWUNG, events)."""
    swing_t, events = None, []
    for obs in frames:
        if e := detector.update(obs):
            events.append(e)
        if swing_t is None and detector.arms["right"].phase is Phase.SWUNG:
            swing_t = obs.t
    return swing_t, events


# --------------------------------------------------------------------------- the happy path

def test_throw_triggers_once_with_release_position_and_upward_velocity():
    events = run(ThrowDetector(), observations(right=THROW))
    assert len(events) == 1
    e = events[0]
    assert e.side == "right"
    assert e.label == "broccoli"
    assert 0.6 <= e.t <= 0.7                       # just after the swing, not before
    assert e.position[1] < SHOULDERS["right"][1]   # released above the shoulder
    assert e.velocity[1] < 0                       # moving up (image y points down)


def test_left_hand_throw_triggers_too():
    cup = THROW.mirrored()
    cup.label = "cup"
    events = run(ThrowDetector(), observations(left=cup))
    assert [(e.side, e.label) for e in events] == [("left", "cup")]


@pytest.mark.parametrize("fps", [15, 24, 30, 60])
def test_throw_detected_at_any_reasonable_frame_rate(fps):
    assert len(run(ThrowDetector(), observations(right=THROW, fps=fps))) == 1


def test_detector_walks_through_the_phases_in_order():
    detector = ThrowDetector()
    phases = []
    for obs in observations(right=THROW):
        detector.update(obs)
        phase = detector.arms["right"].phase
        if not phases or phases[-1] != phase:
            phases.append(phase)
    assert phases == [Phase.IDLE, Phase.HOLDING, Phase.SWUNG, Phase.IDLE]


def test_only_empty_frames_after_the_swing_count_as_the_release():
    # In THROW the detector already loses the blurred object during the wind-up.
    # Those frames must not count: the hand has to be seen empty *after* the swing.
    swing_t, events = swing_and_events(ThrowDetector(), observations(right=THROW))
    assert len(events) == 1
    assert events[0].t - swing_t == pytest.approx(P.release_frames / 30)


# --------------------------------------------------------------------------- things that must NOT trigger

def test_fast_swing_with_nothing_in_hand_does_not_trigger():
    wave = ArmScript(path=THROW.path, hold=[(0.0, "empty")])
    assert run(ThrowDetector(), observations(right=wave)) == []


def test_slow_raise_does_not_trigger():
    # Raise the object past the shoulder over 1.5 s (under 1 shoulder-width/s), then let go
    # right after crossing the line: the same order as a throw, just far too slow.
    slow = ArmScript(path=[(0.0, (560.0, 460.0)), (1.5, (600.0, 180.0)), (2.0, (600.0, 180.0))],
                     hold=[(0.0, "held"), (1.0, "empty")])
    assert run(ThrowDetector(), observations(right=slow)) == []


def test_object_still_in_hand_after_the_swing_does_not_trigger():
    # Fake throw: the object blurs out mid-swing, then shows up in the hand again.
    fake = ArmScript(path=THROW.path, hold=[(0.0, "held"), (0.45, "empty"), (0.6, "held")])
    assert run(ThrowDetector(), observations(right=fake)) == []


def test_detector_flicker_after_the_swing_is_not_a_release():
    # The object keeps dropping out for two frames at a time, but is never gone for three.
    flicker = [(0.45 + i / 30, "held" if i % 3 == 2 else "empty") for i in range(20)]
    fake = ArmScript(path=THROW.path, hold=[(0.0, "held")] + flicker + [(1.12, "held")])
    assert run(ThrowDetector(GestureParams(release_frames=3)), observations(right=fake)) == []


def test_hand_out_of_view_after_the_swing_is_not_a_release():
    # Unknown (hand not visible) isn't the same as empty.
    gone = ArmScript(path=THROW.path, hold=[(0.0, "held"), (0.45, None)])
    assert run(ThrowDetector(), observations(right=gone)) == []


def test_release_too_long_after_the_swing_does_not_trigger():
    late = ArmScript(path=THROW.path + [(2.0, (610.0, 160.0))], hold=[(0.0, "held"), (1.4, "empty")])
    assert run(ThrowDetector(), observations(right=late)) == []


def test_putting_the_object_down_does_not_trigger():
    drop = ArmScript(path=[(0.0, (560.0, 460.0)), (1.5, (560.0, 460.0))],
                     hold=[(0.0, "held"), (0.5, "empty")])
    assert run(ThrowDetector(), observations(right=drop)) == []


def test_fast_motion_that_stays_below_the_shoulder_does_not_trigger():
    punch = ArmScript(path=[(0.0, (560.0, 600.0)), (0.4, (560.0, 600.0)), (0.55, (560.0, 330.0)),
                            (1.0, (560.0, 330.0))],
                      hold=[(0.0, "held"), (0.45, "empty")])
    assert run(ThrowDetector(), observations(right=punch)) == []


def test_object_already_above_the_shoulder_does_not_count_as_crossing():
    flick = ArmScript(path=[(0.0, (560.0, 280.0)), (0.4, (560.0, 280.0)), (0.5, (570.0, 120.0)),
                            (1.0, (570.0, 120.0))],
                      hold=[(0.0, "held"), (0.45, "empty")])
    assert run(ThrowDetector(), observations(right=flick)) == []


def test_single_frame_detection_glitch_does_not_arm():
    glitch = ArmScript(path=THROW.path, hold=[(0.0, "empty"), (0.4, "held"), (0.41, "empty")])
    assert run(ThrowDetector(GestureParams(hold_frames=2)), observations(right=glitch)) == []


def test_hold_forgotten_if_object_unseen_for_too_long():
    stale = ArmScript(path=[(0.0, (560.0, 460.0)), (1.2, (560.0, 460.0)), (1.4, (600.0, 180.0)),
                            (1.8, (600.0, 180.0))],
                      hold=[(0.0, "held"), (0.3, "empty")])
    assert run(ThrowDetector(GestureParams(hold_memory=0.6)), observations(right=stale)) == []


def test_tracking_dropout_does_not_fake_a_fast_swing():
    # The pose model loses the wrist at chest height and finds it again above the shoulder
    # half a second later. That jump isn't a measured speed.
    jump = ArmScript(path=[(0.0, (560.0, 460.0)), (0.3, None), (0.8, (600.0, 180.0)), (1.2, (600.0, 180.0))],
                     hold=[(0.0, "held"), (0.3, None), (0.8, "empty")])
    assert run(ThrowDetector(), observations(right=jump)) == []


def test_no_body_means_no_trigger():
    assert run(ThrowDetector(), observations(right=THROW, shoulder_width=None)) == []


# --------------------------------------------------------------------------- cooldown, both hands, reset

def test_cooldown_blocks_a_second_throw_then_allows_a_third():
    p = GestureParams(cooldown=3.0)
    arm = chain(THROW, THROW.shifted(1.5), THROW.shifted(4.5))
    events = run(ThrowDetector(p), observations(right=arm))
    assert len(events) == 2
    assert events[1].t - events[0].t >= p.cooldown


def test_both_hands_throwing_at_once_fire_only_once():
    events = run(ThrowDetector(), observations(right=THROW, left=THROW.mirrored()))
    assert len(events) == 1


def test_either_hand_can_follow_the_other_after_the_cooldown():
    p = GestureParams(cooldown=1.0)
    events = run(ThrowDetector(p), observations(right=THROW, left=THROW.mirrored().shifted(2.0)))
    assert [e.side for e in events] == ["right", "left"]


def test_reset_mid_throw_cancels_it():
    detector = ThrowDetector()
    events = []
    for obs in observations(right=THROW):
        if detector.arms["right"].phase is Phase.SWUNG:
            detector.reset()  # e.g. the pause hotkey
        if e := detector.update(obs):
            events.append(e)
    assert events == []


def test_cooldown_remaining_counts_down():
    detector = ThrowDetector(GestureParams(cooldown=3.0))
    event = run(detector, observations(right=THROW))[0]
    assert detector.cooldown_remaining(event.t) == pytest.approx(3.0)
    assert detector.cooldown_remaining(event.t + 1.0) == pytest.approx(2.0)
    assert detector.cooldown_remaining(event.t + 5.0) == 0.0


# --------------------------------------------------------------------------- release velocity and landing

def test_release_velocity_falls_back_to_the_swing_peak_when_the_hand_brakes():
    # Hand stops dead above the shoulder and only lets go 0.15 s later.
    brake = ArmScript(path=[(0.0, (560.0, 460.0)), (0.4, (560.0, 460.0)), (0.55, (580.0, 200.0)),
                            (1.0, (580.0, 200.0))],
                      hold=[(0.0, "held"), (0.7, "empty")])
    events = run(ThrowDetector(), observations(right=brake))
    assert len(events) == 1
    vx, vy = events[0].velocity
    assert vy < -1000  # the swing's ~1700 px/s upward speed, not the ~0 at the moment of release
    assert vx > 0


def test_landing_point_projects_along_the_velocity():
    assert landing_point((640, 360), (100, -100), 0.5, (1280, 720)) == (690, 310)


def test_landing_point_is_clamped_inside_the_frame():
    x, y = landing_point((100, 100), (1000, -2000), 0.25, (1280, 720))
    assert (x, y) == (350, 0)
    x, y = landing_point((100, 100), (-5000, 5000), 0.25, (1280, 720), margin=0.1)
    assert (x, y) == (72, 720 - 72)


def test_landing_from_a_real_throw_lands_up_and_inside():
    event = run(ThrowDetector(), observations(right=THROW))[0]
    x, y = landing_point(event.position, event.velocity, 0.25, (1280, 720), margin=0.12)
    assert y < event.position[1]
    assert 0 <= x <= 1280 and 0 <= y <= 720


# --------------------------------------------------------------------------- params

@pytest.mark.parametrize("bad", [
    dict(hold_frames=0),
    dict(release_frames=0),
    dict(wrist_speed_threshold=0),
    dict(hold_memory=0),
    dict(cooldown=-1),
])
def test_invalid_params_are_rejected(bad):
    with pytest.raises(ValueError):
        ThrowDetector(GestureParams(**bad))


def test_speed_is_measured_in_shoulder_widths():
    # Same throw filmed from twice as far away: everything is half the size in pixels.
    far = ArmScript(path=[(t, p and (p[0] / 2, p[1] / 2)) for t, p in THROW.path], hold=THROW.hold)
    detector = ThrowDetector()
    events = []
    for obs in observations(right=far, shoulder_width=100.0):
        for arm in obs.arms.values():
            arm.shoulder = arm.shoulder and (arm.shoulder[0] / 2, arm.shoulder[1] / 2)
        if e := detector.update(obs):
            events.append(e)
    assert len(events) == 1
    assert math.isclose(events[0].position[1] * 2, run(ThrowDetector(), observations(right=THROW))[0].position[1])
