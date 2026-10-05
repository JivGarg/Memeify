"""Config loading, hand/object geometry, and the real-time-synced recorder."""

import logging

import cv2
import numpy as np
import pytest

from memeify.config import PROJECT_DIR, Config, ConfigError, load_config
from memeify.gesture import GestureParams
from memeify.output import Recorder
from memeify.tracking import DetectedObject, detection_crop, hand_box, objects_in_hands, overlap


# --------------------------------------------------------------------------- config

def test_shipped_config_yaml_matches_the_code_defaults():
    # config.yaml documents every setting; it should agree with the dataclass defaults.
    assert load_config(PROJECT_DIR / "config.yaml") == Config(base_dir=PROJECT_DIR)


def test_partial_config_keeps_defaults(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("gesture:\n  cooldown: 5\ncamera:\n  name: Iriun Webcam\n")
    cfg = load_config(path)
    assert cfg.gesture.cooldown == 5
    assert cfg.gesture.hold_frames == GestureParams().hold_frames
    assert cfg.camera.name == "Iriun Webcam" and cfg.camera.width == 1280
    assert cfg.path("assets") == tmp_path / "assets"  # relative paths resolve next to the config


def test_unknown_keys_warn_but_load(tmp_path, caplog):
    path = tmp_path / "c.yaml"
    path.write_text("gesture:\n  coldown: 5\nvideo:\n  x: 1\n")
    with caplog.at_level(logging.WARNING, logger="memeify"):
        cfg = load_config(path)
    assert cfg.gesture.cooldown == GestureParams().cooldown
    assert "gesture.coldown" in caplog.text and "'video'" in caplog.text


@pytest.mark.parametrize("text", [
    "gesture:\n  release_frames: 0\n",  # must be at least 1
    "gesture: [1, 2]\n",
    "just a string\n",
    "camera: {index: 0\n",  # broken YAML
])
def test_bad_config_is_a_clear_error(tmp_path, text):
    path = tmp_path / "c.yaml"
    path.write_text(text)
    with pytest.raises(ConfigError):
        load_config(path)


def test_missing_explicit_config_file_is_an_error(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")


def test_misspelled_object_class_warns(tmp_path, caplog):
    path = tmp_path / "c.yaml"
    path.write_text("tracking:\n  object_classes: [brocoli, cup]\n")
    with caplog.at_level(logging.WARNING, logger="memeify"):
        load_config(path)
    assert "'brocoli'" in caplog.text and "'cup'" not in caplog.text


# --------------------------------------------------------------------------- hand / object geometry

def test_hand_box_grows_by_the_margin():
    # Wrist + knuckles span 40x20 px, shoulder width 200 px: half-size = max(20, 0.15 * 200) + 0.2 * 200.
    box = hand_box([(100, 100), (120, 80), (140, 100)], scale=200, margin=0.2)
    assert box == (120 - 70, 90 - 70, 120 + 70, 90 + 70)


def test_overlap():
    assert overlap((0, 0, 10, 10), (5, 5, 20, 20)) == 25
    assert overlap((0, 0, 10, 10), (10, 0, 20, 10)) == 0  # edges touching isn't overlap
    assert overlap((0, 0, 10, 10), (50, 50, 60, 60)) == 0


def test_detection_crop_is_square_padded_and_inside_the_frame():
    x0, y0, x1, y1 = detection_crop([(100, 100, 200, 200), (300, 150, 350, 250)], 50, (1280, 720), min_side=100)
    assert x1 - x0 == y1 - y0 == 350  # widest padded span (50..400) wins
    assert x0 <= 50 and x1 >= 400 and y0 <= 50 and y1 >= 300
    assert detection_crop([(0, 650, 40, 700)], 50, (1280, 720), min_side=300) == (0, 420, 300, 720)
    assert detection_crop([], 50, (1280, 720), 100) is None


def test_objects_in_hands_picks_the_most_overlapping_object():
    broccoli = DetectedObject("broccoli", 0.8, (90, 90, 140, 140))
    cup = DetectedObject("cup", 0.9, (130, 130, 300, 300))
    hands = {"right": (100, 100, 150, 150), "left": (500, 100, 550, 150)}
    assert objects_in_hands(hands, [broccoli, cup]) == {"right": broccoli, "left": None}


def test_one_object_can_be_in_both_hands():
    ball = DetectedObject("sports ball", 0.7, (140, 100, 210, 150))
    hands = {"right": (100, 100, 150, 150), "left": (200, 100, 250, 150)}
    assert objects_in_hands(hands, [ball]) == {"right": ball, "left": ball}


# --------------------------------------------------------------------------- recorder

def test_recorder_repeats_frames_to_stay_real_time(tmp_path):
    path = tmp_path / "out.mp4"
    rec = Recorder(path, 64, 48, fps=30)
    frame = np.zeros((48, 64, 3), np.uint8)
    for t in (0.0, 0.1, 0.2):  # only 10 fps arriving
        rec.write(frame, t)
    rec.close()
    assert rec.frames_written == 7  # 0.2 s at 30 fps, plus the first frame
    cap = cv2.VideoCapture(str(path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 7
    cap.release()
