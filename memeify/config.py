"""Loads config.yaml into typed dataclasses. Every key is optional; missing keys use the defaults below."""

from __future__ import annotations

import dataclasses
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .gesture import GestureParams

log = logging.getLogger("memeify")

# Next to memeify.exe when packaged with PyInstaller, else the repo folder.
PROJECT_DIR = (Path(sys.executable).resolve().parent if getattr(sys, "frozen", False)
               else Path(__file__).resolve().parent.parent)


class MemeifyError(Exception):
    """An error with a message meant for the user (no traceback needed)."""


class ConfigError(MemeifyError):
    pass


@dataclass
class CameraConfig:
    name: str | None = None       # input camera by name; None = the last one picked in the app, else the first real one
    width: int = 1280             # what to ask the input camera for
    height: int = 720
    fps: int = 30
    mirror: bool = False          # flip horizontally (meeting apps already mirror your self-view)
    backend: str = "auto"         # auto | dshow | msmf | v4l2 | avfoundation | any
    fourcc: str | None = "MJPG"   # most webcams only reach 720p30 over USB in MJPG


# Everything the object detector (EfficientDet, trained on COCO) can recognise.
COCO_LABELS = frozenset({
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
})

# The handheld everyday subset used by default.
HANDHELD_OBJECTS = [
    "broccoli", "carrot", "banana", "apple", "orange", "sandwich", "hot dog", "pizza", "donut", "cake",
    "bottle", "wine glass", "cup", "bowl", "fork", "spoon",
    "cell phone", "remote", "mouse", "book", "clock", "vase", "toothbrush", "hair drier",
    "sports ball", "frisbee", "teddy bear", "baseball bat", "baseball glove", "tennis racket",
]


@dataclass
class TrackingConfig:
    pose_model: str = "lite"      # lite | full
    process_width: int = 640      # frames are downscaled to this width for the pose model
    min_detection_confidence: float = 0.5
    min_visibility: float = 0.5   # pose landmarks below this visibility are ignored
    object_model: str = "lite0"   # lite0 (fast) | lite2 (more accurate, ~3x slower)
    object_score: float = 0.3     # detector confidence needed to count an object
    object_classes: list[str] = field(default_factory=lambda: list(HANDHELD_OBJECTS))
    touch_margin: float = 0.2     # shoulder widths around the hand that still count as touching it
    models_dir: str = "models"


@dataclass
class ExplosionConfig:
    scale: float = 1.0            # 1.0 = explosion is about 70% of the frame height
    lead_time: float = 0.25       # s to project the thrown object forward from release
    edge_margin: float = 0.2      # keep the explosion centre this far (fraction of short side) from edges
    behind_person: bool = True    # cut you out with a segmentation mask so the boom is behind you
    drop: float = 0.05            # place the explosion this much below the target (fraction of its height)
    asset_dir: str = "assets/explosion"
    fps: float | None = None      # PNG playback rate; None = fps.txt next to the PNGs, else 30


@dataclass
class SoundConfig:
    enabled: bool = True
    file: str | None = None       # None = first .wav/.ogg/.mp3 found in assets/
    volume: float = 0.8


@dataclass
class OutputConfig:
    width: int = 1280             # the memeify camera's picture size; input cameras are fitted to it
    height: int = 720
    fps: int = 30
    virtual_camera: str = "auto"  # auto | memeify | obs | off
    virtual_camera_device: str | None = None  # override the device name (Linux: e.g. /dev/video10)


@dataclass
class HotkeyConfig:
    pause: str = "p"                       # keys in the memeify window
    test_boom: str = "b"
    global_pause: str | None = "<ctrl>+<alt>+b"  # works from any app (pynput syntax); null to disable


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    gesture: GestureParams = field(default_factory=GestureParams)
    explosion: ExplosionConfig = field(default_factory=ExplosionConfig)
    sound: SoundConfig = field(default_factory=SoundConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    base_dir: Path = PROJECT_DIR  # relative paths in the config resolve against this

    def path(self, value: str) -> Path:
        p = Path(value).expanduser()
        return p if p.is_absolute() else self.base_dir / p


STATE_FILE = ".memeify-state.json"  # remembers choices made in the window (e.g. the camera)


def load_state(base_dir: Path) -> dict:
    try:
        state = json.loads((base_dir / STATE_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def save_state(base_dir: Path, **changes) -> None:
    state = load_state(base_dir) | changes
    try:
        (base_dir / STATE_FILE).write_text(json.dumps(state, indent=2), encoding="utf-8")
    except OSError as e:
        log.warning("couldn't save %s: %s", STATE_FILE, e)


def default_config_path() -> Path:
    local = Path.cwd() / "config.yaml"
    return local if local.exists() else PROJECT_DIR / "config.yaml"


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else default_config_path()
    raw: dict = {}
    if path.exists():
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"{path} is not valid YAML: {e}") from e
        if not isinstance(raw, dict):
            raise ConfigError(f"{path} should contain sections like 'camera:' and 'gesture:'")
    elif path != default_config_path():
        raise ConfigError(f"config file not found: {path}")

    cfg = Config(base_dir=path.resolve().parent)
    for section, value in raw.items():
        if not hasattr(cfg, section) or section == "base_dir":
            log.warning("config: ignoring unknown section '%s'", section)
            continue
        setattr(cfg, section, _build(type(getattr(cfg, section)), value or {}, section))

    try:
        cfg.gesture.validate()
    except ValueError as e:
        raise ConfigError(f"gesture: {e}") from e
    for name in cfg.tracking.object_classes:
        if name not in COCO_LABELS:
            log.warning("config: tracking.object_classes has '%s', which the detector doesn't know "
                        "(see the README for the 80 names it does)", name)
    return cfg


def _build(cls, values: dict, section: str):
    if not isinstance(values, dict):
        raise ConfigError(f"'{section}' should be a mapping of settings")
    known = {f.name for f in dataclasses.fields(cls)}
    for key in values.keys() - known:
        log.warning("config: ignoring unknown setting '%s.%s'", section, key)
    return cls(**{k: v for k, v in values.items() if k in known})
