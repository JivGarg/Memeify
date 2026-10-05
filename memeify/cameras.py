"""Find the cameras on this computer by name, without opening them (so no camera light comes on)."""

from __future__ import annotations

import sys
from collections.abc import Iterable
from dataclasses import dataclass

import cv2

from .config import ConfigError

# memeify's own output must never be offered as an input, or the picture would feed back into itself.
OWN_OUTPUT_NAMES = ("memeify", "unity video capture")

_APIS = {
    "any": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
    "avfoundation": cv2.CAP_AVFOUNDATION,
}


@dataclass(frozen=True)
class CameraInfo:
    name: str
    index: int           # what cv2.VideoCapture(index, api) expects
    api: int
    physical: bool = False  # a USB device (has vendor/product ids), not a virtual camera


def capture_api(backend: str, platform: str = sys.platform) -> int:
    """OpenCV capture API for a camera.backend setting."""
    if backend == "auto":
        # DirectShow sees virtual cameras too, opens fast and honours resolution requests.
        return {"win32": cv2.CAP_DSHOW, "darwin": cv2.CAP_AVFOUNDATION}.get(platform, cv2.CAP_V4L2)
    if backend not in _APIS:
        raise ConfigError(f"unknown camera.backend '{backend}' (use one of: auto, {', '.join(_APIS)})")
    return _APIS[backend]


def without_own_output(cameras: Iterable[CameraInfo], exclude: Iterable[str] = ()) -> list[CameraInfo]:
    """Drop memeify's own camera (and any extra names, e.g. the OBS camera when that's the output)."""
    extra = {name.lower() for name in exclude if name}
    return [c for c in cameras
            if not any(own in c.name.lower() for own in OWN_OUTPUT_NAMES) and c.name.lower() not in extra]


def list_cameras(api: int, exclude: Iterable[str] = ()) -> list[CameraInfo]:
    """Cameras usable as memeify's input, in the system's order."""
    from cv2_enumerate_cameras import enumerate_cameras

    found = [CameraInfo(c.name, c.index, c.backend or api, c.vid is not None) for c in enumerate_cameras(api)]
    return without_own_output(found, exclude)


def labels(cameras: list[CameraInfo]) -> list[str]:
    """Names for a dropdown, numbered when two cameras share a name."""
    seen: dict[str, int] = {}
    out = []
    for c in cameras:
        seen[c.name] = seen.get(c.name, 0) + 1
        out.append(c.name if seen[c.name] == 1 else f"{c.name} ({seen[c.name]})")
    return out


def pick_default(cameras: list[CameraInfo], preferred: str | None = None) -> CameraInfo | None:
    """The preferred camera if it's plugged in, else the first physical one, else the first one."""
    if preferred:
        for c in cameras:
            if c.name.lower() == preferred.lower():
                return c
    return next((c for c in cameras if c.physical), cameras[0] if cameras else None)
