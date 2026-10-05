"""The "memeify" camera: the virtual camera Zoom, Meet, Discord and OBS pick up.

Windows: Unity Capture's driver, registered under the name "memeify"
(`python -m memeify.install_camera`), with OBS's virtual camera as a fallback.
macOS: OBS's virtual camera (Apple only allows signed camera extensions).
Linux: a v4l2loopback device (load it with card_label=memeify).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

import numpy as np

from .config import ConfigError

log = logging.getLogger("memeify")

DEVICE_NAME = "memeify"
CHOICES = ("auto", "memeify", "obs", "off")


@dataclass
class OutputStatus:
    device: str | None   # the camera name apps see, or None when nothing is being output
    message: str         # one line for the status bar
    missing: bool = False  # the memeify camera isn't installed (offer to install it)


def backend_order(choice: str, platform: str = sys.platform) -> list[tuple[str, str | None]]:
    """(pyvirtualcam backend, device) pairs to try, best first."""
    if choice not in CHOICES:
        raise ConfigError(f"output.virtual_camera must be one of {', '.join(CHOICES)}, not '{choice}'")
    if choice == "off":
        return []
    if platform == "win32":
        memeify, obs = [("unitycapture", DEVICE_NAME)], [("obs", None)]
        return {"auto": memeify + obs, "memeify": memeify, "obs": obs}[choice]
    if platform == "darwin":
        return [("obs", None)] if choice in ("auto", "obs") else []
    return [("v4l2loopback", None)] if choice in ("auto", "memeify") else []


def setup_hint(platform: str = sys.platform) -> str:
    if platform == "win32":
        return "click 'Install memeify camera' (or run: python -m memeify.install_camera)"
    if platform == "darwin":
        return "install OBS Studio and start its virtual camera once"
    return "load v4l2loopback: sudo modprobe v4l2loopback devices=1 exclusive_caps=1 card_label=memeify"


class VirtualCamera:
    def __init__(self, cam, backend: str):
        self._cam = cam
        self.backend = backend
        self.device = cam.device

    def send(self, frame: np.ndarray) -> None:
        self._cam.send(frame)

    def close(self) -> None:
        self._cam.close()


def open_virtual_camera(width: int, height: int, fps: int, choice: str = "auto",
                        device: str | None = None, platform: str = sys.platform) -> tuple[VirtualCamera | None, OutputStatus]:
    """Start the best available virtual camera. Never raises for a missing driver: the app runs without one."""
    order = backend_order(choice, platform)
    if not order:
        return None, OutputStatus(None, "memeify camera: off (output.virtual_camera is 'off')")
    import pyvirtualcam

    errors = []
    for backend, default_device in order:
        try:
            cam = pyvirtualcam.Camera(width, height, fps, fmt=pyvirtualcam.PixelFormat.BGR, backend=backend,
                                      device=device or default_device, print_fps=False)
        except Exception as e:  # pyvirtualcam raises RuntimeError/ValueError subclasses per backend
            errors.append(f"{backend}: {e}")
            continue
        vcam = VirtualCamera(cam, backend)
        if backend == "unitycapture":
            return vcam, OutputStatus(vcam.device, "memeify camera: on")
        if backend == "v4l2loopback":
            return vcam, OutputStatus(vcam.device, f"memeify camera: on ({vcam.device})")
        # OBS's camera. On Windows that means the memeify one is missing and could be installed.
        note = " (install the memeify camera to stop needing OBS)" if platform == "win32" else ""
        return vcam, OutputStatus(vcam.device, f"output: {vcam.device}{note}", missing=platform == "win32")
    log.info("no virtual camera: %s", "; ".join(errors))
    return None, OutputStatus(None, f"memeify camera: not installed. To add it, {setup_hint(platform)}",
                              missing=platform == "win32")
