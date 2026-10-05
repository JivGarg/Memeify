"""Frame sources (a camera, or a video file for replaying test clips) and fitting frames to the output size."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from .cameras import CameraInfo
from .config import CameraConfig, MemeifyError


class CameraError(MemeifyError):
    pass


def fit_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Scale and centre-crop `frame` to exactly `size` (w, h), filling it with no black bars.

    The memeify camera has a fixed size, but input cameras differ (4:3, 16:9, phones...).
    """
    w, h = size
    fh, fw = frame.shape[:2]
    if (fw, fh) == (w, h):
        return frame
    scale = max(w / fw, h / fh)
    rw, rh = max(w, round(fw * scale)), max(h, round(fh * scale))
    resized = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    x0, y0 = (rw - w) // 2, (rh - h) // 2
    return resized[y0:y0 + h, x0:x0 + w]


class Camera:
    """Reads frames from a camera, or from a video file (looped, paced to real time)."""

    def __init__(self, cfg: CameraConfig, source: CameraInfo | Path):
        self.mirror = cfg.mirror
        self.is_file = isinstance(source, Path)
        self.name = source.name

        if self.is_file:
            if not source.exists():
                raise CameraError(f"input file not found: {source}")
            self._cap = cv2.VideoCapture(str(source))
            if not self._cap.isOpened():
                raise CameraError(f"could not open video file {source}")
        else:
            self._cap = cv2.VideoCapture(source.index, source.api)
            if not self._cap.isOpened():
                raise CameraError(
                    f"could not open '{source.name}'. Is another app (Zoom, Teams, OBS, a browser tab) "
                    "using it? Only one program can hold a real camera at a time.")
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always process the newest frame, don't build up lag
            if cfg.fourcc:
                self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cfg.fourcc))
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
            self._cap.set(cv2.CAP_PROP_FPS, cfg.fps)

        ok, first = self._cap.read()
        if not ok or first is None:
            self._cap.release()
            raise CameraError(f"'{self.name}' opened but returned no picture. Is it in use by another app?")
        self._pending: np.ndarray | None = first
        reported = self._cap.get(cv2.CAP_PROP_FPS)
        self.fps = reported if 1 <= reported <= 240 else float(cfg.fps)
        self._next_due = time.monotonic()

    def read(self) -> np.ndarray | None:
        if self._pending is not None:
            frame, self._pending = self._pending, None
        else:
            ok, frame = self._cap.read()
            if (not ok or frame is None) and self.is_file:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop the clip
                ok, frame = self._cap.read()
            if not ok or frame is None:
                return None
        if self.is_file:  # play files at their own speed, like a camera would deliver them
            self._next_due += 1 / self.fps
            delay = self._next_due - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                self._next_due = time.monotonic()  # running behind: don't try to catch up
        return cv2.flip(frame, 1) if self.mirror else frame

    def close(self) -> None:
        self._cap.release()
