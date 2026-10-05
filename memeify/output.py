"""Saving the output to an MP4 file."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .config import MemeifyError


class OutputError(MemeifyError):
    pass


class Recorder:
    """Writes an MP4 that plays back in real time.

    If processing runs slower than the target fps, frames are repeated so the
    recording stays in sync with the timestamps instead of playing back sped up.
    """

    def __init__(self, path: str | Path, width: int, height: int, fps: int):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.fps = fps
        self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not self._writer.isOpened():
            raise OutputError(f"couldn't open {path} for recording")
        self._t0: float | None = None
        self.frames_written = 0

    def write(self, frame: np.ndarray, t: float) -> None:
        if self._t0 is None:
            self._t0 = t
        target = round((t - self._t0) * self.fps) + 1  # frames that should exist by time t
        for _ in range(target - self.frames_written):
            self._writer.write(frame)
            self.frames_written += 1

    def close(self) -> None:
        self._writer.release()
