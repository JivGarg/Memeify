"""The processing loop: camera -> tracking -> throw detection -> explosion -> memeify camera / recording.

Runs on a background thread so the window stays responsive. Everything heavy (MediaPipe,
the virtual camera, the recorder) is created and used only on that thread. The window
talks to it through a command queue and reads back the latest preview and status.
"""

from __future__ import annotations

import contextlib
import dataclasses
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

# Quieten MediaPipe's native logging before it's imported.
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from .cameras import CameraInfo  # noqa: E402
from .capture import Camera, CameraError, fit_frame  # noqa: E402
from .config import Config, MemeifyError  # noqa: E402
from .effects import EffectRenderer, ExplosionSprite, SoundPlayer, find_sound  # noqa: E402
from .gesture import ThrowDetector, landing_point  # noqa: E402
from .hotkey import GlobalHotkey  # noqa: E402
from .output import Recorder  # noqa: E402
from .overlay import draw_debug, draw_paused  # noqa: E402
from .virtualcam import OutputStatus, open_virtual_camera  # noqa: E402

log = logging.getLogger("memeify")

PREVIEW_WIDTH = 640
MAX_READ_FAILURES = 50  # consecutive empty reads before we decide the camera is gone
EFFECT_NAME = "Throw → Boom"

Source = CameraInfo | Path


@dataclass
class EngineStatus:
    state: str = "starting"         # starting | running | error | stopped
    message: str = "Starting..."    # what the engine is doing, or the fatal error
    fps: float = 0.0
    source: str | None = None       # input camera (or file) name
    camera_error: str | None = None
    output: OutputStatus = field(default_factory=lambda: OutputStatus(None, "memeify camera: starting..."))
    recording: Path | None = None
    recording_since: float | None = None  # time.monotonic() when it started
    last_recording: Path | None = None
    paused: bool = False
    last_boom: str | None = None


def _default_tracker(cfg: Config):
    from .tracking import Tracker  # imports MediaPipe; kept lazy so tests can swap in a fake
    return Tracker(cfg.tracking, cfg.path(cfg.tracking.models_dir), segmentation=cfg.explosion.behind_person)


def idle_frame(size: tuple[int, int], text: str) -> np.ndarray:
    """What the memeify camera shows while there's no input camera."""
    w, h = size
    frame = np.full((h, w, 3), 24, np.uint8)
    scale = h / 720
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale, 2)
    cv2.putText(frame, text, ((w - tw) // 2, (h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2 * scale,
                (200, 200, 200), 2, cv2.LINE_AA)
    return frame


class Engine:
    def __init__(self, cfg: Config, source: Source | None = None, debug: bool = False, *,
                 camera_factory=Camera, tracker_factory=_default_tracker, output_opener=open_virtual_camera,
                 clock=time.monotonic):
        self.cfg = cfg
        self._initial_source = source
        self._debug = debug
        self._camera_factory = camera_factory
        self._tracker_factory = tracker_factory
        self._output_opener = output_opener
        self._clock = clock
        self._commands: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._preview: np.ndarray | None = None
        self._status = EngineStatus()
        self._thread = threading.Thread(target=self._run, name="memeify-engine", daemon=True)

    # ------------------------------------------------------------------ called from the window thread

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout)

    def snapshot(self) -> tuple[np.ndarray | None, EngineStatus]:
        """Latest preview (RGB) and a copy of the status."""
        with self._lock:
            return self._preview, dataclasses.replace(self._status)

    def switch_source(self, source: Source) -> None:
        self._commands.put(("switch_source", source))

    def set_debug(self, on: bool) -> None:
        self._commands.put(("set_debug", on))

    def start_recording(self, path: Path) -> None:
        self._commands.put(("start_recording", path))

    def stop_recording(self) -> None:
        self._commands.put(("stop_recording", None))

    def toggle_pause(self) -> None:
        self._commands.put(("toggle_pause", None))

    def test_boom(self) -> None:
        self._commands.put(("test_boom", None))

    def reopen_output(self) -> None:
        self._commands.put(("reopen_output", None))

    # ------------------------------------------------------------------ engine thread

    def _set(self, **changes) -> None:
        with self._lock:
            for k, v in changes.items():
                setattr(self._status, k, v)

    def _run(self) -> None:
        try:
            with contextlib.ExitStack() as stack:
                self._setup(stack)
                self._set(state="running", message=f"{EFFECT_NAME} ready")
                self._loop()
        except MemeifyError as e:
            log.error("%s", e)
            self._set(state="error", message=str(e))
            return
        except Exception as e:  # keep the window alive and show what went wrong
            log.exception("engine crashed")
            self._set(state="error", message=f"memeify stopped: {e}")
            return
        self._set(state="stopped", message="stopped")

    def _setup(self, stack: contextlib.ExitStack) -> None:
        cfg = self.cfg
        self.size = (cfg.output.width, cfg.output.height)
        self.fps = cfg.output.fps
        self._idle = idle_frame(self.size, "memeify: no camera selected")
        self._camera = None
        self._recorder = None
        self._vcam = None
        stack.callback(self._close_camera)
        stack.callback(self._close_recorder)
        stack.callback(self._close_output)

        self._open_output()  # first, so meeting apps see the memeify camera as early as possible
        if self._initial_source is not None:
            self._open_source(self._initial_source)

        self._set(message="Loading models (first run downloads about 11 MB)...")
        self._tracker = self._tracker_factory(cfg)
        stack.callback(self._tracker.close)
        self._detector = ThrowDetector(cfg.gesture)
        sprite = ExplosionSprite.from_folder_or_placeholder(cfg.path(cfg.explosion.asset_dir), cfg.explosion.fps)
        self._effects = EffectRenderer(sprite, cfg.explosion.scale, cfg.explosion.behind_person, cfg.explosion.drop)
        sprite.prepare(self._effects.sprite_height(self.size[1]))

        self._sound = None
        if cfg.sound.enabled:
            configured = cfg.path(cfg.sound.file) if cfg.sound.file else None
            self._sound = SoundPlayer(find_sound(cfg.path("assets"), configured), cfg.sound.volume)
            stack.callback(self._sound.close)
        self._hotkey = GlobalHotkey(cfg.hotkeys.global_pause) if cfg.hotkeys.global_pause else None
        if self._hotkey:
            stack.callback(self._hotkey.close)

        self._paused = False
        self._last_throw = None
        self._failures = 0
        self._t0 = self._clock()

    def _loop(self) -> None:
        fps_meter, prev_tick = float(self.fps), time.perf_counter()
        while not self._stop.is_set():
            self._handle_commands()
            if self._hotkey is not None and self._hotkey.consume():
                self._toggle_pause()

            frame = self._read()
            t = self._clock() - self._t0
            track = None
            if frame is None:
                if self._camera is not None:
                    time.sleep(0.005)  # a dropped frame; try again
                    continue
                frame = self._idle
                time.sleep(1 / self.fps)
            else:
                frame = fit_frame(frame, self.size)
                if not self._paused:
                    track = self._detect(frame, t)

            out = self._effects.render(frame, t, lambda: self._tracker.person_mask(frame, t))
            if self._vcam is not None:
                self._vcam.send(out)
            if self._recorder is not None:
                self._recorder.write(out, t)

            now = time.perf_counter()
            if now > prev_tick:
                fps_meter = 0.9 * fps_meter + 0.1 / (now - prev_tick)
            prev_tick = now
            self._publish(out, track, t, fps_meter)

    def _detect(self, frame: np.ndarray, t: float):
        # Nothing can trigger during the cooldown, so skip the object detector (the slowest
        # model) exactly while the explosion is playing.
        track = self._tracker.process(frame, t, detect_objects=self._detector.cooldown_remaining(t) == 0)
        event = self._detector.update(track.observation)
        if event:
            landing = landing_point(event.position, event.velocity, self.cfg.explosion.lead_time, self.size,
                                    self.cfg.explosion.edge_margin)
            self._boom(landing, t)
            self._last_throw = (event, landing)
            what = event.label or "object"
            log.info("BOOM: %s thrown with the %s hand, target %d,%d", what, event.side, *landing)
            self._set(last_boom=f"boom! ({what}, {event.side} hand)")
        return track

    def _publish(self, out: np.ndarray, track, t: float, fps: float) -> None:
        scale = PREVIEW_WIDTH / out.shape[1]
        view = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if self._paused:
            hint = f"'{self.cfg.hotkeys.pause}'" + (f" or {self.cfg.hotkeys.global_pause}"
                                                    if self._hotkey is not None and self._hotkey.active else "")
            draw_paused(view, hint)
        if self._debug:
            draw_debug(view, scale, track, self._detector, t, fps, self._last_throw)
        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        with self._lock:
            self._preview = rgb
            self._status.fps = fps

    def _read(self) -> np.ndarray | None:
        if self._camera is None:
            return None
        frame = self._camera.read()
        if frame is not None:
            self._failures = 0
            return frame
        self._failures += 1
        if self._failures > MAX_READ_FAILURES:
            name = self._camera.name
            self._close_camera()
            self._set(camera_error=f"lost '{name}' (unplugged, or taken by another app?)")
        return None

    def _boom(self, center, t: float) -> None:
        self._effects.trigger(center, t)
        if self._sound is not None:
            self._sound.play()

    def _toggle_pause(self) -> None:
        self._paused = not self._paused
        self._detector.reset()
        self._set(paused=self._paused)
        log.info("detection %s", "paused" if self._paused else "resumed")

    # ------------------------------------------------------------------ commands

    def _handle_commands(self) -> None:
        while True:
            try:
                name, arg = self._commands.get_nowait()
            except queue.Empty:
                return
            if name == "switch_source":
                self._close_camera()
                self._open_source(arg)
                self._detector.reset()
            elif name == "set_debug":
                self._debug = bool(arg)
            elif name == "start_recording":
                self._start_recording(arg)
            elif name == "stop_recording":
                self._close_recorder()
            elif name == "toggle_pause":
                self._toggle_pause()
            elif name == "test_boom":
                self._boom((self.size[0] / 2, self.size[1] * 0.3), self._clock() - self._t0)
            elif name == "reopen_output":
                self._close_output()
                self._open_output()

    def _open_source(self, source: Source) -> None:
        self._set(source=source.name, camera_error=None)
        try:
            self._camera = self._camera_factory(self.cfg.camera, source)
            self._failures = 0
            log.info("input: %s", source.name)
        except CameraError as e:
            self._camera = None
            self._set(camera_error=str(e))

    def _close_camera(self) -> None:
        if self._camera is not None:
            self._camera.close()
            self._camera = None

    def _open_output(self) -> None:
        self._vcam, status = self._output_opener(self.size[0], self.size[1], self.fps,
                                                 self.cfg.output.virtual_camera, self.cfg.output.virtual_camera_device)
        self._set(output=status)
        log.info("%s", status.message)

    def _close_output(self) -> None:
        if self._vcam is not None:
            self._vcam.close()
            self._vcam = None

    def _start_recording(self, path: Path) -> None:
        self._close_recorder()
        try:
            self._recorder = Recorder(path, self.size[0], self.size[1], self.fps)
        except MemeifyError as e:
            self._set(camera_error=str(e))
            return
        self._set(recording=path, recording_since=time.monotonic())
        log.info("recording to %s", path)

    def _close_recorder(self) -> None:
        if self._recorder is not None:
            self._recorder.close()
            self._set(recording=None, recording_since=None, last_recording=self._recorder.path)
            self._recorder = None
