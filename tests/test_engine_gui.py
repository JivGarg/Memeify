"""The processing engine and the window, with fake cameras / tracker / virtual camera (no hardware)."""

import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from memeify.cameras import CameraInfo
from memeify.capture import CameraError
from memeify.config import Config, load_state
from memeify.engine import Engine, EngineStatus
from memeify.gesture import ArmObservation, Observation
from memeify.tracking import TrackResult
from memeify.virtualcam import OutputStatus

CAM_A = CameraInfo("Cam A", 0, cv2.CAP_DSHOW, physical=True)
CAM_B = CameraInfo("Cam B", 1, cv2.CAP_DSHOW)
BROKEN = CameraInfo("Busy Cam", 2, cv2.CAP_DSHOW)
GREY = 200


class FakeCamera:
    opened, closed = [], []

    def __init__(self, cfg, source):
        if source == BROKEN:
            raise CameraError("could not open 'Busy Cam'")
        self.name, self.is_file = source.name, False
        FakeCamera.opened.append(source.name)

    def read(self):
        time.sleep(0.005)
        return np.full((240, 320, 3), GREY, np.uint8)  # 4:3, gets fitted to the 16:9 output

    def close(self):
        FakeCamera.closed.append(self.name)


class FakeTracker:
    def process(self, frame, t, detect_objects=True):  # like the real one: both arms, nobody in view
        return TrackResult(Observation(t, {"left": ArmObservation(), "right": ArmObservation()}))

    def person_mask(self, frame, t):
        return None

    def close(self):
        pass


class FakeOutput:
    def __init__(self):
        self.frames, self.closed = [], False

    def send(self, frame):
        self.frames.append(frame.copy())

    def close(self):
        self.closed = True


def make_engine(source=CAM_A, **kw):
    cfg = Config()
    cfg.output.width, cfg.output.height = 320, 180
    cfg.sound.enabled = False
    cfg.hotkeys.global_pause = None
    outputs = []

    def opener(w, h, fps, choice, device):
        outputs.append(FakeOutput())
        return outputs[-1], OutputStatus("memeify", "memeify camera: on")

    FakeCamera.opened, FakeCamera.closed = [], []
    engine = Engine(cfg, source, camera_factory=FakeCamera, tracker_factory=lambda c: FakeTracker(),
                    output_opener=opener, **kw)
    return engine, outputs


def wait_for(condition, timeout=10.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("timed out waiting for the engine")


@pytest.fixture
def running():
    started = []

    def start(source=CAM_A, **kw):
        engine, outputs = make_engine(source, **kw)
        engine.start()
        started.append(engine)
        wait_for(lambda: outputs and len(outputs[0].frames) >= 3)
        return engine, outputs

    yield start
    for engine in started:
        engine.stop()


def test_frames_flow_to_the_memeify_camera_at_the_output_size(running):
    engine, outputs = running()
    frame = outputs[0].frames[-1]
    assert frame.shape == (180, 320, 3) and frame.mean() == pytest.approx(GREY, abs=1)
    preview, status = engine.snapshot()
    assert preview.shape == (360, 640, 3)
    assert status.state == "running" and status.source == "Cam A" and status.output.device == "memeify"


def test_switching_cameras_closes_the_old_one(running):
    engine, _ = running()
    engine.switch_source(CAM_B)
    wait_for(lambda: engine.snapshot()[1].source == "Cam B" and "Cam B" in FakeCamera.opened)
    assert FakeCamera.closed == ["Cam A"]


def test_busy_camera_is_reported_and_the_output_keeps_going(running):
    engine, outputs = running()
    engine.switch_source(BROKEN)
    wait_for(lambda: engine.snapshot()[1].camera_error is not None)
    count = len(outputs[0].frames)
    wait_for(lambda: len(outputs[0].frames) > count + 2)  # meeting apps don't freeze
    assert outputs[0].frames[-1].mean() < 100              # they see the "no camera" card


def test_starting_without_a_camera_shows_the_idle_card(running):
    engine, outputs = running(source=None)
    assert outputs[0].frames[-1].mean() < 100
    assert engine.snapshot()[1].source is None


def test_record_button_writes_an_mp4(running, tmp_path):
    engine, outputs = running()
    path = tmp_path / "recordings" / "clip.mp4"
    engine.start_recording(path)
    wait_for(lambda: engine.snapshot()[1].recording == path)
    time.sleep(0.3)
    engine.stop_recording()
    wait_for(lambda: engine.snapshot()[1].last_recording == path)
    cap = cv2.VideoCapture(str(path))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= 3
    cap.release()


def test_debug_view_changes_the_preview_only(running):
    engine, outputs = running()
    plain = engine.snapshot()[0]
    engine.set_debug(True)
    wait_for(lambda: engine.snapshot()[0][-5:].mean() < plain[-5:].mean() - 50)  # dimmed debug panel
    assert outputs[0].frames[-1].mean() == pytest.approx(GREY, abs=1)            # the camera output is clean


def test_test_boom_draws_an_explosion_on_the_output(running):
    engine, outputs = running()
    engine.test_boom()
    wait_for(lambda: np.abs(outputs[0].frames[-1].astype(int) - GREY).max() > 60)


def test_pause_shows_in_the_status(running):
    engine, _ = running()
    engine.toggle_pause()
    wait_for(lambda: engine.snapshot()[1].paused)


def test_stop_releases_everything():
    engine, outputs = make_engine()
    engine.start()
    wait_for(lambda: outputs and outputs[0].frames)
    engine.stop()
    assert engine.snapshot()[1].state == "stopped"
    assert FakeCamera.closed == ["Cam A"] and outputs[0].closed


def test_reopen_output_after_installing(running):
    engine, outputs = running()
    engine.reopen_output()
    wait_for(lambda: len(outputs) == 2 and outputs[1].frames)
    assert outputs[0].closed


# --------------------------------------------------------------------------- the window

class StubEngine:
    def __init__(self):
        self.calls = []
        self.status = EngineStatus(state="running", source="Cam A",
                                   output=OutputStatus(None, "memeify camera: not installed", missing=True))
        self.preview = np.zeros((360, 640, 3), np.uint8)

    def __getattr__(self, name):  # start, stop, switch_source, set_debug, start_recording, ...
        return lambda *args: self.calls.append((name, *args))

    def snapshot(self):
        return self.preview, self.status


def pump(root, seconds=0.2):
    """Let Tk run its timers (the window refreshes every 15 ms)."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        root.update()
        time.sleep(0.01)


@pytest.fixture(scope="module")
def tk_root():
    # One Tk interpreter for all window tests: creating and destroying several in one process
    # makes Tk on Windows intermittently lose its init.tcl. The real app only ever creates one.
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError as e:
        pytest.skip(f"no display ({e})")
    root.withdraw()
    yield root
    root.destroy()


@pytest.fixture
def window(tk_root, tmp_path):
    import tkinter as tk
    from memeify.gui import MemeifyApp
    top = tk.Toplevel(tk_root)
    cfg = Config(base_dir=tmp_path)
    engine = StubEngine()
    app = MemeifyApp(top, engine, cfg, lambda exclude=(): [CAM_A, CAM_B], CAM_A)
    pump(top)
    yield app, engine, top, tmp_path
    if app._after_id is not None:
        top.after_cancel(app._after_id)
    try:
        top.destroy()
    except tk.TclError:
        pass


def test_window_lists_cameras_and_switches(window):
    app, engine, root, base = window
    assert list(app.camera_box["values"]) == ["Cam A", "Cam B"] and app.camera_box.get() == "Cam A"
    app.camera_box.current(1)
    app._on_camera_selected()
    assert ("switch_source", CAM_B) in engine.calls
    assert load_state(base) == {"camera": "Cam B"}  # remembered for next time


def test_window_debug_and_record_controls(window):
    app, engine, root, base = window
    app.debug_check.invoke()
    assert ("set_debug", True) in engine.calls
    app.record_button.invoke()
    (name, path), = [c for c in engine.calls if c[0] == "start_recording"]
    assert path.parent == base / "recordings" and path.suffix == ".mp4"


def test_window_offers_install_only_when_missing(window):
    app, engine, root, base = window
    assert app.install_frame.winfo_manager() == "grid"  # shown
    engine.status.output = OutputStatus("memeify", "memeify camera: on")
    pump(root)
    assert app.install_frame.winfo_manager() == ""      # hidden
    assert "memeify camera: on" in app.status.cget("text")


def test_closing_the_window_stops_the_engine(window):
    app, engine, root, base = window
    app.close()
    assert ("stop",) in engine.calls
