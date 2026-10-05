"""Camera listing, frame fitting, the memeify virtual camera, the driver installer, and remembered state."""

import hashlib
import sys
import types

import cv2
import numpy as np
import pytest

from memeify import install_camera as inst
from memeify.cameras import CameraInfo, capture_api, labels, list_cameras, pick_default, without_own_output
from memeify.capture import fit_frame
from memeify.config import ConfigError, load_state, save_state
from memeify.virtualcam import backend_order, open_virtual_camera

WEBCAM = CameraInfo("USB2.0 HD UVC WebCam", 1, cv2.CAP_DSHOW, physical=True)
PHONE = CameraInfo("Iriun Webcam", 0, cv2.CAP_DSHOW)
OBS = CameraInfo("OBS Virtual Camera", 2, cv2.CAP_DSHOW)
OURS = CameraInfo("memeify", 3, cv2.CAP_DSHOW)
UNITY = CameraInfo("Unity Video Capture", 4, cv2.CAP_DSHOW)


# --------------------------------------------------------------------------- cameras

def test_own_output_is_never_offered_as_an_input():
    assert without_own_output([PHONE, WEBCAM, OBS, OURS, UNITY]) == [PHONE, WEBCAM, OBS]


def test_obs_camera_hidden_only_when_it_is_the_output():
    assert without_own_output([WEBCAM, OBS], exclude=["obs virtual camera"]) == [WEBCAM]


def test_list_cameras_reads_names_from_the_enumerator(monkeypatch):
    import cv2_enumerate_cameras
    fake = [types.SimpleNamespace(name=c.name, index=c.index, backend=cv2.CAP_DSHOW, vid=0x3277 if c.physical else None)
            for c in (PHONE, WEBCAM, OURS)]
    monkeypatch.setattr(cv2_enumerate_cameras, "enumerate_cameras", lambda api: fake)
    assert list_cameras(cv2.CAP_DSHOW) == [PHONE, WEBCAM]


def test_duplicate_camera_names_get_numbered():
    twin = CameraInfo(WEBCAM.name, 5, cv2.CAP_DSHOW, physical=True)
    assert labels([WEBCAM, PHONE, twin]) == [WEBCAM.name, PHONE.name, f"{WEBCAM.name} (2)"]


def test_default_camera_prefers_the_remembered_one_then_a_physical_one():
    cams = [PHONE, WEBCAM]
    assert pick_default(cams, "iriun webcam") == PHONE      # remembered (case-insensitive)
    assert pick_default(cams, "unplugged camera") == WEBCAM  # remembered one is gone: first physical
    assert pick_default(cams) == WEBCAM
    assert pick_default([PHONE]) == PHONE
    assert pick_default([]) is None


def test_capture_api_per_platform():
    assert capture_api("auto", "win32") == cv2.CAP_DSHOW
    assert capture_api("auto", "darwin") == cv2.CAP_AVFOUNDATION
    assert capture_api("auto", "linux") == cv2.CAP_V4L2
    assert capture_api("msmf") == cv2.CAP_MSMF
    with pytest.raises(ConfigError):
        capture_api("webcam9000")


# --------------------------------------------------------------------------- fitting frames

def test_fit_frame_fills_the_output_by_cropping_the_overflow():
    frame = np.zeros((480, 640, 3), np.uint8)  # 4:3
    frame[:50] = 255                             # a band at the very top...
    frame[230:250] = 128                         # ...and one across the middle
    out = fit_frame(frame, (1280, 720))          # 16:9: scaled 2x, then 120 px cropped top and bottom
    assert out.shape == (720, 1280, 3)
    assert out[0].max() == 0                     # the top band was cropped away
    assert out[360].min() == 128                 # the middle stays in the middle


def test_fit_frame_handles_portrait_and_same_size():
    assert fit_frame(np.zeros((1280, 720, 3), np.uint8), (1280, 720)).shape == (720, 1280, 3)
    same = np.zeros((720, 1280, 3), np.uint8)
    assert fit_frame(same, (1280, 720)) is same


# --------------------------------------------------------------------------- the memeify virtual camera

def test_backend_order_per_platform():
    assert backend_order("auto", "win32") == [("unitycapture", "memeify"), ("obs", None)]
    assert backend_order("memeify", "win32") == [("unitycapture", "memeify")]
    assert backend_order("obs", "win32") == [("obs", None)]
    assert backend_order("auto", "darwin") == [("obs", None)]
    assert backend_order("auto", "linux") == [("v4l2loopback", None)]
    assert backend_order("off", "win32") == []
    with pytest.raises(ConfigError):
        backend_order("snapcamera", "win32")


def fake_pyvirtualcam(monkeypatch, working: dict):
    """A stand-in pyvirtualcam where only the backends in `working` ({backend: device name}) open."""
    calls = []

    class Camera:
        def __init__(self, width, height, fps, fmt, backend, device, print_fps):
            calls.append((backend, device))
            if backend not in working:
                raise RuntimeError(f"{backend} not installed")
            self.device = working[backend]

        def send(self, frame):
            pass

        def close(self):
            pass

    module = types.SimpleNamespace(Camera=Camera, PixelFormat=types.SimpleNamespace(BGR="BGR"))
    monkeypatch.setitem(sys.modules, "pyvirtualcam", module)
    return calls


def test_memeify_camera_is_used_when_installed(monkeypatch):
    calls = fake_pyvirtualcam(monkeypatch, {"unitycapture": "memeify", "obs": "OBS Virtual Camera"})
    vcam, status = open_virtual_camera(1280, 720, 30, "auto", platform="win32")
    assert calls == [("unitycapture", "memeify")]
    assert vcam.device == "memeify" and status.message == "memeify camera: on" and not status.missing


def test_falls_back_to_obs_and_offers_the_install(monkeypatch):
    calls = fake_pyvirtualcam(monkeypatch, {"obs": "OBS Virtual Camera"})
    vcam, status = open_virtual_camera(1280, 720, 30, "auto", platform="win32")
    assert calls == [("unitycapture", "memeify"), ("obs", None)]
    assert vcam.device == "OBS Virtual Camera" and status.missing and "OBS" in status.message


def test_no_virtual_camera_keeps_the_app_running(monkeypatch):
    fake_pyvirtualcam(monkeypatch, {})
    vcam, status = open_virtual_camera(1280, 720, 30, "auto", platform="win32")
    assert vcam is None and status.device is None and status.missing
    assert "Install memeify camera" in status.message


def test_virtual_camera_off_tries_nothing(monkeypatch):
    calls = fake_pyvirtualcam(monkeypatch, {"unitycapture": "memeify"})
    vcam, status = open_virtual_camera(1280, 720, 30, "off", platform="win32")
    assert vcam is None and calls == [] and not status.missing


# --------------------------------------------------------------------------- the driver installer

def test_driver_download_is_checked_against_the_pinned_fingerprint(tmp_path, monkeypatch):
    good = b"pretend dll"
    monkeypatch.setattr(inst, "DLLS", {"UnityCaptureFilter64.dll": hashlib.sha256(good).hexdigest()})
    urls = []
    paths = inst.download_driver(tmp_path, fetch=lambda url: urls.append(url) or good)
    assert [p.read_bytes() for p in paths] == [good]
    assert inst.UNITY_CAPTURE_COMMIT in urls[0]

    with pytest.raises(inst.InstallError, match="fingerprint"):
        inst.download_driver(tmp_path / "x", fetch=lambda url: b"tampered")
    assert not (tmp_path / "x").exists()


def test_pinned_fingerprints_cover_both_dlls():
    assert set(inst.DLLS) == {"UnityCaptureFilter64.dll", "UnityCaptureFilter32.dll"}
    assert all(len(h) == 64 for h in inst.DLLS.values())


def test_install_script_registers_both_dlls_under_the_memeify_name(tmp_path):
    src, target = tmp_path / "dl with space", tmp_path / "Program Data" / "memeify"
    script = inst.install_script(src, target)
    assert script.startswith("@echo off\r\nchcp 65001")
    for name in inst.DLLS:
        assert f'copy /y "{src / name}" "{target / name}"' in script
        assert f'regsvr32 /s "{target / name}" "/i:UnityCaptureName=memeify" || exit /b 3' in script


def test_uninstall_script_unregisters_and_removes(tmp_path):
    script = inst.uninstall_script(tmp_path)
    assert script.count("regsvr32 /u /s") == 2 and f'rmdir /s /q "{tmp_path}"' in script


def test_elevation_asks_once_and_waits(tmp_path):
    script = tmp_path / "it's here.cmd"
    cmd = inst.elevate_command(script)
    assert cmd[0] == "powershell" and "-Verb RunAs -Wait -PassThru" in cmd[-1]
    assert "'" + str(script).replace("'", "''") + "'" in cmd[-1]
    ran = []
    code = inst.run_elevated("@echo off\r\n", tmp_path, run=lambda c: ran.append(c) or types.SimpleNamespace(returncode=7))
    assert code == 7 and (tmp_path / "memeify-camera.cmd").exists() and len(ran) == 1


@pytest.mark.parametrize("code, installed, message", [
    (1223, False, "declined"),
    (2, False, "Close any app"),
    (0, False, "doesn't list"),
])
def test_install_reports_what_went_wrong(monkeypatch, code, installed, message):
    monkeypatch.setattr(inst, "download_driver", lambda dest: [])
    monkeypatch.setattr(inst, "run_elevated", lambda text, workdir: code)
    monkeypatch.setattr(inst, "is_installed", lambda: installed)
    with pytest.raises(inst.InstallError, match=message):
        inst.install()


def test_install_succeeds_when_windows_lists_the_camera(monkeypatch):
    monkeypatch.setattr(inst, "download_driver", lambda dest: [])
    monkeypatch.setattr(inst, "run_elevated", lambda text, workdir: 0)
    monkeypatch.setattr(inst, "is_installed", lambda: True)
    inst.install()


def test_installer_explains_itself_off_windows(monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert inst.main([]) == 1
    assert "OBS" in capsys.readouterr().err


# --------------------------------------------------------------------------- remembered choices

def test_state_round_trip_and_bad_file(tmp_path):
    assert load_state(tmp_path) == {}
    save_state(tmp_path, camera="Iriun Webcam")
    save_state(tmp_path, other=1)
    assert load_state(tmp_path) == {"camera": "Iriun Webcam", "other": 1}
    (tmp_path / ".memeify-state.json").write_text("{not json")
    assert load_state(tmp_path) == {}


# --------------------------------------------------------------------------- Windows power throttling

@pytest.mark.skipif(sys.platform != "win32", reason="Windows-only API")
def test_opts_out_of_windows_power_throttling():
    from memeify.app import keep_full_speed
    assert keep_full_speed() is True


def test_power_throttling_opt_out_is_a_no_op_elsewhere(monkeypatch):
    from memeify.app import keep_full_speed
    monkeypatch.setattr(sys, "platform", "linux")
    assert keep_full_speed() is False


def test_exe_flags_install_and_remove_the_camera(monkeypatch):
    from memeify import app
    calls = []
    monkeypatch.setattr(inst, "main", lambda argv=None: calls.append(argv) or 0)
    assert app.main(["--install-camera"]) == 0
    assert app.main(["--uninstall-camera"]) == 0
    assert calls == [[], ["--uninstall"]]
