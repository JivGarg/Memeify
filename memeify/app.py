"""memeify: a camera app that is also a camera.

Pick any camera, and memeify passes its picture on, with meme effects, as a new
camera called "memeify" that Zoom, Meet, Teams, Discord and OBS can select.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .cameras import capture_api, list_cameras, pick_default
from .config import ConfigError, MemeifyError, load_config, load_state

log = logging.getLogger("memeify")


def keep_full_speed() -> bool:
    """Opt out of Windows 11 power throttling ("efficiency mode") for this process.

    Windows slows down processes whose window isn't the active one. Measured here, memeify
    dropped from ~27 to ~11 fps whenever Zoom or Meet had focus, which is exactly when it's used.
    """
    if sys.platform != "win32":
        return False
    import ctypes
    from ctypes import wintypes

    class PowerThrottlingState(ctypes.Structure):
        _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]

    process_power_throttling = 4    # PROCESS_INFORMATION_CLASS.ProcessPowerThrottling
    execution_speed = 0x1           # PROCESS_POWER_THROTTLING_EXECUTION_SPEED; StateMask 0 = never throttle
    state = PowerThrottlingState(1, execution_speed, 0)
    kernel32 = ctypes.windll.kernel32
    kernel32.SetProcessInformation.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    ok = bool(kernel32.SetProcessInformation(kernel32.GetCurrentProcess(), process_power_throttling,
                                             ctypes.byref(state), ctypes.sizeof(state)))
    if not ok:  # older Windows without the API: nothing to opt out of
        log.debug("couldn't opt out of power throttling (error %d)", kernel32.GetLastError())
    return ok


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="memeify",
        description="A camera app that is also a camera: pick any camera, and memeify shows up in "
                    "Zoom / Meet / OBS as a camera called 'memeify', with meme effects.",
        epilog="Install the memeify camera (Windows): memeify --install-camera.  "
               "Use your own explosion GIF: python -m memeify.convert your.gif --replace")
    ap.add_argument("--config", metavar="PATH", help="settings file (default: config.yaml)")
    ap.add_argument("--input", metavar="VIDEO",
                    help="start from a video file instead of a camera (looped), e.g. a recording of yourself "
                         "throwing, to tune the settings on the same throw again and again")
    ap.add_argument("--debug", action="store_true", help="start with the debug view on")
    ap.add_argument("--install-camera", action="store_true", help="install the 'memeify' camera (Windows), then exit")
    ap.add_argument("--uninstall-camera", action="store_true", help="remove the 'memeify' camera (Windows), then exit")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="memeify: %(message)s")
    args = parse_args(argv)
    if args.install_camera or args.uninstall_camera:  # same as python -m memeify.install_camera, for the packaged exe
        from . import install_camera
        return install_camera.main(["--uninstall"] if args.uninstall_camera else [])
    keep_full_speed()
    try:
        cfg = load_config(args.config)
        api = capture_api(cfg.camera.backend)
        input_file = Path(args.input).resolve() if args.input else None
        if input_file and not input_file.exists():
            raise ConfigError(f"input file not found: {args.input}")
    except MemeifyError as e:
        log.error("%s", e)
        return 1

    def cameras(exclude=()):
        try:
            return list_cameras(api, exclude)
        except Exception as e:  # a broken camera driver shouldn't stop the app from opening
            log.warning("couldn't list cameras: %s", e)
            return []

    initial = input_file or pick_default(cameras(), cfg.camera.name or load_state(cfg.base_dir).get("camera"))

    try:
        import tkinter as tk
    except ImportError:
        log.error("memeify's window needs Tkinter. On Linux: sudo apt install python3-tk")
        return 1
    from .engine import Engine
    from .gui import MemeifyApp

    root = tk.Tk()
    MemeifyApp(root, Engine(cfg, initial, debug=args.debug), cfg, cameras, initial, input_file, args.debug)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        pass
    return 0
