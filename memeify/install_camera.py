"""Install (or remove) the "memeify" camera on Windows. No OBS needed.

    python -m memeify.install_camera
    python -m memeify.install_camera --uninstall

Downloads the virtual-camera driver from Unity Capture (MIT licence, two small
DLLs) at a pinned commit, checks their SHA-256 fingerprints, copies them to
%ProgramData%\\memeify\\camera and registers them under the name "memeify".
Registering a camera driver needs admin rights, so Windows asks once (UAC).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

from .virtualcam import DEVICE_NAME

UNITY_CAPTURE_COMMIT = "3ed54c325e0ad71afcf4f246c07e5e17b3d7f2d2"  # 2023-04-14, "New release with new DLL files"
DLLS = {  # file name -> SHA-256
    "UnityCaptureFilter64.dll": "72812f5363d8ecb45632253f8c8c888844b1b62e27616f3c8cc21064ccde25e5",
    "UnityCaptureFilter32.dll": "aa3ebdf03dea7f3aab3dd7b724751f49ed71672256b57c6a19aa6809cabf30ba",
}
DLL_URL = "https://raw.githubusercontent.com/schellingb/UnityCapture/{commit}/Install/{name}"


class InstallError(Exception):
    pass


def install_dir() -> Path:
    # Must be somewhere every account (and Windows' camera services) can read, not the user profile.
    return Path(os.environ.get("ProgramData", r"C:\ProgramData")) / "memeify" / "camera"


def _fetch(url: str) -> bytes:
    import ssl
    try:
        import certifi
        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()
    with urllib.request.urlopen(url, timeout=60, context=context) as resp:
        return resp.read()


def download_driver(dest: Path, fetch=_fetch) -> list[Path]:
    """Download the pinned DLLs into `dest`, refusing anything whose fingerprint doesn't match."""
    paths = []
    for name, expected in DLLS.items():
        url = DLL_URL.format(commit=UNITY_CAPTURE_COMMIT, name=name)
        try:
            data = fetch(url)
        except OSError as e:
            raise InstallError(f"couldn't download {name}: {e}") from e
        actual = hashlib.sha256(data).hexdigest()
        if actual != expected:
            raise InstallError(f"{name} doesn't match its pinned fingerprint (got {actual[:12]}..., "
                               f"expected {expected[:12]}...). Not installing it.")
        path = dest / name
        path.write_bytes(data)
        paths.append(path)
    return paths


def install_script(source_dir: Path, target: Path) -> str:
    """Batch script for the elevated step. Exit codes say which step failed."""
    lines = ["@echo off", "chcp 65001 >nul", f'if not exist "{target}" mkdir "{target}" || exit /b 1']
    for name in DLLS:
        lines.append(f'copy /y "{source_dir / name}" "{target / name}" >nul || exit /b 2')
    for name in DLLS:  # same command line as Unity Capture's own InstallCustomName.bat, plus /s (silent)
        lines.append(f'regsvr32 /s "{target / name}" "/i:UnityCaptureName={DEVICE_NAME}" || exit /b 3')
    return "\r\n".join(lines) + "\r\n"


def uninstall_script(target: Path) -> str:
    lines = ["@echo off", "chcp 65001 >nul"]
    for name in DLLS:
        lines.append(f'if exist "{target / name}" regsvr32 /u /s "{target / name}"')
    lines.append(f'if exist "{target}" rmdir /s /q "{target}"')
    return "\r\n".join(lines) + "\r\n"


def elevate_command(script: Path) -> list[str]:
    """Run `script` as administrator (one UAC prompt), wait for it, and pass its exit code back."""
    quoted = str(script).replace("'", "''")  # PowerShell single-quoted string
    return ["powershell", "-NoProfile", "-NonInteractive", "-Command",
            f"try {{ $p = Start-Process -FilePath '{quoted}' -Verb RunAs -Wait -PassThru -WindowStyle Hidden; "
            f"exit $p.ExitCode }} catch {{ exit 1223 }}"]  # 1223 = the user said no to the prompt


def run_elevated(script_text: str, workdir: Path, run=subprocess.run) -> int:
    script = workdir / "memeify-camera.cmd"
    script.write_text(script_text, encoding="utf-8")  # read as UTF-8 thanks to `chcp 65001` (non-ASCII user names)
    return run(elevate_command(script)).returncode


def is_installed() -> bool:
    import cv2
    from cv2_enumerate_cameras import enumerate_cameras
    return any(c.name.lower() == DEVICE_NAME for c in enumerate_cameras(cv2.CAP_DSHOW))


FAILURES = {
    1223: "you declined the Windows admin prompt, so nothing was changed",
    1: "couldn't create the install folder",
    2: "couldn't copy the driver files. Close any app using the memeify camera and try again",
    3: "Windows refused to register the driver",
}


def install() -> None:
    with tempfile.TemporaryDirectory(prefix="memeify-") as tmp:
        tmp_dir = Path(tmp)
        download_driver(tmp_dir)
        code = run_elevated(install_script(tmp_dir, install_dir()), tmp_dir)
    if code != 0:
        raise InstallError(FAILURES.get(code, f"the install step failed (code {code})"))
    if not is_installed():
        raise InstallError("the driver was registered but Windows doesn't list a 'memeify' camera yet. "
                           "Try restarting the computer.")


def uninstall() -> None:
    with tempfile.TemporaryDirectory(prefix="memeify-") as tmp:
        code = run_elevated(uninstall_script(install_dir()), Path(tmp))
    if code != 0:
        raise InstallError(FAILURES.get(code, f"the uninstall step failed (code {code})"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m memeify.install_camera",
                                 description="Install the 'memeify' camera (Windows, no OBS needed).")
    ap.add_argument("--uninstall", action="store_true", help="remove the memeify camera")
    args = ap.parse_args(argv)
    if sys.platform != "win32":
        print("install_camera is for Windows. On macOS memeify uses OBS's virtual camera; on Linux load "
              "v4l2loopback with card_label=memeify (see the README).", file=sys.stderr)
        return 1
    try:
        if args.uninstall:
            uninstall()
            print("The memeify camera has been removed.")
        else:
            print("Downloading the camera driver and asking Windows for permission to install it...")
            install()
            print("Done. Pick 'memeify' as your camera in Zoom, Meet, Teams, Discord or OBS.")
    except InstallError as e:
        print(f"install_camera: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
