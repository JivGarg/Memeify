"""The memeify window: pick a camera, watch the live picture, toggle the debug view, record."""

from __future__ import annotations

import os
import sys
import threading
import time
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox, ttk

from PIL import Image, ImageTk

from . import install_camera
from .cameras import CameraInfo, labels
from .config import Config, save_state
from .engine import EFFECT_NAME, Engine, Source

POLL_MS = 15  # how often the window picks up a new preview frame


class MemeifyApp:
    def __init__(self, root: tk.Tk | tk.Toplevel, engine: Engine, cfg: Config,
                 list_cameras: Callable[..., list[CameraInfo]], initial: Source | None,
                 input_file: Path | None = None, debug: bool = False):
        self.root, self.engine, self.cfg = root, engine, cfg
        self._list_cameras = list_cameras
        self._input_file = input_file
        self._current: Source | None = initial
        self._sources: list[Source] = []
        self._output_device: str | None = None
        self._last_preview = None
        self._photo = None
        self._install_result: tuple[int, str] | None = None
        self._after_id = None

        root.title("memeify")
        root.minsize(560, 440)
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        top = ttk.Frame(root, padding=(8, 8, 8, 0))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)
        ttk.Label(top, text="Camera:").grid(row=0, column=0, padx=(0, 6))
        self.camera_box = ttk.Combobox(top, state="readonly")
        self.camera_box.grid(row=0, column=1, sticky="ew")
        self.camera_box.bind("<<ComboboxSelected>>", self._on_camera_selected)
        ttk.Button(top, text="Refresh", command=self.refresh_cameras).grid(row=0, column=2, padx=(6, 0))

        self.canvas = tk.Canvas(root, width=640, height=360, bg="black", highlightthickness=0)
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self._image_item = self.canvas.create_image(0, 0, anchor="center")
        self._text_item = self.canvas.create_text(0, 0, fill="#dddddd", font=("Segoe UI", 12), width=520,
                                                  justify="center")

        controls = ttk.Frame(root, padding=(8, 0))
        controls.grid(row=2, column=0, sticky="ew")
        self.debug_var = tk.BooleanVar(value=debug)
        self.debug_check = ttk.Checkbutton(controls, text="Debug view", variable=self.debug_var,
                                           command=lambda: engine.set_debug(self.debug_var.get()))
        self.debug_check.pack(side="left")
        self.record_button = ttk.Button(controls, text="● Record", command=self._on_record)
        self.record_button.pack(side="left", padx=(12, 6))
        self.record_label = ttk.Label(controls, text="")
        self.record_label.pack(side="left")
        self.record_label.bind("<Button-1>", self._open_recordings)

        self.install_frame = ttk.Frame(root, padding=(8, 6, 8, 0))
        ttk.Label(self.install_frame, text="The memeify camera isn't installed, so Zoom / Meet / OBS "
                                           "can't pick it yet.").pack(side="left")
        self.install_button = ttk.Button(self.install_frame, text="Install memeify camera...",
                                         command=self._on_install)
        self.install_button.pack(side="right")

        self.status = ttk.Label(root, anchor="w", relief="sunken", padding=(6, 2))
        self.status.grid(row=4, column=0, sticky="ew", padx=8, pady=8)

        root.bind("<KeyPress-%s>" % cfg.hotkeys.pause, lambda e: engine.toggle_pause())
        root.bind("<KeyPress-%s>" % cfg.hotkeys.test_boom, lambda e: engine.test_boom())
        root.protocol("WM_DELETE_WINDOW", self.close)

        self.refresh_cameras()
        engine.start()
        self._after_id = root.after(POLL_MS, self._tick)

    # ------------------------------------------------------------------ cameras

    def refresh_cameras(self) -> None:
        cameras = self._list_cameras([self._output_device] if self._output_device else [])
        self._sources = ([self._input_file] if self._input_file else []) + cameras
        names = ([f"Video file: {self._input_file.name}"] if self._input_file else []) + labels(cameras)
        self.camera_box["values"] = names or ["(no cameras found)"]
        index = self._index_of(self._current)
        if index is not None:
            self.camera_box.current(index)
        elif names:
            self.camera_box.set("Pick a camera")
        else:
            self.camera_box.current(0)

    def _index_of(self, source: Source | None) -> int | None:
        if source is None:
            return None
        for i, s in enumerate(self._sources):
            if s == source or (isinstance(s, CameraInfo) and isinstance(source, CameraInfo) and s.name == source.name):
                return i
        return None

    def _on_camera_selected(self, _event=None) -> None:
        i = self.camera_box.current()
        if not 0 <= i < len(self._sources):
            return
        source = self._sources[i]
        if source == self._current:
            return
        self._current = source
        self.engine.switch_source(source)
        if isinstance(source, CameraInfo):
            save_state(self.cfg.base_dir, camera=source.name)

    # ------------------------------------------------------------------ recording

    def _on_record(self) -> None:
        _, status = self.engine.snapshot()
        if status.recording:
            self.engine.stop_recording()
        else:
            path = self.cfg.base_dir / "recordings" / time.strftime("memeify-%Y%m%d-%H%M%S.mp4")
            self.engine.start_recording(path)

    def _open_recordings(self, _event=None) -> None:
        folder = self.cfg.base_dir / "recordings"
        if sys.platform == "win32" and folder.is_dir():
            os.startfile(folder)  # noqa: S606 - opens Explorer on the user's own folder

    # ------------------------------------------------------------------ installing the camera

    def _on_install(self) -> None:
        self.install_button.configure(state="disabled", text="Installing... approve the Windows prompt")

        def work():
            # In-process: a packaged memeify.exe has no `python -m`, and no console to print to.
            try:
                install_camera.install()
                self._install_result = (0, "")
            except install_camera.InstallError as e:
                self._install_result = (1, str(e))
            except Exception as e:
                self._install_result = (1, f"unexpected error: {e}")

        threading.Thread(target=work, name="memeify-install", daemon=True).start()

    def _finish_install(self) -> None:
        code, text = self._install_result
        self._install_result = None
        self.install_button.configure(state="normal", text="Install memeify camera...")
        if code == 0:
            self.engine.reopen_output()
            messagebox.showinfo("memeify", "The memeify camera is installed. Pick 'memeify' as your camera "
                                           "in Zoom, Meet, Teams, Discord or OBS.")
        else:
            messagebox.showerror("memeify", text or "The install didn't finish.")

    # ------------------------------------------------------------------ refresh loop

    def _tick(self) -> None:
        preview, status = self.engine.snapshot()
        if preview is not None and preview is not self._last_preview:
            self._last_preview = preview
            self._show(preview)
        self._update_text(status, preview is None)
        if status.output.device != self._output_device:
            self._output_device = status.output.device
            self.refresh_cameras()  # e.g. hide OBS's camera from the inputs once it's the output
        if status.output.missing:
            self.install_frame.grid(row=3, column=0, sticky="ew")
        else:
            self.install_frame.grid_remove()
        if self._install_result is not None:
            self._finish_install()
        self._after_id = self.root.after(POLL_MS, self._tick)

    def _show(self, rgb) -> None:
        cw, ch = max(self.canvas.winfo_width(), 1), max(self.canvas.winfo_height(), 1)
        h, w = rgb.shape[:2]
        scale = min(cw / w, ch / h)
        image = Image.fromarray(rgb)
        if abs(scale - 1) > 0.01:
            image = image.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        self._photo = ImageTk.PhotoImage(image)  # keep a reference or Tk drops the image
        self.canvas.itemconfigure(self._image_item, image=self._photo)
        self.canvas.coords(self._image_item, cw / 2, ch / 2)

    def _update_text(self, status, no_preview: bool) -> None:
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        overlay = ""
        if status.state == "error":
            overlay = status.message
        elif no_preview:
            overlay = status.message
        elif status.camera_error:
            overlay = status.camera_error
        self.canvas.itemconfigure(self._text_item, text=overlay,
                                  fill="#ff6b6b" if status.state == "error" or status.camera_error else "#dddddd")
        self.canvas.coords(self._text_item, cw / 2, ch - 40 if not no_preview else ch / 2)
        self.canvas.tag_raise(self._text_item)

        if status.recording:
            self.record_button.configure(text="■ Stop recording")
            secs = int(time.monotonic() - (status.recording_since or time.monotonic()))
            self.record_label.configure(text=f"REC {secs // 60:02d}:{secs % 60:02d}", foreground="#c00000",
                                        cursor="")
        else:
            self.record_button.configure(text="● Record")
            if status.last_recording:
                self.record_label.configure(text=f"Saved {status.last_recording.name}", foreground="",
                                            cursor="hand2" if sys.platform == "win32" else "")

        parts = [status.output.message, status.source or "no camera"]
        if status.state == "running":
            parts.append(f"{status.fps:.0f} fps")
            parts.append("detection PAUSED" if status.paused else (status.last_boom or f"{EFFECT_NAME} ready"))
        else:
            parts.append(status.message)
        self.status.configure(text="   |   ".join(parts))

    def close(self) -> None:
        if self._after_id is not None:
            self.root.after_cancel(self._after_id)
        self.engine.stop()
        self.root.destroy()
