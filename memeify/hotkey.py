"""System-wide hotkey, so detection can be paused while a meeting app has focus."""

from __future__ import annotations

import logging
import threading

log = logging.getLogger("memeify")


class GlobalHotkey:
    """Listens for a key combo in any app (pynput syntax, e.g. "<ctrl>+<alt>+b").

    Fails soft: if the OS doesn't allow a global listener (macOS without
    Accessibility permission, Linux under Wayland), it logs why and stays inactive.
    """

    def __init__(self, combo: str):
        self.combo = combo
        self._hit = threading.Event()
        self._listener = None
        try:
            from pynput import keyboard
            self._listener = keyboard.GlobalHotKeys({combo: self._hit.set})
            self._listener.start()
        except Exception as e:
            log.warning("global hotkey %s unavailable (%s); use the preview window key instead", combo, e)
            self._listener = None

    @property
    def active(self) -> bool:
        return self._listener is not None

    def consume(self) -> bool:
        """True once per press since the last call."""
        if self._hit.is_set():
            self._hit.clear()
            return True
        return False

    def close(self) -> None:
        if self._listener is not None:
            self._listener.stop()
