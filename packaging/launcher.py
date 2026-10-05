"""Entry point for the packaged memeify.exe (PyInstaller can't use memeify/__main__.py: relative imports).

The exe has no console, so log lines go to memeify.log next to it, and a failed start shows a dialog.
"""

import sys
from pathlib import Path

if getattr(sys, "frozen", False) and sys.stderr is None:
    log_path = Path(sys.executable).resolve().parent / "memeify.log"
    sys.stderr = sys.stdout = open(log_path, "w", encoding="utf-8", buffering=1)

from memeify.app import main  # noqa: E402

if __name__ == "__main__":
    code = main()
    if code and getattr(sys, "frozen", False) and not any("-camera" in a for a in sys.argv):
        from tkinter import Tk, messagebox
        root = Tk()
        root.withdraw()
        messagebox.showerror("memeify", "memeify couldn't start. The details are in memeify.log next to memeify.exe.")
    raise SystemExit(code)
