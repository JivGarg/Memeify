"""Turn an animated GIF / WebP / APNG into the PNG sequence memeify plays.

    python -m memeify.convert explosion.gif            # -> assets/explosion/
    python -m memeify.convert boom.webp --out other/ --replace

Frames with real transparency keep it. Frames shot on black (most explosion
clips) are luma-keyed, so black becomes see-through. The clip's frame rate is
saved to fps.txt so it plays back at its original speed.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

from .config import PROJECT_DIR
from .effects import FPS_FILE, luma_key

DEFAULT_FRAME_MS = 100  # GIF frames with no duration usually play at 10 fps


def read_animation(path: Path) -> tuple[list[np.ndarray], float]:
    """All frames as BGRA, plus the playback rate in fps."""
    from PIL import Image, ImageSequence

    with Image.open(path) as im:
        frames, durations = [], []
        for frame in ImageSequence.Iterator(im):
            durations.append(frame.info.get("duration") or None)
            rgba = np.array(frame.convert("RGBA"))
            bgr = cv2.cvtColor(rgba[..., :3], cv2.COLOR_RGB2BGR)
            if rgba[..., 3].min() < 255:  # the file has its own transparency
                frames.append(np.dstack([bgr, rgba[..., 3]]))
            else:
                frames.append(luma_key(bgr))
    known = [d for d in durations if d]
    frame_ms = statistics.median(known) if known else DEFAULT_FRAME_MS
    return frames, 1000 / frame_ms


def convert(src: Path, out: Path, replace: bool = False) -> tuple[int, float]:
    frames, fps = read_animation(src)
    if not frames:
        raise ValueError(f"{src} has no frames")
    out.mkdir(parents=True, exist_ok=True)
    existing = list(out.glob("*.png"))
    if existing and not replace:
        raise FileExistsError(f"{out} already has {len(existing)} PNGs; pass --replace to overwrite them")
    for old in existing:
        old.unlink()
    for i, frame in enumerate(frames):
        cv2.imwrite(str(out / f"frame_{i:03d}.png"), frame)
    (out / FPS_FILE).write_text(f"{fps:g}\n", encoding="utf-8")
    return len(frames), fps


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m memeify.convert", description=__doc__.split("\n\n")[0])
    ap.add_argument("source", type=Path, help="animated .gif, .webp or .png")
    ap.add_argument("--out", type=Path, default=PROJECT_DIR / "assets" / "explosion",
                    help="folder for the PNG frames (default: assets/explosion)")
    ap.add_argument("--replace", action="store_true", help="delete PNGs already in the output folder")
    args = ap.parse_args(argv)
    try:
        count, fps = convert(args.source, args.out, args.replace)
    except (OSError, ValueError) as e:
        print(f"convert: {e}", file=sys.stderr)
        return 1
    print(f"wrote {count} frames at {fps:g} fps to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
