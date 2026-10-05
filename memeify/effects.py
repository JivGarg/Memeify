"""Explosion rendering: PNG-sequence (or generated placeholder) sprites alpha-blended onto frames, plus sound."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

log = logging.getLogger("memeify")

Point = tuple[float, float]
SOUND_EXTENSIONS = (".wav", ".ogg", ".mp3")
DEFAULT_FPS = 30.0
FPS_FILE = "fps.txt"  # optional, next to the PNGs: playback rate written by `python -m memeify.convert`


# --------------------------------------------------------------------------- sprites

def make_placeholder_frames(size: int = 384, count: int = 30, seed: int = 7) -> list[np.ndarray]:
    """Draw a cartoon explosion in code: fireball, shockwave ring and sparks. Returns BGRA frames."""
    rng = np.random.default_rng(seed)
    half = size / 2
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    dx, dy = (xx + 0.5 - half) / half, (yy + 0.5 - half) / half
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)

    # Lumpy outline: random sine harmonics around the circle.
    lumps = sum(rng.uniform(0.4, 1.0) * np.sin(k * theta + rng.uniform(0, 2 * np.pi)) for k in range(3, 10)) / 3
    # Billowing inside: two smooth noise fields cross-faded over the animation.
    noise = [cv2.resize(rng.random((7, 7)).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
             for _ in range(2)]
    # Heat -> colour (BGR): smoke, dark red, red, orange, yellow, white-hot.
    stops = np.array([0.0, 0.15, 0.35, 0.55, 0.75, 1.0])
    colors = np.array([(45, 45, 50), (25, 30, 130), (15, 45, 215), (20, 140, 255), (90, 230, 255), (225, 250, 255)],
                      dtype=np.float32)
    spark_angles = rng.uniform(0, 2 * np.pi, 36)
    spark_speeds = rng.uniform(0.55, 1.0, 36)

    frames = []
    for i in range(count):
        p = i / (count - 1)                                  # 0 -> 1 over the animation
        radius = 0.22 + 0.58 * (1 - (1 - p) ** 3)             # bursts out fast, then settles
        body = np.clip(1 - r / (radius * (1 + 0.12 * lumps)), 0, 1)
        turbulence = (1 - p) * noise[0] + p * noise[1]
        heat = np.clip(body ** 0.6 * (0.65 + 0.7 * turbulence) * (1.05 - p) ** 0.9, 0, 1)
        bgr = np.stack([np.interp(heat, stops, colors[:, c]) for c in range(3)], axis=-1).astype(np.float32)
        alpha = np.clip(body * 4, 0, 1) * (1 - _smoothstep(0.55, 1.0, p))

        if p < 0.35:  # shockwave ring
            ring = np.exp(-((r - (0.3 + 1.7 * p)) / 0.035) ** 2) * (1 - p / 0.35)
            bgr = _over(bgr, alpha, np.float32(255), 0.75 * ring)
            alpha = 0.75 * ring + alpha * (1 - 0.75 * ring)

        if p < 0.7:   # sparks flying outward
            sparks = np.zeros((size, size), np.float32)
            for a, s in zip(spark_angles, spark_speeds):
                dist = s * (0.25 + 0.75 * p) * half
                c = (int(half + dist * np.cos(a)), int(half + dist * np.sin(a)))
                cv2.circle(sparks, c, max(1, int(size * 0.012)), 1.0, -1, cv2.LINE_AA)
            sparks *= 1 - p / 0.7
            bgr = _over(bgr, alpha, np.array((120, 235, 255), np.float32), sparks)
            alpha = sparks + alpha * (1 - sparks)

        frames.append(np.dstack([bgr, alpha * 255]).clip(0, 255).astype(np.uint8))
    return frames


def _over(bgr, alpha, top_bgr, top_alpha):
    """Straight-alpha 'over' compositing of a colour layer onto (bgr, alpha); returns the new straight colour."""
    ta = top_alpha[..., None]
    out_a = ta + alpha[..., None] * (1 - ta)
    return np.where(out_a > 1e-6, (top_bgr * ta + bgr * alpha[..., None] * (1 - ta)) / np.maximum(out_a, 1e-6), bgr)


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    t = min(max((x - edge0) / (edge1 - edge0), 0.0), 1.0)
    return t * t * (3 - 2 * t)


def _natural_key(path: Path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", path.name)]


def luma_key(bgr: np.ndarray, black: int = 16) -> np.ndarray:
    """Turn a frame shot on black into BGRA: brightness becomes transparency.

    Meant for fire and explosion footage. Each pixel keeps its hue at full
    brightness and its original brightness becomes the alpha, so the result
    glows over any background like an additive blend. The ramp starts at
    `black`, so dark compression noise in the background fades out instead of
    turning into coloured specks.
    """
    bright = bgr.max(axis=2).astype(np.float32)
    alpha = np.clip((bright - black) / (255 - black), 0, 1)
    color = bgr.astype(np.float32) * (255 / np.maximum(bright, 1))[..., None]
    return np.dstack([color.clip(0, 255), alpha * 255]).round().astype(np.uint8)


def load_png_sequence(folder: Path) -> list[np.ndarray]:
    """Load *.png frames (sorted naturally, so frame_2 comes before frame_10) as BGRA.

    Frames without an alpha channel are assumed to be shot on black and are luma-keyed.
    """
    frames = []
    for path in sorted(folder.glob("*.png"), key=_natural_key):
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            log.warning("skipping unreadable image %s", path.name)
            continue
        if img.dtype != np.uint8:  # 16-bit PNGs
            img = (img / 257).astype(np.uint8)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 3:
            img = luma_key(img)
        frames.append(img)
    return frames


def read_fps(folder: Path) -> float | None:
    try:
        fps = float((folder / FPS_FILE).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return None
    return fps if fps > 0 else None


class SpriteFrame(NamedTuple):
    """One animation frame at display size, ready for cv2.blendLinear."""
    bgr: np.ndarray        # uint8 colour (straight, not premultiplied)
    alpha: np.ndarray      # float32 0..1
    inv_alpha: np.ndarray  # float32, 1 - alpha


class ExplosionSprite:
    """An animation as BGRA frames, plus a cache of frames resized for display."""

    def __init__(self, frames: list[np.ndarray], fps: float):
        if not frames:
            raise ValueError("an explosion needs at least one frame")
        self.frames = frames
        self.fps = fps
        self._cache: dict[tuple[int, int], SpriteFrame] = {}

    @classmethod
    def from_folder_or_placeholder(cls, folder: Path, fps: float | None = None) -> ExplosionSprite:
        """PNGs from `folder`, or the built-in placeholder if there are none.

        `fps` None means: the folder's fps.txt if it has one, otherwise 30.
        """
        frames = load_png_sequence(folder) if folder.is_dir() else []
        if frames:
            fps = fps or read_fps(folder) or DEFAULT_FPS
            log.info("explosion: %d frames at %g fps from %s", len(frames), fps, folder)
            return cls(frames, fps)
        fps = fps or DEFAULT_FPS
        log.info("explosion: no PNGs in %s, using the built-in placeholder", folder)
        return cls(make_placeholder_frames(count=max(8, round(fps))), fps)  # about one second long

    @property
    def duration(self) -> float:
        return len(self.frames) / self.fps

    def get(self, index: int, height: int) -> SpriteFrame:
        """Frame `index` scaled to `height` px."""
        key = (index, height)
        if key not in self._cache:
            src = self.frames[index]
            width = max(1, round(src.shape[1] * height / src.shape[0]))
            interp = cv2.INTER_AREA if height < src.shape[0] else cv2.INTER_LINEAR
            # Resize premultiplied so transparent pixels don't bleed dark fringes, then un-premultiply.
            rgba = src.astype(np.float32) / 255
            rgba[..., :3] *= rgba[..., 3:]
            rgba = cv2.resize(rgba, (width, height), interpolation=interp)
            alpha = np.ascontiguousarray(rgba[..., 3])
            bgr = rgba[..., :3] / np.maximum(alpha, 1e-4)[..., None]
            self._cache[key] = SpriteFrame((bgr * 255).clip(0, 255).astype(np.uint8), alpha, 1 - alpha)
        return self._cache[key]

    def prepare(self, height: int) -> None:
        """Resize every frame up front, so the first explosion doesn't stutter."""
        for i in range(len(self.frames)):
            self.get(i, height)


def alpha_blend(dst: np.ndarray, sprite: SpriteFrame, center: Point) -> tuple[int, int, int, int] | None:
    """Blend a sprite frame onto a uint8 BGR frame in place, centred at `center`.

    Parts that fall outside the frame are clipped. Returns the touched box (x0, y0, x1, y1), or None.
    """
    fh, fw = dst.shape[:2]
    sh, sw = sprite.alpha.shape
    x0, y0 = round(center[0] - sw / 2), round(center[1] - sh / 2)
    cx0, cy0, cx1, cy1 = max(x0, 0), max(y0, 0), min(x0 + sw, fw), min(y0 + sh, fh)
    if cx0 >= cx1 or cy0 >= cy1:
        return None
    src = (slice(cy0 - y0, cy1 - y0), slice(cx0 - x0, cx1 - x0))
    roi = dst[cy0:cy1, cx0:cx1]
    roi[:] = cv2.blendLinear(sprite.bgr[src], roi, sprite.alpha[src], sprite.inv_alpha[src])
    return cx0, cy0, cx1, cy1


@dataclass
class _Boom:
    start: float
    center: Point


class EffectRenderer:
    """Keeps track of running explosions and draws them onto frames."""

    def __init__(self, sprite: ExplosionSprite, scale: float = 1.0, behind_person: bool = True, drop: float = 0.0):
        self.sprite = sprite
        self.scale = scale
        self.behind_person = behind_person
        self.drop = drop  # shift the explosion below its target by this fraction of its height
        self._booms: list[_Boom] = []

    @property
    def active(self) -> bool:
        return bool(self._booms)

    def trigger(self, center: Point, t: float) -> None:
        self._booms.append(_Boom(t, center))

    def sprite_height(self, frame_height: int) -> int:
        return max(8, round(frame_height * 0.7 * self.scale))

    def render(self, frame: np.ndarray, t: float,
               person_mask: Callable[[], np.ndarray | None] | None = None) -> np.ndarray:
        """Return `frame` with explosions drawn on. `frame` itself is left untouched."""
        n = len(self.sprite.frames)
        self._booms = [b for b in self._booms if int((t - b.start) * self.sprite.fps) < n]
        if not self._booms:
            return frame

        out = frame.copy()
        height = self.sprite_height(frame.shape[0])
        boxes = []
        for boom in self._booms:
            index = max(0, int((t - boom.start) * self.sprite.fps))
            center = (boom.center[0], boom.center[1] + self.drop * height)
            box = alpha_blend(out, self.sprite.get(index, height), center)
            if box:
                boxes.append(box)

        mask = person_mask() if (self.behind_person and person_mask and boxes) else None
        if mask is not None:
            # Put the person back on top, so the explosion looks like it's behind them.
            x0, y0 = min(b[0] for b in boxes), min(b[1] for b in boxes)
            x1, y1 = max(b[2] for b in boxes), max(b[3] for b in boxes)
            m = np.ascontiguousarray(mask[y0:y1, x0:x1], dtype=np.float32)
            out[y0:y1, x0:x1] = cv2.blendLinear(frame[y0:y1, x0:x1], out[y0:y1, x0:x1], m, 1 - m)
        return out


# --------------------------------------------------------------------------- sound

def find_sound(assets_dir: Path, configured: Path | None) -> Path | None:
    if configured is not None:
        if configured.exists():
            return configured
        log.warning("sound file %s not found; continuing without sound", configured)
        return None
    if assets_dir.is_dir():
        for path in sorted(assets_dir.iterdir()):
            if path.suffix.lower() in SOUND_EXTENSIONS:
                return path
    return None


class SoundPlayer:
    """Plays one sound file without blocking. Silently does nothing if audio isn't available."""

    def __init__(self, path: Path | None, volume: float = 0.8):
        self._sound = None
        self._mixer = None
        if path is None:
            return
        try:
            os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
            import pygame
            pygame.mixer.init()
            self._sound = pygame.mixer.Sound(str(path))
            self._sound.set_volume(volume)
            self._mixer = pygame.mixer
            log.info("sound: %s", path.name)
        except Exception as e:  # no audio device, unsupported file, pygame missing...
            log.warning("sound disabled (%s)", e)

    def play(self) -> None:
        if self._sound is not None:
            self._sound.play()

    def close(self) -> None:
        if self._mixer is not None:
            self._mixer.quit()
