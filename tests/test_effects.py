"""Explosion sprites, blending, the behind-the-person composite, the GIF converter, and sound lookup."""

import cv2
import numpy as np
import pytest
from PIL import Image

from memeify.config import PROJECT_DIR
from memeify.convert import convert
from memeify.effects import (
    EffectRenderer, ExplosionSprite, alpha_blend, find_sound, load_png_sequence, luma_key, make_placeholder_frames,
)


@pytest.fixture(scope="module")
def placeholder():
    return make_placeholder_frames(size=96, count=12)


def solid_sprite(color=(0, 0, 255), size=20, frames=3, fps=30):
    frame = np.zeros((size, size, 4), np.uint8)
    frame[..., :3] = color
    frame[..., 3] = 255
    return ExplosionSprite([frame.copy() for _ in range(frames)], fps)


def test_placeholder_is_an_animated_bgra_sequence(placeholder):
    assert len(placeholder) == 12
    assert all(f.shape == (96, 96, 4) and f.dtype == np.uint8 for f in placeholder)
    assert placeholder[2][48, 48, 3] > 200    # solid fireball early on
    assert placeholder[-1][..., 3].max() < 10  # faded out by the end
    assert placeholder[0][0, 0, 3] == 0        # corners stay transparent


def test_placeholder_is_deterministic():
    a, b = make_placeholder_frames(size=48, count=4), make_placeholder_frames(size=48, count=4)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_alpha_blend_paints_opaque_pixels():
    frame = np.zeros((100, 100, 3), np.uint8)
    box = alpha_blend(frame, solid_sprite().get(0, 20), (50, 50))
    assert box == (40, 40, 60, 60)
    assert tuple(frame[50, 50]) == (0, 0, 255)
    assert tuple(frame[10, 10]) == (0, 0, 0)


def test_alpha_blend_clips_at_the_frame_edge():
    frame = np.zeros((100, 100, 3), np.uint8)
    sprite = solid_sprite().get(0, 20)
    assert alpha_blend(frame, sprite, (0, 0)) == (0, 0, 10, 10)
    assert tuple(frame[5, 5]) == (0, 0, 255)
    assert alpha_blend(frame, sprite, (-50, 500)) is None  # entirely outside: no-op


def test_alpha_blend_respects_partial_alpha():
    frame = np.full((10, 10, 3), 200, np.uint8)
    sprite = np.zeros((10, 10, 4), np.uint8)
    sprite[..., 3] = 128  # 50% black
    alpha_blend(frame, ExplosionSprite([sprite], 30).get(0, 10), (5, 5))
    assert abs(int(frame[5, 5, 0]) - 100) <= 2


def test_renderer_leaves_the_input_frame_alone_and_finishes():
    frame = np.zeros((60, 80, 3), np.uint8)
    fx = EffectRenderer(solid_sprite(frames=3, fps=30), scale=1.0, behind_person=False)
    assert fx.render(frame, 0.0) is frame  # nothing running: no copy
    fx.trigger((40, 30), 0.0)
    out = fx.render(frame, 0.05)
    assert out is not frame and frame.max() == 0 and out.max() == 255
    assert fx.render(frame, 0.2) is frame  # 3 frames at 30 fps = 0.1 s, so it's over
    assert not fx.active


def test_behind_person_keeps_the_person_on_top():
    frame = np.full((60, 80, 3), 30, np.uint8)
    person = np.zeros((60, 80), np.float32)
    person[:, :40] = 1.0  # the person fills the left half
    fx = EffectRenderer(solid_sprite(size=60), scale=2.0, behind_person=True)
    fx.trigger((40, 30), 0.0)
    out = fx.render(frame, 0.0, person_mask=lambda: person)
    assert np.array_equal(out[:, :40], frame[:, :40])   # person untouched
    assert tuple(out[30, 60]) == (0, 0, 255)             # background exploded


def test_mask_is_only_computed_while_an_explosion_runs():
    calls = []
    fx = EffectRenderer(solid_sprite(), behind_person=True)
    fx.render(np.zeros((40, 40, 3), np.uint8), 0.0, person_mask=lambda: calls.append(1))
    assert calls == []


def test_png_sequence_loads_in_natural_order_with_alpha(tmp_path):
    for n in (10, 2, 1):
        img = np.zeros((8, 8, 4), np.uint8)
        img[..., 3] = n  # tag each frame through its alpha
        cv2.imwrite(str(tmp_path / f"boom_{n}.png"), img)
    cv2.imwrite(str(tmp_path / "boom_11.png"), np.full((8, 8, 3), (0, 128, 255), np.uint8))  # no alpha
    frames = load_png_sequence(tmp_path)
    assert [int(f[0, 0, 3]) for f in frames] == [1, 2, 10, 255]  # no-alpha frame is luma-keyed: bright = opaque
    assert all(f.shape[2] == 4 for f in frames)


def test_missing_or_empty_asset_folder_falls_back_to_placeholder(tmp_path):
    assert len(ExplosionSprite.from_folder_or_placeholder(tmp_path / "nope", 30).frames) == 30
    slow = ExplosionSprite.from_folder_or_placeholder(tmp_path, 12.5)
    assert len(slow.frames) == 12  # placeholder stays about a second long at any frame rate


def test_playback_rate_comes_from_fps_txt_unless_configured(tmp_path):
    cv2.imwrite(str(tmp_path / "f0.png"), np.zeros((8, 8, 4), np.uint8))
    assert ExplosionSprite.from_folder_or_placeholder(tmp_path).fps == 30  # no fps.txt
    (tmp_path / "fps.txt").write_text("12.5")
    assert ExplosionSprite.from_folder_or_placeholder(tmp_path).fps == 12.5
    assert ExplosionSprite.from_folder_or_placeholder(tmp_path, 24).fps == 24  # config wins


def test_luma_key_turns_black_transparent_and_keeps_fire_colours():
    px = np.array([[[0, 0, 0], [8, 4, 10], [0, 128, 255], [0, 64, 128]]], np.uint8)  # BGR
    out = luma_key(px)
    assert out[0, 0, 3] == 0 and out[0, 1, 3] == 0   # black and dark noise: see-through
    assert tuple(out[0, 2]) == (0, 128, 255, 255)      # bright orange: opaque, unchanged
    assert tuple(out[0, 3, :3]) == (0, 128, 255)       # dim orange: same hue...
    assert 100 < out[0, 3, 3] < 140                    # ...about half transparent


def test_bundled_explosion_loads():
    sprite = ExplosionSprite.from_folder_or_placeholder(PROJECT_DIR / "assets" / "explosion")
    assert len(sprite.frames) == 22 and sprite.fps == 12.5
    assert sprite.frames[0][0, 0, 3] == 0  # black corners keyed out


def test_drop_places_the_explosion_below_the_target():
    frame = np.zeros((100, 100, 3), np.uint8)
    fx = EffectRenderer(solid_sprite(size=20), scale=20 / 70, behind_person=False, drop=0.5)  # 20 px sprite
    fx.trigger((50, 50), 0.0)
    out = fx.render(frame, 0.0)
    rows = np.where(out[..., 2].max(axis=1) == 255)[0]
    assert (rows.min(), rows.max()) == (50, 69)  # centred 10 px (half its height) below the target


def make_gif(path, frames, duration=80):
    images = [Image.fromarray(f) for f in frames]
    images[0].save(path, save_all=True, append_images=images[1:], duration=duration, loop=0)


def test_converter_writes_keyed_png_sequence_and_fps(tmp_path):
    frames = []
    for i in range(3):
        f = np.zeros((32, 32, 3), np.uint8)
        f[8:24, 8:24] = (255, 160 + i * 20, 40)  # orange square on black (RGB)
        frames.append(f)
    make_gif(tmp_path / "boom.gif", frames)
    out = tmp_path / "out"
    assert convert(tmp_path / "boom.gif", out) == (3, 12.5)
    assert sorted(p.name for p in out.glob("*.png")) == ["frame_000.png", "frame_001.png", "frame_002.png"]
    assert (out / "fps.txt").read_text().strip() == "12.5"
    loaded = load_png_sequence(out)
    assert loaded[0][0, 0, 3] == 0 and loaded[0][16, 16, 3] == 255


def test_converter_wont_overwrite_without_replace(tmp_path):
    make_gif(tmp_path / "a.gif", [np.full((8, 8, 3), 200, np.uint8), np.full((8, 8, 3), 100, np.uint8)])
    out = tmp_path / "out"
    convert(tmp_path / "a.gif", out)
    with pytest.raises(FileExistsError):
        convert(tmp_path / "a.gif", out)
    assert convert(tmp_path / "a.gif", out, replace=True)[0] == 2


def test_sprite_resizes_to_the_requested_height(placeholder):
    frame = ExplosionSprite(placeholder, 30).get(3, 200)
    assert frame.alpha.shape == (200, 200) and frame.bgr.shape == (200, 200, 3)
    assert np.allclose(frame.alpha + frame.inv_alpha, 1)


def test_find_sound(tmp_path):
    assert find_sound(tmp_path, None) is None
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "boom.ogg").write_bytes(b"")
    assert find_sound(tmp_path, None).name == "boom.ogg"
    assert find_sound(tmp_path, tmp_path / "missing.wav") is None
