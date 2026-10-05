# Assets

## Explosion animation: `assets/explosion/`

A PNG sequence, one file per frame, played in natural filename order (`frame_2.png`
before `frame_10.png`). The bundled explosion (22 frames, 12.5 fps) was converted from
an animated WebP.

- **To use another GIF / WebP / APNG**, run
  `python -m memeify.convert path/to/boom.gif --replace`. The app itself only reads PNGs.
- **Transparency:** frames with an alpha channel use it. Frames without one (most
  explosion clips are shot on black) are brightness-keyed: black becomes see-through and
  the fire glows over your background.
- **Speed:** `fps.txt` holds the playback rate. The converter writes it from the clip's
  own timing. Setting `explosion.fps` in `config.yaml` overrides it. Without either, 30 fps is used.
- If the folder has no PNGs, a placeholder explosion drawn in code is used instead.
- Square-ish frames work best. Size is set by `explosion.scale`, so any resolution is fine.

## Sound: `assets/boom.wav` (or `.ogg` / `.mp3`)

The first `.wav`, `.ogg` or `.mp3` file found directly in `assets/` plays on each
explosion. To use a specific file instead, set `sound.file` in `config.yaml`.

Only use assets you have the right to use, especially if you share recordings or publish this repo.
