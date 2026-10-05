# memeify

A camera app that is also a camera. Pick any camera on your computer, and memeify passes
its picture on, with meme effects, as a new camera called **memeify** that Zoom, Google
Meet, Teams, Discord and OBS can select like any webcam.

The first effect is **Throw → Boom**, based on the Fanum broccoli-throw meme: throw
something over your shoulder and it explodes behind you.

**Status:** Windows 10/11 is the tested platform. macOS and Linux code paths exist but are untested;
treat them as experimental.

```
any camera ──▶ memeify window ──▶ pose + object detection ──▶ throw detector ──▶ explosion ──▶ "memeify" camera
(webcam, phone, (pick it, watch it,                            hold → swing → let go          ├─▶ Zoom / Meet / OBS
 OBS, a file)    debug view, record)                                                          └─▶ recordings/*.mp4
```

## Setup

### Windows installer (easiest, no Python needed)

Download `memeify-setup.exe` from the project's Releases page and run it. It installs for your
user only (no admin prompt) and adds a Start-menu entry. On first start memeify downloads its models
(about 11 MB) and offers to install the **memeify** camera (one Windows admin prompt). Windows
SmartScreen may warn about an unsigned app; choose **More info → Run anyway**. Uninstalling asks
whether to remove the memeify camera too, and keeps your `recordings/` folder.

The installer is about 85 MB to download and takes about 300 MB on disk, because it bundles Python,
OpenCV and MediaPipe. Settings, models, recordings and `memeify.log` all live in the install folder
(`%LOCALAPPDATA%\Programs\memeify`). To change a setting, edit `config.yaml` there; updating
the app keeps your copy.

### From source

You need **Python 3.10–3.14 (64-bit)** and git.

```bash
git clone https://github.com/JivGarg/memeify.git
cd memeify
```

#### Windows (no OBS needed)

```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m memeify.install_camera
```

The last command adds the **memeify** camera to Windows, once. It downloads the small open-source
[Unity Capture](https://github.com/schellingb/UnityCapture) driver (two DLLs, MIT licence) from a
pinned version and checks their SHA-256 fingerprints. It then copies them to
`%ProgramData%\memeify\camera` and registers them under the name "memeify". Windows asks for admin
permission once (UAC). You can also do this later from the **Install memeify camera…** button in the
app. To remove it: `python -m memeify.install_camera --uninstall`. (The packaged app does the same with
`memeify.exe --install-camera` / `--uninstall-camera`.)

If OBS Studio is already installed, memeify falls back to OBS's virtual camera until the memeify
camera is installed.

#### macOS (Apple Silicon)

Apple only allows signed camera extensions, so on a Mac memeify uses OBS's virtual camera:

1. Install [OBS Studio](https://obsproject.com/) 30+. Open it, click **Start Virtual Camera**, approve the
   system extension in **System Settings → Privacy & Security**, then stop it and quit OBS.
2. From the cloned folder: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
3. Allow **Camera** access for your terminal when asked. For the global pause hotkey, also allow it under
   **Privacy & Security → Accessibility**.

In meeting apps the camera is called "OBS Virtual Camera". MediaPipe 1.0 has no Intel-Mac build.

#### Linux

```bash
sudo apt install v4l2loopback-dkms python3-tk python3-dev build-essential
sudo modprobe v4l2loopback devices=1 exclusive_caps=1 card_label="memeify"
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt   # from the cloned folder
```

`exclusive_caps=1` lets Chrome-based apps (Meet, Discord) see the camera. The global pause hotkey
works under X11 only.

### First run

memeify downloads three official MediaPipe models (about 11 MB) into `models/`: pose, object
detection and the person cut-out. After that it works offline.

## Privacy and permissions

memeify runs entirely on your computer. Your camera picture is never sent anywhere, and there is no
account, analytics or telemetry. What it does touch:

- **Your camera:** read while memeify is running, and nothing is recorded unless you press **Record**
  (files go to `recordings/` on your disk).
- **The network, twice:** the three MediaPipe models on first run, and the camera driver if you choose
  to install it (see below). Nothing else.
- **A camera driver (Windows, optional):** the Unity Capture DLLs are registered system-wide, which
  needs one admin approval. Every file is checked against a pinned SHA-256 fingerprint first.
  `memeify --uninstall-camera` removes it.
- **A global hotkey:** `Ctrl+Alt+B` pauses detection from any app. To listen for it, memeify installs a
  system-wide keyboard listener ([pynput](https://github.com/moses-palmer/pynput)). It only acts on that
  one combination and does not store or send keystrokes, but the process can see key presses. To turn
  it off, set `hotkeys.global_pause: null` in `config.yaml`.

## Using it

```bash
python -m memeify
```

The memeify window:

- **Camera** dropdown: every camera on the computer (webcams, phone-as-webcam apps, OBS's camera),
  listed by name without switching any of them on. **Refresh** picks up newly plugged-in ones.
  memeify remembers your choice. Its own camera is never listed, so it can't feed back into itself.
- **Live preview** of exactly what the memeify camera is sending.
- **Debug view**: shows what the detector sees (see [Tuning](#tuning)). Only in the window; the
  meeting never sees it.
- **● Record** saves the output to `recordings/memeify-<date>-<time>.mp4`. Click the "Saved…"
  label to open the folder.
- **Status bar**: memeify camera on/off, input, fps and the effect state.
- Keys while the window is focused: `p` pauses detection, `b` fires a test explosion.
  `Ctrl+Alt+B` pauses from any app (e.g. while Zoom is focused).

Then, in your meeting app, pick **memeify** as the camera:

- **Zoom:** Settings → Video → Camera
- **Google Meet:** Settings (⚙) → Video → Camera
- **Microsoft Teams:** Settings → Devices → Camera
- **Discord:** User Settings → Voice & Video → Camera
- **OBS:** Sources → + → Video Capture Device → memeify

Keep memeify running while you use its camera. When it's closed, the memeify camera shows a
plain placeholder picture.

**One app per real camera.** memeify holds the real camera it reads from. In Zoom/Meet, pick
**memeify**, not your webcam. If memeify says it can't open a camera, another app has it.

Other options: `python -m memeify --debug` starts with the debug view on, `--config my.yaml` uses
another settings file, and `--input clip.mp4` plays a video file (looped) instead of a camera.

Windows 11 normally slows apps whose window isn't in front ("efficiency mode"), which dropped memeify
from ~27 to ~11 fps behind Zoom. memeify opts out of that at startup.

## How Throw → Boom works

It fires when these happen in order, with either hand:

1. **Hold:** an object (broccoli, a bottle, a phone, a ball…) is seen touching your hand.
2. **Swing:** your wrist flies **upward past your shoulder line**, fast.
3. **Let go:** right after the swing, your hand is seen **without** the object.

The hand comes from the pose model. The object comes from a detector trained on the 80
everyday categories of the COCO dataset; about 30 handheld ones count by default
(`tracking.object_classes`). Things it doesn't know, like a pen or a crumpled paper, won't trigger it.
The explosion is placed where the object would be ~0.25 s after release, a little below that point,
and **behind** you (you're cut out and put back on top). A 3 s cooldown stops double booms.

Swinging an empty hand, raising an object slowly, putting it down, or swinging it and keeping hold
of it won't trigger it.

## Tuning

Turn on **Debug view** and throw a few times. The preview shows:

- **Grey square:** where the object detector is looking (around your hands)
- **Pink boxes:** detected objects with their name and confidence
- **Hand boxes:** green = holding an object, grey = empty
- **Panel, per arm:** phase (`IDLE → HOLDING → SWUNG → boom`), what's in the hand, and upward wrist
  speed in shoulder-widths per second

Common fixes in `config.yaml`:

| Problem | Try |
|---|---|
| Your object never gets a pink box | Hold it bigger / closer, improve lighting, lower `tracking.object_score` (e.g. 0.2), or `tracking.object_model: lite2` (slower). Check it's in `tracking.object_classes` |
| Pink box, but the hand box stays grey | Raise `tracking.touch_margin` (e.g. 0.35) |
| Throws don't trigger | Lower `gesture.wrist_speed_threshold` (e.g. 2.0), or raise `gesture.hold_memory` |
| Fires when you stretch or wave | Raise `gesture.wrist_speed_threshold` (e.g. 3.5) |
| Fires when you swing but keep hold | Raise `gesture.release_frames` (e.g. 5) |
| Fires late, or misses the release | Lower `gesture.release_frames` (e.g. 2), or raise `gesture.release_window` |
| Explosion too high / low | `explosion.drop` (fraction of its height below the target), or `explosion.lead_time` |

To tune on the same throw again and again: record yourself throwing (press `p` first so no
explosion gets baked in), then run `python -m memeify --input recordings/<file>.mp4 --debug`.

Objects the detector knows (names for `tracking.object_classes`): person, bicycle, car, motorcycle,
airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird,
cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie,
suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard,
surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich,
orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table,
toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator,
book, clock, vase, scissors, teddy bear, hair drier, toothbrush.

## Your own explosion

The bundled explosion lives in `assets/explosion/` as a PNG sequence. To swap in any animated
GIF, WebP or APNG:

```bash
python -m memeify.convert path/to/explosion.gif --replace
```

Black backgrounds become see-through automatically, and the clip keeps its own speed. For sound,
see [assets/README.md](assets/README.md).

## Sending the boom sound into the meeting (optional)

The explosion sound plays on your speakers; without headphones your mic may pick it up. To send
it cleanly, route it through a virtual audio cable mixed with your mic:

**Windows: [VB-Cable](https://vb-audio.com/Cable/)**
1. Install VB-Cable and reboot.
2. Settings → System → Sound → Volume mixer: set `python.exe`'s output to **CABLE Input**.
3. Control Panel → Sound → Recording → your microphone → Properties → Listen → tick **Listen to this
   device**, playback through **CABLE Input**.
4. In the meeting app, pick **CABLE Output** as the microphone.

**macOS: [BlackHole](https://existential.audio/blackhole/)**: in **Audio MIDI Setup**, make a
**Multi-Output Device** (speakers + BlackHole) as the system output, and an **Aggregate Device**
(microphone + BlackHole) as the meeting app's microphone. If the boom doesn't come through, change
the device order or use a mixer app such as Loopback.

## Troubleshooting

- **"memeify camera: not installed"**: run `python -m memeify.install_camera` (or click the button).
  If you declined the Windows prompt, nothing was changed; just run it again.
- **Zoom / Meet shows a placeholder instead of you**: memeify isn't running, or no camera is picked
  in its window.
- **"could not open '…'"**: another app is using that camera. Close it or point it at memeify.
- **Low frame rate**: everything runs on the CPU (~27 fps on a typical laptop, ~20 while an explosion
  plays). Keep `pose_model: lite` and `object_model: lite0`, lower `tracking.process_width`
  (e.g. 480), or set `explosion.behind_person: false`.
- **Black or garbled picture from a camera**: set `camera.fourcc: null`, or `camera.backend: msmf` on Windows.
- **No sound**: put a `.wav`/`.ogg`/`.mp3` in `assets/` and keep `sound.enabled: true`.
- **The installed app closes or won't start:** open `memeify.log` in the install folder
  (`%LOCALAPPDATA%\Programs\memeify`); it says why. Please attach it when reporting a bug.
- **Windows warns about an unknown publisher:** the installer and `memeify.exe` aren't code-signed.
  Choose **More info → Run anyway**, or build it yourself from source (see Development).
- **INFO/WARNING lines from MediaPipe at startup** (XNNPACK, absl): harmless.

## Development

```bash
pip install -r requirements-dev.txt
pytest
```

To build `memeify.exe` and the installer (Windows; needs [Inno Setup 6](https://jrsoftware.org/isinfo.php)
for the installer step):

```powershell
pip install -r requirements-build.txt
powershell -File packaginguild.ps1     # -> dist\memeify\memeify.exe and dist\memeify-setup.exe
```

The tests need no camera, no admin rights and no driver. They cover:
- the throw state machine, using synthetic per-frame sequences;
- hand/object geometry, camera listing, frame fitting and the virtual-camera fallback order;
- the driver installer (download, fingerprint check, the elevated script);
- the engine thread and the window, with fake cameras;
- effects, the GIF converter, config and the recorder.

```
memeify/
  app.py            entry point (python -m memeify); Windows power-throttling opt-out
  gui.py            the Tkinter window
  engine.py         background loop: camera → tracking → detection → effects → outputs
  cameras.py        list cameras by name without opening them
  capture.py        read a camera or video file; fit frames to the output size
  virtualcam.py     the "memeify" camera (Unity Capture on Windows, OBS on macOS, v4l2loopback on Linux)
  install_camera.py one-time Windows driver install (python -m memeify.install_camera)
  tracking.py       MediaPipe pose + object detector → per-arm observations; model download
  gesture.py        the throw state machine (pure Python)
  effects.py        explosion sprites, brightness key, blending, behind-person cut-out, sound
  convert.py        GIF / WebP / APNG → PNG sequence (python -m memeify.convert)
  output.py         MP4 recorder
  overlay.py        debug drawing
  hotkey.py         global pause hotkey
packaging/          PyInstaller spec, Inno Setup script and build.ps1 for memeify.exe / memeify-setup.exe
tests/              no-hardware tests
assets/             explosion PNG sequence (+ optional sound)
config.yaml         all settings
```

## Third-party

- [Unity Capture](https://github.com/schellingb/UnityCapture) by Bernhard Schelling (MIT): the Windows
  virtual-camera driver, downloaded by `install_camera` at commit `3ed54c3`, not bundled here.
- [MediaPipe](https://github.com/google-ai-edge/mediapipe) models by Google (Apache 2.0), downloaded on first run.

## License

MIT. See [LICENSE](LICENSE).
