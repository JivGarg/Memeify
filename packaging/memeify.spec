# PyInstaller spec for memeify.exe. Build with packaging/build.ps1 (it also copies config.yaml and assets/ next to the exe).
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = [], [], []
for pkg in ("mediapipe", "cv2_enumerate_cameras"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h
hiddenimports += collect_submodules("pynput")  # picks its Windows backend at runtime

a = Analysis(
    ["launcher.py"],
    pathex=[".."],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["pytest", "scipy", "IPython", "jax", "tensorflow", "torch"],
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="memeify", console=False, upx=False)
coll = COLLECT(exe, a.binaries, a.datas, name="memeify", upx=False)
