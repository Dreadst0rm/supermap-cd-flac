# -*- mode: python ; coding: utf-8 -*-
# Reference CLI onedir spec. Prefer: python packaging/build_executables.py

from PyInstaller.utils.hooks import collect_all

block_cipher = None

datas, binaries, hidden = [], [], []
for pkg in ("numba", "llvmlite", "soundfile"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hidden += h

hidden += [
    "scipy.signal",
    "scipy.fft",
    "numpy",
    "mutagen",
    "mutagen.flac",
    "musicbrainzngs",
    "supermap_cd",
    "supermap_cd.cli",
    "supermap_cd.pipeline",
    "supermap_cd.io_audio",
    "supermap_cd.gapfill",
    "supermap_cd.encode",
    "supermap_cd.lossy_repair",
    "supermap_cd.ml_upscaler",
    "supermap_cd.mb",
    "supermap_cd.rip",
    "supermap_cd.conversion_log",
]

excludes = [
    "torch",
    "torchvision",
    "torchaudio",
    "tensorflow",
    "tensorboard",
    "IPython",
    "jupyter",
    "notebook",
    "matplotlib",
    "tkinter",
    "PySide6",
    "shiboken6",
]

a = Analysis(
    ["cli_main.py"],
    pathex=["../src"],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="supermap-cd",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="supermap-cd",
)
