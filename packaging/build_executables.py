#!/usr/bin/env python3
"""Build onedir frozen CLI + GUI with PyInstaller (torch excluded).

Run from the repo root after ``pip install -e ".[packaging]"``:

    python packaging/build_executables.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from PyInstaller.__main__ import run as pyinstaller_run

ROOT = Path(__file__).resolve().parents[1]
PACKAGING = Path(__file__).resolve().parent
DIST = ROOT / "dist"
BUILD = ROOT / "build" / "pyinstaller"

# Keep default freezes lean: torch is an optional [ml] extra, not shipped here.
EXCLUDES = [
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
]

EXCLUDES_CLI = [
    *EXCLUDES,
    "PySide6",
    "shiboken6",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]

COLLECT_CLI = [
    "numba",
    "llvmlite",
    "soundfile",
]

COLLECT_GUI = [
    *COLLECT_CLI,
    "PySide6",
    "shiboken6",
]

HIDDEN_CORE = [
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

HIDDEN_GUI = [
    *HIDDEN_CORE,
    "supermap_cd.gui",
]


def _common_args(
    name: str,
    *,
    collect: list[str],
    excludes: list[str],
    hidden: list[str],
) -> list[str]:
    args = [
        "--noconfirm",
        "--clean",
        "--onedir",
        f"--name={name}",
        f"--distpath={DIST}",
        f"--workpath={BUILD / name}",
        f"--specpath={BUILD}",
        f"--paths={ROOT / 'src'}",
    ]
    for mod in excludes:
        args.append(f"--exclude-module={mod}")
    for pkg in collect:
        args.append(f"--collect-all={pkg}")
    for mod in hidden:
        args.append(f"--hidden-import={mod}")
    return args


def build_one(
    name: str,
    script: Path,
    *,
    console: bool,
    collect: list[str],
    excludes: list[str],
    hidden: list[str],
) -> None:
    print(f"=== Building {name} (console={console}) ===")
    args = _common_args(name, collect=collect, excludes=excludes, hidden=hidden)
    args.append("--console" if console else "--windowed")
    args.append(str(script))
    pyinstaller_run(args)


def main() -> int:
    if not (ROOT / "src" / "supermap_cd").is_dir():
        print("Run from a checkout that contains src/supermap_cd", file=sys.stderr)
        return 1

    for name in ("supermap-cd", "SuperMap-Converter"):
        target = DIST / name
        if target.exists():
            shutil.rmtree(target)

    build_one(
        "supermap-cd",
        PACKAGING / "cli_main.py",
        console=True,
        collect=COLLECT_CLI,
        excludes=EXCLUDES_CLI,
        hidden=HIDDEN_CORE,
    )
    build_one(
        "SuperMap-Converter",
        PACKAGING / "gui_main.py",
        console=False,
        collect=COLLECT_GUI,
        excludes=EXCLUDES,
        hidden=HIDDEN_GUI,
    )
    print(f"Artifacts under {DIST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
