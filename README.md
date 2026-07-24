# SuperMap CD → 20/24-bit FLAC Ripper

Secure Windows CD ripper that extracts Red Book audio (16-bit / 44.1 kHz) and optionally expands it with an **SBM-style** pipeline:

1. Synthesize **+16 bits** inside each 16-bit quantization bin → **32-bit** working precision  
2. **Noise-shaped quantize** down to user-selected **20-bit** or **24-bit**  
3. Pack into **24-bit FLAC** (20-bit is left-aligned)

The same expand path works on existing files via `supermap-cd upconvert` (FLAC / WAV / Ogg Vorbis natively; anything else via **ffmpeg**).

Optional **lossy repair** cleans common MP3/AAC artifacts and applies bandwidth extension before expand. Optional **ML upscaler** supplies the 32-bit preference prior (torch residual CNN if `supermap-cd[ml]` is installed, otherwise a numpy STFT spectral residual).

## Honesty note

**Super Bit Mapping (SBM) is encode-only.** Extra bits were never stored on the CD. SuperMap invents a high-resolution waveform *consistent with* the ripped 16-bit stream under an SBM-like model — it does **not** recover the original studio master.

**Lossy repair** improves many poor encodes for casual listening. It does **not** recreate a discarded studio master or a true CD/studio FLAC.

For archival purity, rip with expand **off** (`--no-expand` / uncheck in the GUI) and leave lossy repair disabled.

## Features

- Multi-pass CD-DA rip via Windows SPTI
- **File upconvert**: FLAC, WAV, Ogg Vorbis (soundfile) + ffmpeg fallback
- **Lossy repair**: MP3/AAC/etc. artifact cleanup + bandwidth extension (`--repair-lossy` / GUI checkbox)
- MusicBrainz metadata when a valid disc ID is available
- CRC32 + AccurateRip-style fingerprints in log and tags
- Expand path: 16+16→32 then SBM quantize to **20** or **24** bit
- Optional ML residual upscaler (`--ml-upscaler`)
- `DESCRIPTION` + `COMMENT` tags describing the process
- Optional 16-bit FLAC sidecar
- CLI and PySide6 GUI

## Install

```powershell
cd ~/Projects/supermap-cd-flac
python -m pip install -e ".[dev]"
# optional torch ML backend:
python -m pip install -e ".[ml]"
# ffmpeg recommended for non-native formats / forced decode
```

## CLI

```powershell
supermap-cd drives
supermap-cd toc --drive D:\

# Bit-perfect 16-bit
supermap-cd rip --drive D:\ --output .\rips --no-expand

# Expand to 24-bit FLAC
supermap-cd rip --drive D:\ --output .\rips --bits 24

# Expand to 20-bit (packed in 24-bit FLAC) + ML upscaler + 16-bit sidecar
supermap-cd rip --drive D:\ --output .\rips --bits 20 --ml-upscaler --keep-16bit

# Upconvert existing 16-bit/44.1 files (FLAC, WAV, Ogg, or ffmpeg-decodable)
supermap-cd upconvert .\track.flac --bits 24 -o .\rips
supermap-cd upconvert .\album_dir -r --bits 20 --ml-upscaler
supermap-cd upconvert .\song.mp3 --ffmpeg --bits 24 -o .\rips

# Repair lossy MP3/AAC then write 24-bit FLAC
supermap-cd repair .\song.mp3 --bits 24 -o .\rips
supermap-cd upconvert .\song.mp3 --repair-lossy --repair-strength strong --bits 24 -o .\rips
```

## GUI

Double-click **`Launch SuperMap Converter.bat`** (dev install), run a frozen **SuperMap-Converter** build, or:

```powershell
supermap-cd gui
# or
supermap-cd-gui
```

The **Convert files** tab lets you add / drag-and-drop FLAC, WAV, Ogg, MP3, AAC (or other audio), enable **Repair lossy**, set 20/24-bit output, and convert. The **Rip CD** tab is shown on **Windows only** (SPTI optical access).

Each conversion writes a step-by-step `*.convert.log` next to the output FLAC (progress ends at **100%** once when the file is finished).

## Frozen executables (Windows / Linux / macOS)

There is no single binary for all OSes. Build **per platform** (onedir) with PyInstaller. Default freezes **exclude torch**; use `pip install -e ".[ml]"` for the ML backend in a normal Python install. **ffmpeg** is not bundled — install it separately if you need non-native formats.

```powershell
python -m pip install -e ".[packaging]"
python packaging/build_executables.py
```

Artifacts land in `dist/supermap-cd/` (CLI) and `dist/SuperMap-Converter/` (GUI). CI builds all three OSes on version tags (`v*`) or via workflow dispatch (see `.github/workflows/build-executables.yml`).

## Tags

Expanded / repaired files get a `DESCRIPTION` (and matching `COMMENT` lines) such as:

> SuperMap lossy repair (medium): codec artifact cleanup + bandwidth extension … SuperMap SBM-style expand: synthesized +16 bits …

## Development tests

```powershell
pytest -q
```
