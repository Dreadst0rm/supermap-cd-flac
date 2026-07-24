"""Load 16-bit / 44.1 kHz PCM from files (soundfile + ffmpeg fallback)."""

from __future__ import annotations

import shutil
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .rip import (
    SAMPLES_PER_SECTOR,
    RipResult,
    TrackInfo,
    accuraterip_crc_v1,
    accuraterip_crc_v2,
    pcm_crc32,
)

SAMPLE_RATE = 44100

# Formats soundfile/libsndfile typically handles natively.
NATIVE_SUFFIXES = {".wav", ".flac", ".ogg", ".oga", ".aiff", ".aif", ".w64", ".rf64"}


@dataclass
class SourceMeta:
    """Tags/metadata pulled from an input audio file when available."""

    path: Path
    album: str | None = None
    artist: str | None = None
    title: str | None = None
    track_number: int | None = None
    track_total: int | None = None
    backend: str = "unknown"
    notes: list[str] = field(default_factory=list)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def read_file_tags(path: Path) -> SourceMeta:
    """Best-effort tag read via mutagen."""
    meta = SourceMeta(path=path)
    try:
        from mutagen import File as MutagenFile

        audio = MutagenFile(str(path), easy=True)
        if audio is None:
            return meta

        def _first(key: str) -> str | None:
            vals = audio.get(key)
            if not vals:
                return None
            return str(vals[0]).strip() or None

        meta.album = _first("album")
        meta.artist = _first("artist") or _first("albumartist")
        meta.title = _first("title")
        track = _first("tracknumber")
        if track:
            # Handle "3/12"
            part = track.split("/")[0].strip()
            if part.isdigit():
                meta.track_number = int(part)
            if "/" in track:
                total = track.split("/", 1)[1].strip()
                if total.isdigit():
                    meta.track_total = int(total)
    except Exception as exc:
        meta.notes.append(f"tag-read-skipped:{exc}")
    return meta


def _normalize_pcm(data: np.ndarray) -> np.ndarray:
    """Ensure int16 stereo (n, 2)."""
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.dtype != np.int16:
        # Float [-1,1] or wider int -> int16
        if np.issubdtype(arr.dtype, np.floating):
            clipped = np.clip(arr, -1.0, 1.0 - 1.0 / 32768.0)
            arr = np.round(clipped * 32768.0).astype(np.int16)
        else:
            # Assume already PCM-like integer; clip to int16 range
            arr = np.clip(arr, -32768, 32767).astype(np.int16)
    if arr.shape[1] > 2:
        arr = arr[:, :2]
    elif arr.shape[1] == 1:
        arr = np.column_stack([arr[:, 0], arr[:, 0]])
    return np.ascontiguousarray(arr)


def _load_soundfile(path: Path) -> tuple[np.ndarray, int, str]:
    import soundfile as sf

    with sf.SoundFile(str(path)) as f:
        subtype = (f.subtype or "").upper()
        fmt = f.format or "unknown"
        if "PCM_16" in subtype or subtype in {"PCM_S8", "PCM_U8"}:
            data = f.read(dtype="int16", always_2d=True)
        else:
            data = f.read(dtype="float64", always_2d=True)
        sr = int(f.samplerate)
    return _normalize_pcm(data), sr, f"soundfile:{fmt}/{subtype}"


def _load_ffmpeg(path: Path) -> tuple[np.ndarray, int, str]:
    if not ffmpeg_available():
        raise RuntimeError(
            "ffmpeg not found on PATH; install ffmpeg to decode this format "
            f"({path.suffix})"
        )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "2",
        "-ar",
        str(SAMPLE_RATE),
        "pipe:1",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None and proc.stderr is not None

    # Drain stderr concurrently so a full pipe cannot deadlock stdout reads.
    err_buf = bytearray()
    err_cap = 65536

    def _drain_stderr() -> None:
        assert proc.stderr is not None
        while True:
            chunk = proc.stderr.read(4096)
            if not chunk:
                break
            room = err_cap - len(err_buf)
            if room > 0:
                err_buf.extend(chunk[:room])

    err_thread = threading.Thread(target=_drain_stderr, daemon=True)
    err_thread.start()

    buf = bytearray()
    read_size = 1 << 20  # 1 MiB
    try:
        while True:
            chunk = proc.stdout.read(read_size)
            if not chunk:
                break
            buf.extend(chunk)
    finally:
        err_thread.join(timeout=60)
        ret = proc.wait()

    if ret != 0:
        err = err_buf.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed for {path}: {err or ret}")
    if len(buf) < 4 or len(buf) % 4:
        raise RuntimeError(f"ffmpeg produced empty/odd PCM for {path}")
    pcm = np.frombuffer(buf, dtype="<i2").reshape(-1, 2).copy()
    return pcm, SAMPLE_RATE, "ffmpeg:s16le@44100"


def load_pcm16_44100(
    path: Path | str,
    *,
    force_ffmpeg: bool = False,
    allow_resample_via_ffmpeg: bool = True,
) -> tuple[np.ndarray, SourceMeta]:
    """Load audio as int16 stereo @ 44.1 kHz.

    Tries soundfile for WAV/FLAC/Ogg (and similar), then ffmpeg for anything else
    or when sample rate is not 44.1 kHz and allow_resample_via_ffmpeg is True.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    meta = read_file_tags(path)
    suffix = path.suffix.lower()

    if force_ffmpeg or suffix not in NATIVE_SUFFIXES:
        pcm, sr, backend = _load_ffmpeg(path)
        meta.backend = backend
        meta.notes.append(backend)
        return pcm, meta

    try:
        pcm, sr, backend = _load_soundfile(path)
    except Exception as exc:
        # Fall through to ffmpeg
        meta.notes.append(f"soundfile-failed:{exc}")
        pcm, sr, backend = _load_ffmpeg(path)
        meta.backend = backend
        meta.notes.append(backend)
        return pcm, meta

    if sr != SAMPLE_RATE:
        if allow_resample_via_ffmpeg:
            meta.notes.append(f"resampled-from-{sr}-via-ffmpeg")
            pcm, sr, backend = _load_ffmpeg(path)
            meta.backend = backend
            meta.notes.append(backend)
            return pcm, meta
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr} for {path}")

    meta.backend = backend
    meta.notes.append(backend)
    return pcm, meta


def load_as_rip_result(
    path: Path | str,
    *,
    track_number: int | None = None,
    force_ffmpeg: bool = False,
) -> tuple[RipResult, SourceMeta]:
    """Load a file into a RipResult for the shared expand pipeline."""
    path = Path(path)
    pcm, meta = load_pcm16_44100(path, force_ffmpeg=force_ffmpeg)
    number = track_number or meta.track_number or 1
    track = TrackInfo(
        number=number,
        start_lba=0,
        length_sectors=max(1, pcm.shape[0] // SAMPLES_PER_SECTOR),
        control=0,
    )
    result = RipResult(
        track=track,
        pcm=pcm,
        crc32=pcm_crc32(pcm),
        accuraterip_v1=accuraterip_crc_v1(pcm),
        accuraterip_v2=accuraterip_crc_v2(pcm),
        verified_passes=1,
        notes=list(meta.notes),
    )
    return result, meta


def collect_audio_inputs(paths: list[Path], *, recursive: bool = False) -> list[Path]:
    """Expand files/dirs into a sorted list of audio paths."""
    found: list[Path] = []
    audio_suffixes = {
        ".wav",
        ".flac",
        ".ogg",
        ".oga",
        ".mp3",
        ".m4a",
        ".aac",
        ".wma",
        ".aiff",
        ".aif",
        ".opus",
        ".wv",
        ".ape",
    }
    for p in paths:
        p = Path(p)
        if p.is_file():
            found.append(p)
        elif p.is_dir():
            iterator = p.rglob("*") if recursive else p.glob("*")
            for candidate in iterator:
                if candidate.is_file() and candidate.suffix.lower() in audio_suffixes:
                    found.append(candidate)
        else:
            raise FileNotFoundError(p)
    # De-dupe preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for f in sorted(found, key=lambda x: str(x).lower()):
        key = f.resolve()
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out
