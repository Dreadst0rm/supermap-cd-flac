"""Shared rip -> optional expand -> FLAC pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .conversion_log import ConversionLog
from .encode import tags_for_track, write_flac
from .gapfill import gap_fill, process_description
from .mb import AlbumMeta
from .rip import RipResult, TrackInfo, rip_track

ProgressCb = Callable[[str, float], None]


@dataclass
class RipOptions:
    output_dir: Path
    gap_fill: bool = True
    keep_16bit_sidecar: bool = False
    flac_level: int = 5
    passes: int = 2
    output_bits: int = 24  # 20 or 24
    ml_upscaler: bool = False
    repair_lossy: bool = False
    repair_strength: str = "medium"  # light | medium | strong


def _safe_name(s: str) -> str:
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in s).strip().rstrip(".")
    return out or "Unknown"


def process_rip_result(
    result: RipResult,
    meta: AlbumMeta,
    options: RipOptions,
    *,
    progress: ProgressCb | None = None,
    conversion_log: ConversionLog | None = None,
) -> list[Path]:
    """Encode one ripped track to FLAC (and optional 16-bit sidecar)."""
    if options.output_bits not in (20, 24):
        raise ValueError("output_bits must be 20 or 24")
    if options.repair_strength not in ("light", "medium", "strong"):
        raise ValueError("repair_strength must be light, medium, or strong")

    options.output_dir.mkdir(parents=True, exist_ok=True)
    track = result.track
    title = meta.title_for(track.number)
    album_dir = options.output_dir / _safe_name(meta.artist) / _safe_name(meta.album)
    album_dir.mkdir(parents=True, exist_ok=True)

    base = f"{track.number:02d} - {_safe_name(title)}"
    written: list[Path] = []

    clog = conversion_log
    if clog is None:
        clog = ConversionLog(album_dir / f"{base}.convert.log", source=title)

    def prog(msg: str, frac: float) -> None:
        pct = int(max(0.0, min(1.0, frac)) * 100)
        clog.step(pct, msg)
        if progress:
            progress(msg, frac)

    clog.step(10, f"Prepare output dir: {album_dir}")
    clog.step(
        15,
        f"Options: bits={options.output_bits} expand={options.gap_fill} "
        f"ml={options.ml_upscaler} repair={options.repair_lossy}:{options.repair_strength}",
    )

    repair_backend_name = None
    if options.repair_lossy:
        from .lossy_repair import repair_backend, repair_lossy_pcm
        from .rip import accuraterip_crc_v1, accuraterip_crc_v2, pcm_crc32

        repair_backend_name = repair_backend(prefer_torch=True)
        prog(
            f"Track {track.number}: lossy repair ({options.repair_strength})",
            0.12,
        )
        clog.step(
            12,
            f"Lossy repair starting ({options.repair_strength}, {repair_backend_name})",
        )

        def _repair_progress(msg: str, frac: float) -> None:
            pct = int(max(12, min(38, frac * 100)))
            clog.step(pct, msg)
            if progress:
                progress(f"Track {track.number}: {msg}", frac)

        result.pcm = repair_lossy_pcm(
            result.pcm,
            strength=options.repair_strength,  # type: ignore[arg-type]
            use_torch=True,
            progress=_repair_progress,
        )
        result.crc32 = pcm_crc32(result.pcm)
        result.accuraterip_v1 = accuraterip_crc_v1(result.pcm)
        result.accuraterip_v2 = accuraterip_crc_v2(result.pcm)
        result.notes.append(
            f"lossy-repair:{options.repair_strength}:{repair_backend_name}"
        )
        clog.step(38, "Lossy repair complete")

    ml_backend = None
    prefer_fn = None
    if options.gap_fill and options.ml_upscaler:
        from .ml_upscaler import backend_name, make_prefer_fn

        prefer_fn = make_prefer_fn(use_torch=True)
        ml_backend = backend_name(prefer_torch=True)

    desc = process_description(
        gap_fill_enabled=options.gap_fill,
        output_bits=options.output_bits if options.gap_fill else None,
        ml_upscaler=bool(options.ml_upscaler and options.gap_fill),
    )
    if options.repair_lossy:
        from .lossy_repair import repair_description

        desc = (
            repair_description(
                options.repair_strength,  # type: ignore[arg-type]
                repair_backend_name or "spectral-bwe",
            )
            + " "
            + desc
        )
    if ml_backend and options.gap_fill:
        desc = f"{desc} Backend: {ml_backend}."

    crc_s = f"{result.crc32:08X}"
    ar1 = f"{result.accuraterip_v1:08X}"
    ar2 = f"{result.accuraterip_v2:08X}"

    filled = None
    if options.gap_fill:
        mode = f"ML:{ml_backend}" if ml_backend else "analytic"
        n_samp = int(result.pcm.shape[0])
        prog(
            f"Track {track.number}: expand 16+16->32 then SBM->{options.output_bits} ({mode})",
            0.40,
        )
        clog.step(40, f"SBM expand starting ({n_samp} samples/ch, {mode})")

        def _expand_progress(msg: str, frac: float) -> None:
            pct = int(max(40, min(68, frac * 100)))
            clog.step(pct, f"SBM expand: {msg}")
            if progress:
                progress(f"Track {track.number}: {msg}", frac)

        t0 = time.perf_counter()
        filled = gap_fill(
            result.pcm,
            output_bits=options.output_bits,
            prefer_fn=prefer_fn,
            progress=_expand_progress,
        )
        dt = time.perf_counter() - t0
        clog.step(50, f"SBM expand complete ({mode}) in {dt:.2f}s")

    if options.keep_16bit_sidecar or not options.gap_fill:
        prog(
            f"Track {track.number}: writing 16-bit FLAC",
            0.70 if options.gap_fill else 0.85,
        )
        path16 = (
            album_dir / f"{base}.flac"
            if not options.gap_fill
            else album_dir / f"{base}.16bit.flac"
        )
        write_flac(
            path16,
            result.pcm,
            subtype="PCM_16",
            compression_level=options.flac_level,
            tags=tags_for_track(
                album=meta.album,
                artist=meta.artist,
                title=title,
                track_number=track.number,
                track_total=len(meta.tracks) or None,
                musicbrainz_albumid=meta.release_id,
                musicbrainz_trackid=meta.recording_id_for(track.number),
                musicbrainz_discid=meta.discid,
                crc32=crc_s,
                ar_v1=ar1,
                ar_v2=ar2,
                gap_fill=False,
                repair_lossy=options.repair_lossy,
                repair_strength=options.repair_strength if options.repair_lossy else None,
                description=(
                    desc
                    if options.repair_lossy and not options.gap_fill
                    else process_description(
                        gap_fill_enabled=False, output_bits=None, ml_upscaler=False
                    )
                ),
            ),
        )
        written.append(path16)
        clog.step(75, f"Wrote 16-bit FLAC: {path16.name}")

    if options.gap_fill:
        assert filled is not None
        prog(f"Track {track.number}: writing {options.output_bits}-bit FLAC", 0.90)
        path_out = album_dir / f"{base}.flac"
        write_flac(
            path_out,
            filled.astype(np.int32),
            subtype="PCM_24",
            compression_level=options.flac_level,
            tags=tags_for_track(
                album=meta.album,
                artist=meta.artist,
                title=title,
                track_number=track.number,
                track_total=len(meta.tracks) or None,
                musicbrainz_albumid=meta.release_id,
                musicbrainz_trackid=meta.recording_id_for(track.number),
                musicbrainz_discid=meta.discid,
                crc32=crc_s,
                ar_v1=ar1,
                ar_v2=ar2,
                gap_fill=True,
                output_bits=options.output_bits,
                ml_upscaler=bool(options.ml_upscaler),
                ml_backend=ml_backend,
                repair_lossy=options.repair_lossy,
                repair_strength=options.repair_strength if options.repair_lossy else None,
                description=desc,
            ),
        )
        written.append(path_out)
        clog.step(95, f"Wrote {options.output_bits}-bit FLAC: {path_out.name}")

    prog(f"Track {track.number}: complete", 1.0)
    clog.complete(outputs=written, ok=True)
    return written


def rip_and_encode_track(
    drive: str,
    track: TrackInfo,
    meta: AlbumMeta,
    options: RipOptions,
    *,
    progress: ProgressCb | None = None,
) -> tuple[RipResult, list[Path]]:
    result = rip_track(drive, track, passes=options.passes, progress=progress)
    paths = process_rip_result(result, meta, options, progress=progress)
    return result, paths


def upconvert_file(
    path: Path,
    options: RipOptions,
    *,
    album: str | None = None,
    artist: str | None = None,
    title: str | None = None,
    track_number: int | None = None,
    force_ffmpeg: bool = False,
    progress: ProgressCb | None = None,
) -> tuple[RipResult, list[Path]]:
    """Load audio and run optional lossy repair + SBM expand pipeline."""
    from .io_audio import load_as_rip_result
    from .lossy_repair import is_likely_lossy
    from .mb import AlbumMeta, TrackMeta

    # Lossy sources decode more reliably via ffmpeg
    use_ffmpeg = force_ffmpeg or (
        options.repair_lossy and is_likely_lossy(path.suffix)
    )

    result, src = load_as_rip_result(
        path, track_number=track_number, force_ffmpeg=use_ffmpeg
    )
    number = result.track.number
    track_title = title or src.title or path.stem
    album_name = album or src.album or (
        "Lossy Repair" if options.repair_lossy else "File Upconvert"
    )
    meta = AlbumMeta(
        discid=f"file:{path.name}",
        album=album_name,
        artist=artist or src.artist or "Unknown Artist",
        tracks=[TrackMeta(number=number, title=track_title)],
    )

    album_dir = options.output_dir / _safe_name(meta.artist) / _safe_name(meta.album)
    album_dir.mkdir(parents=True, exist_ok=True)
    base = f"{number:02d} - {_safe_name(track_title)}"
    clog = ConversionLog(album_dir / f"{base}.convert.log", source=str(path))
    clog.step(5, f"Loaded via {src.backend} ({result.pcm.shape[0]} samples)")

    if progress:
        progress(f"Loaded {path.name} via {src.backend}", 0.05)
        clog.step(5, f"Loaded {path.name} via {src.backend}")

    paths = process_rip_result(
        result, meta, options, progress=progress, conversion_log=clog
    )
    return result, paths
