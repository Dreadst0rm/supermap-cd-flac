"""FLAC encoding and Vorbis comment tagging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from mutagen.flac import FLAC


def write_flac(
    path: Path | str,
    pcm: np.ndarray,
    *,
    sample_rate: int = 44100,
    subtype: str = "PCM_16",
    compression_level: int = 5,
    tags: dict[str, Any] | None = None,
) -> Path:
    """Write PCM to FLAC. pcm is int16 or int32, shape (n,) or (n, ch).

    compression_level uses the common FLAC 0-8 scale; mapped to soundfile's [0, 1].

    For PCM_24, integer samples are treated as signed 24-bit values in the low
    24 bits of int32 (as produced by gap_fill). libsndfile expects those bits
    left-aligned in the int32 word, so we convert via float to avoid a silent
    / near-silent file.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(pcm)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    if subtype == "PCM_24" and np.issubdtype(arr.dtype, np.integer):
        # Normalize signed 24-bit magnitudes to float [-1, 1) for a correct write.
        arr = np.clip(arr.astype(np.float64) / float(1 << 23), -1.0, 1.0 - 1.0 / (1 << 23))

    sf_level = max(0.0, min(1.0, float(compression_level) / 8.0))
    sf.write(
        str(out),
        arr,
        sample_rate,
        format="FLAC",
        subtype=subtype,
        compression_level=sf_level,
    )

    if tags:
        apply_tags(out, tags)
    return out


def apply_tags(path: Path | str, tags: dict[str, Any]) -> None:
    audio = FLAC(str(path))
    for key, value in tags.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            audio[key] = [str(v) for v in value]
        else:
            audio[key] = [str(value)]
    audio.save()


def tags_for_track(
    *,
    album: str | None = None,
    artist: str | None = None,
    title: str | None = None,
    track_number: int | None = None,
    track_total: int | None = None,
    musicbrainz_albumid: str | None = None,
    musicbrainz_trackid: str | None = None,
    musicbrainz_discid: str | None = None,
    gap_fill: bool = False,
    output_bits: int | None = None,
    ml_upscaler: bool = False,
    ml_backend: str | None = None,
    repair_lossy: bool = False,
    repair_strength: str | None = None,
    description: str | None = None,
    crc32: str | None = None,
    ar_v1: str | None = None,
    ar_v2: str | None = None,
    extra_comment: str | None = None,
) -> dict[str, Any]:
    tags: dict[str, Any] = {}
    if album:
        tags["ALBUM"] = album
    if artist:
        tags["ARTIST"] = artist
        tags["ALBUMARTIST"] = artist
    if title:
        tags["TITLE"] = title
    if track_number is not None:
        tags["TRACKNUMBER"] = str(track_number)
    if track_total is not None:
        tags["TRACKTOTAL"] = str(track_total)
    if musicbrainz_albumid:
        tags["MUSICBRAINZ_ALBUMID"] = musicbrainz_albumid
    if musicbrainz_trackid:
        tags["MUSICBRAINZ_TRACKID"] = musicbrainz_trackid
    if musicbrainz_discid:
        tags["MUSICBRAINZ_DISCID"] = musicbrainz_discid

    if description:
        tags["DESCRIPTION"] = description

    comments: list[str] = []
    if repair_lossy:
        comments.append(
            f"SuperMap lossy repair ({repair_strength or 'medium'})"
        )
    if gap_fill:
        bits = output_bits or 24
        comments.append(f"SuperMap SBM expand 16+16->32 then quantize to {bits}-bit")
        if ml_upscaler:
            comments.append(f"ML upscaler={ml_backend or 'enabled'}")
    elif not repair_lossy:
        comments.append("SuperMap bit-perfect 16-bit rip")
    if description:
        comments.append(description)
    if crc32:
        comments.append(f"CRC32={crc32}")
    if ar_v1:
        comments.append(f"AccurateRipV1={ar_v1}")
    if ar_v2:
        comments.append(f"AccurateRipV2={ar_v2}")
    if extra_comment:
        comments.append(extra_comment)
    tags["COMMENT"] = comments
    return tags
