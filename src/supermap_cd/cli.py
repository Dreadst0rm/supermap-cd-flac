"""Command-line interface for SuperMap CD."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .mb import lookup_disc, placeholder_meta
from .pipeline import RipOptions, process_rip_result, rip_and_encode_track, upconvert_file
from .rip import list_cd_drives, read_toc, rip_track_from_wav


def _progress(msg: str, frac: float) -> None:
    pct = int(max(0.0, min(1.0, frac)) * 100)
    print(f"\r[{pct:3d}%] {msg}          ", end="", flush=True)


def cmd_drives(_: argparse.Namespace) -> int:
    drives = list_cd_drives()
    if not drives:
        print("No CD-ROM drives detected.")
        return 1
    for d in drives:
        print(d)
    return 0


def cmd_toc(args: argparse.Namespace) -> int:
    toc = read_toc(args.drive)
    print(f"Drive: {toc.drive}")
    print(f"Lead-out LBA: {toc.leadout_lba}")
    for t in toc.tracks:
        kind = "audio" if t.is_audio else "data"
        print(
            f"  Track {t.number:02d}  LBA {t.start_lba:6d}  "
            f"{t.length_sectors:6d} sectors  ({t.duration_seconds:7.2f}s)  [{kind}]"
        )
    return 0


def _expand_options(args: argparse.Namespace, *, output: str | None = None) -> RipOptions:
    return RipOptions(
        output_dir=Path(output if output is not None else args.output),
        gap_fill=not args.no_expand,
        keep_16bit_sidecar=args.keep_16bit,
        flac_level=args.flac_level,
        passes=getattr(args, "passes", 2),
        output_bits=args.bits,
        ml_upscaler=args.ml_upscaler,
        repair_lossy=getattr(args, "repair_lossy", False),
        repair_strength=getattr(args, "repair_strength", "medium"),
    )


def _add_expand_args(p: argparse.ArgumentParser, *, include_passes: bool = False) -> None:
    p.add_argument("--output", "-o", default="rips", help="Output directory")
    p.add_argument("--no-expand", action="store_true", help="Bit-perfect 16-bit FLAC only")
    p.add_argument("--keep-16bit", action="store_true", help="Also write 16-bit sidecar when expanding")
    p.add_argument(
        "--bits",
        type=int,
        default=24,
        choices=(20, 24),
        help="Output bit depth after SBM quantize (20 left-aligned in 24-bit FLAC, or full 24)",
    )
    p.add_argument(
        "--ml-upscaler",
        action="store_true",
        help="Use ML residual upscaler as 32-bit preference (torch if installed, else spectral)",
    )
    p.add_argument(
        "--repair-lossy",
        action="store_true",
        help="Repair lossy MP3/AAC/etc. (artifact cleanup + bandwidth extension) before expand",
    )
    p.add_argument(
        "--repair-strength",
        choices=("light", "medium", "strong"),
        default="medium",
        help="Lossy repair strength (default: medium)",
    )
    p.add_argument("--flac-level", type=int, default=5, choices=range(0, 9))
    if include_passes:
        p.add_argument("--passes", type=int, default=2, help="Secure rip passes")


def cmd_rip(args: argparse.Namespace) -> int:
    options = _expand_options(args)

    if args.wav:
        from .mb import AlbumMeta, TrackMeta

        result = rip_track_from_wav(Path(args.wav), track_number=1)
        meta = AlbumMeta(
            discid="wav-source",
            album=args.album or "WAV Import",
            artist=args.artist or "Unknown Artist",
            tracks=[TrackMeta(number=1, title=args.title or Path(args.wav).stem)],
        )
        paths = process_rip_result(result, meta, options, progress=_progress)
        print()
        print(
            f"CRC32={result.crc32:08X}  AR_v1={result.accuraterip_v1:08X}  "
            f"AR_v2={result.accuraterip_v2:08X}"
        )
        for p in paths:
            print(f"Wrote {p}")
        return 0

    drive = args.drive or (list_cd_drives()[0] if list_cd_drives() else None)
    if not drive:
        print("No drive specified and none detected.", file=sys.stderr)
        return 1

    toc = read_toc(drive)
    try:
        meta = lookup_disc(toc)
    except Exception as exc:
        print(f"MusicBrainz lookup failed ({exc}); using placeholders")
        meta = placeholder_meta(toc)

    print(f"Album: {meta.artist} - {meta.album} (discid={meta.discid})")
    tracks = toc.toc_audio()
    if args.track is not None:
        tracks = [t for t in tracks if t.number == args.track]
        if not tracks:
            print(f"Track {args.track} not found", file=sys.stderr)
            return 1

    for t in tracks:
        print(f"\nRipping track {t.number}...")
        result, paths = rip_and_encode_track(drive, t, meta, options, progress=_progress)
        print()
        ar_note = f"passes_verified={result.verified_passes}"
        if result.notes:
            ar_note += f" notes={';'.join(result.notes)}"
        print(
            f"Track {t.number}: CRC32={result.crc32:08X}  "
            f"AccurateRipV1={result.accuraterip_v1:08X}  "
            f"AccurateRipV2={result.accuraterip_v2:08X}  ({ar_note})"
        )
        for p in paths:
            print(f"  Wrote {p}")
    return 0


def cmd_upconvert(args: argparse.Namespace) -> int:
    from .io_audio import collect_audio_inputs

    inputs = collect_audio_inputs([Path(p) for p in args.inputs], recursive=args.recursive)
    if not inputs:
        print("No audio files found.", file=sys.stderr)
        return 1

    options = _expand_options(args)
    mode = "repair+upconvert" if options.repair_lossy else "upconvert"
    print(f"{mode}: {len(inputs)} file(s) -> {options.output_dir}")
    failures = 0
    for i, src in enumerate(inputs, start=1):
        print(f"\n[{i}/{len(inputs)}] {src}")
        try:
            result, paths = upconvert_file(
                src,
                options,
                album=args.album,
                artist=args.artist,
                title=args.title if len(inputs) == 1 else None,
                force_ffmpeg=args.ffmpeg,
                progress=_progress,
            )
            print()
            print(
                f"CRC32={result.crc32:08X}  AR_v1={result.accuraterip_v1:08X}  "
                f"AR_v2={result.accuraterip_v2:08X}  notes={';'.join(result.notes)}"
            )
            for p in paths:
                print(f"  Wrote {p}")
        except Exception as exc:
            failures += 1
            print(f"FAILED: {exc}", file=sys.stderr)
    return 1 if failures else 0


def cmd_repair(args: argparse.Namespace) -> int:
    """Alias for upconvert with lossy repair enabled."""
    args.repair_lossy = True
    if not getattr(args, "bits", None):
        args.bits = 24
    return cmd_upconvert(args)


def cmd_gui(_: argparse.Namespace) -> int:
    from .gui import run_gui

    return run_gui()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="supermap-cd",
        description=(
            "Secure CD ripper / file upconverter with optional lossy repair "
            "and SuperMap SBM-style expand (16+16->32) then quantize to 20/24-bit FLAC"
        ),
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("drives", help="List CD-ROM drives")
    d.set_defaults(func=cmd_drives)

    t = sub.add_parser("toc", help="Show disc TOC")
    t.add_argument("--drive", required=True, help=r"Drive root, e.g. D:\ ")
    t.set_defaults(func=cmd_toc)

    r = sub.add_parser("rip", help="Rip disc (or legacy --wav) to FLAC")
    r.add_argument("--drive", help=r"Drive root, e.g. D:\ ")
    _add_expand_args(r, include_passes=True)
    r.add_argument("--track", type=int, help="Rip only this track number")
    r.add_argument(
        "--wav",
        help="Deprecated: use 'upconvert' instead. Offline expand of a 16-bit/44.1 file",
    )
    r.add_argument("--album", help="Album tag for --wav mode")
    r.add_argument("--artist", help="Artist tag for --wav mode")
    r.add_argument("--title", help="Title tag for --wav mode")
    r.set_defaults(func=cmd_rip)

    u = sub.add_parser(
        "upconvert",
        help="Upconvert audio to FLAC (optional lossy repair + SBM expand)",
    )
    u.add_argument(
        "inputs",
        nargs="+",
        help="Audio file(s) and/or directories",
    )
    _add_expand_args(u, include_passes=False)
    u.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Recurse into directories",
    )
    u.add_argument(
        "--ffmpeg",
        action="store_true",
        help="Force decode via ffmpeg (s16le stereo @ 44.1 kHz)",
    )
    u.add_argument("--album", help="Override album tag")
    u.add_argument("--artist", help="Override artist tag")
    u.add_argument("--title", help="Override title tag (single-file only)")
    u.set_defaults(func=cmd_upconvert)

    rep = sub.add_parser(
        "repair",
        help="Repair lossy MP3/AAC/etc. then write 24-bit FLAC (same as upconvert --repair-lossy)",
    )
    rep.add_argument("inputs", nargs="+", help="Lossy audio file(s) and/or directories")
    _add_expand_args(rep, include_passes=False)
    rep.add_argument("--recursive", "-r", action="store_true")
    rep.add_argument("--ffmpeg", action="store_true")
    rep.add_argument("--album", help="Override album tag")
    rep.add_argument("--artist", help="Override artist tag")
    rep.add_argument("--title", help="Override title tag (single-file only)")
    rep.set_defaults(func=cmd_repair, repair_lossy=True)

    g = sub.add_parser("gui", help="Launch the desktop UI")
    g.set_defaults(func=cmd_gui)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
