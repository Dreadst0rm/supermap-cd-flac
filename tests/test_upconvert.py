"""Tests for file upconvert / io_audio loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from supermap_cd.cli import main
from supermap_cd.io_audio import collect_audio_inputs, load_as_rip_result, load_pcm16_44100
from supermap_cd.pipeline import RipOptions, upconvert_file


def _write_tone(path: Path, *, subtype: str = "PCM_16", channels: int = 2) -> Path:
    t = np.arange(4410, dtype=np.float64) / 44100.0
    tone = (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float64)
    if channels == 2:
        data = np.column_stack([tone, tone])
    else:
        data = tone
    if subtype == "PCM_16":
        pcm = np.round(data * 32767.0).astype(np.int16)
        sf.write(str(path), pcm, 44100, subtype="PCM_16")
    else:
        sf.write(str(path), data, 44100, format="OGG", subtype="VORBIS")
    return path


def test_load_wav_and_flac(tmp_path: Path):
    wav = _write_tone(tmp_path / "a.wav")
    flac = tmp_path / "a.flac"
    data, sr = sf.read(str(wav), dtype="int16", always_2d=True)
    sf.write(str(flac), data, sr, subtype="PCM_16")

    pcm_w, meta_w = load_pcm16_44100(wav)
    pcm_f, meta_f = load_pcm16_44100(flac)
    assert pcm_w.shape[1] == 2
    assert pcm_f.dtype == np.int16
    assert "soundfile" in meta_w.backend
    assert "soundfile" in meta_f.backend
    assert np.array_equal(pcm_w, data)


def test_load_ogg_vorbis(tmp_path: Path):
    ogg = _write_tone(tmp_path / "a.ogg", subtype="VORBIS")
    pcm, meta = load_pcm16_44100(ogg)
    assert pcm.shape[1] == 2
    assert pcm.dtype == np.int16
    assert meta.backend.startswith("soundfile") or meta.backend.startswith("ffmpeg")


def test_upconvert_file_writes_24bit_flac(tmp_path: Path):
    wav = _write_tone(tmp_path / "in.wav")
    out = tmp_path / "out"
    options = RipOptions(output_dir=out, gap_fill=True, output_bits=24, ml_upscaler=False)
    result, paths = upconvert_file(wav, options)
    assert result.pcm.dtype == np.int16
    assert len(paths) == 1
    assert paths[0].suffix == ".flac"
    data, sr = sf.read(str(paths[0]), dtype="int32", always_2d=True)
    assert sr == 44100
    assert data.shape[1] == 2


def test_collect_audio_inputs(tmp_path: Path):
    _write_tone(tmp_path / "one.wav")
    _write_tone(tmp_path / "two.flac")
    nested = tmp_path / "sub"
    nested.mkdir()
    _write_tone(nested / "three.ogg", subtype="VORBIS")
    flat = collect_audio_inputs([tmp_path], recursive=False)
    assert len(flat) == 2
    deep = collect_audio_inputs([tmp_path], recursive=True)
    assert len(deep) == 3


def test_cli_upconvert(tmp_path: Path):
    wav = _write_tone(tmp_path / "cli.wav")
    out = tmp_path / "rips"
    rc = main(["upconvert", str(wav), "-o", str(out), "--bits", "20"])
    assert rc == 0
    flacs = list(out.rglob("*.flac"))
    assert len(flacs) == 1
    logs = list(out.rglob("*.convert.log"))
    assert len(logs) == 1
    text = logs[0].read_text(encoding="utf-8")
    assert "[100%]" in text
    assert "COMPLETED" in text


def test_load_as_rip_result_notes(tmp_path: Path):
    wav = _write_tone(tmp_path / "n.wav")
    result, meta = load_as_rip_result(wav)
    assert result.verified_passes == 1
    assert meta.backend
    assert result.pcm.shape[0] > 0


@pytest.mark.skipif(
    __import__("shutil").which("ffmpeg") is None,
    reason="ffmpeg not on PATH",
)
def test_ffmpeg_force_decode(tmp_path: Path):
    wav = _write_tone(tmp_path / "ff.wav")
    pcm, meta = load_pcm16_44100(wav, force_ffmpeg=True)
    assert pcm.dtype == np.int16
    assert meta.backend.startswith("ffmpeg")
