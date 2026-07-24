"""Tests for SuperMap expand / SBM quantize consistency."""

from __future__ import annotations

import numpy as np

from supermap_cd.encode import tags_for_track
from supermap_cd.gapfill import (
    consistency_error,
    expand_to_32bit_float,
    float_to_int20,
    gap_fill,
    gap_fill_channel,
    int16_to_float,
    int20_to_float,
    process_description,
    sbm_forward_quantize,
)
from supermap_cd.ml_upscaler import apply_ml_upscaler, backend_name, make_prefer_fn


def _tone(n: int = 4096, freq: float = 440.0, amp: float = 0.5) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / 44100.0
    return amp * np.sin(2 * np.pi * freq * t)


def test_sbm_forward_deterministic():
    x = _tone()
    a = sbm_forward_quantize(x, out_bits=16, rng=None, shaped=True)
    b = sbm_forward_quantize(x, out_bits=16, rng=None, shaped=True)
    assert np.array_equal(a, b)


def test_gap_fill_roundtrip_within_one_lsb():
    master = _tone(8192, freq=1000.0, amp=0.6)
    master20 = int20_to_float(float_to_int20(master))
    disc16 = sbm_forward_quantize(master20, out_bits=16, rng=None, shaped=True)

    filled = gap_fill_channel(disc16, iterations=28)
    err = consistency_error(disc16, filled)
    assert err <= 1, f"consistency error {err} LSB"


def test_expand_32_then_pack_20_and_24():
    left = sbm_forward_quantize(_tone(2048), out_bits=16, rng=None, shaped=True)
    right = sbm_forward_quantize(_tone(2048, freq=880.0), out_bits=16, rng=None, shaped=True)
    stereo = np.column_stack([left, right])

    out20 = gap_fill(stereo, output_bits=20)
    out24 = gap_fill(stereo, output_bits=24)
    assert out20.shape == stereo.shape == out24.shape
    assert out20.dtype == out24.dtype == np.int32
    # 20-bit left-aligned in 24 -> low 4 bits zero
    assert np.all((out20 & 0xF) == 0)


def test_ml_prefer_stays_consistent():
    pcm = sbm_forward_quantize(_tone(4096), out_bits=16, rng=None, shaped=True)
    prefer = make_prefer_fn(use_torch=False)
    filled = expand_to_32bit_float(pcm, prefer_fn=prefer, iterations=12)
    assert consistency_error(pcm, filled) <= 1
    assert "spectral" in backend_name(prefer_torch=False)


def test_description_and_tags():
    desc = process_description(gap_fill_enabled=True, output_bits=20, ml_upscaler=True)
    assert "20-bit" in desc
    assert "ML" in desc
    tags = tags_for_track(
        title="T",
        gap_fill=True,
        output_bits=20,
        ml_upscaler=True,
        ml_backend="numpy-spectral-stft",
        description=desc,
    )
    assert "DESCRIPTION" in tags
    assert tags["DESCRIPTION"] == desc
    assert any("20-bit" in c for c in tags["COMMENT"])


def test_gap_fill_off_embedding_matches_centers():
    pcm = sbm_forward_quantize(_tone(1024, amp=0.25), out_bits=16, rng=None, shaped=True)
    back = np.round(int16_to_float(pcm) * 32768.0).astype(np.int16)
    assert np.array_equal(pcm, back)


def test_encode_flac_bitperfect_16(tmp_path):
    from supermap_cd.encode import write_flac
    import soundfile as sf

    pcm = sbm_forward_quantize(_tone(4096), out_bits=16, rng=None, shaped=True)
    stereo = np.column_stack([pcm, pcm])
    path = tmp_path / "t.flac"
    write_flac(
        path,
        stereo,
        subtype="PCM_16",
        tags={"TITLE": "Test", "DESCRIPTION": "bit-perfect", "COMMENT": ["bit-perfect"]},
    )
    data, sr = sf.read(str(path), dtype="int16", always_2d=True)
    assert sr == 44100
    assert np.array_equal(data, stereo)
    from mutagen.flac import FLAC

    meta = FLAC(str(path))
    assert meta["DESCRIPTION"] == ["bit-perfect"]


def test_apply_ml_upscaler_numpy():
    x = int16_to_float(sbm_forward_quantize(_tone(2048), out_bits=16, rng=None, shaped=True))
    y = apply_ml_upscaler(x, prefer_torch=False)
    assert y.shape == x.shape
    assert np.max(np.abs(y - x)) < 1.0 / 32768.0
