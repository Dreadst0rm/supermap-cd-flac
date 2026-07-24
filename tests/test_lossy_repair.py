"""Tests for lossy repair path."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from mutagen.flac import FLAC

from supermap_cd.cli import main
from supermap_cd.lossy_repair import (
    is_likely_lossy,
    repair_description,
    repair_lossy_pcm,
)
from supermap_cd.pipeline import RipOptions, upconvert_file


def _bandlimited_tone(n: int = 8820) -> np.ndarray:
    """Simulate a lossy-ish signal: tone + soft HF cutoff."""
    t = np.arange(n, dtype=np.float64) / 44100.0
    tone = 0.35 * np.sin(2 * np.pi * 880.0 * t)
    # Mild HF hash below ~12 kHz wall
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.02, size=n)
    # Simple FIR-ish lowpass via cumulative moving average on noise
    kernel = np.ones(9) / 9.0
    noise = np.convolve(noise, kernel, mode="same")
    x = np.clip(tone + noise, -0.95, 0.95)
    pcm = np.round(x * 32767.0).astype(np.int16)
    return np.column_stack([pcm, pcm])


def test_is_likely_lossy():
    assert is_likely_lossy(".mp3")
    assert is_likely_lossy(".M4A")
    assert is_likely_lossy(".aac")
    assert not is_likely_lossy(".flac")
    assert not is_likely_lossy(".wav")


def test_repair_lossy_pcm_shape_and_dtype():
    pcm = _bandlimited_tone()
    out = repair_lossy_pcm(pcm, strength="light", use_torch=False)
    assert out.shape == pcm.shape
    assert out.dtype == np.int16
    # Should change something (not a no-op on this signal)
    assert not np.array_equal(out, pcm)


def test_repair_description_honesty():
    text = repair_description("medium", "spectral-bwe")
    assert "lossy repair" in text.lower()
    assert "not a studio master" in text.lower()


def test_upconvert_with_repair_writes_tags(tmp_path: Path):
    wav = tmp_path / "lossy_sim.wav"
    sf.write(str(wav), _bandlimited_tone(), 44100, subtype="PCM_16")
    out = tmp_path / "out"
    options = RipOptions(
        output_dir=out,
        gap_fill=True,
        output_bits=24,
        repair_lossy=True,
        repair_strength="medium",
        ml_upscaler=False,
    )
    result, paths = upconvert_file(wav, options)
    assert len(paths) == 1
    assert any("lossy-repair" in n for n in result.notes)
    meta = FLAC(str(paths[0]))
    desc = " ".join(meta.get("DESCRIPTION", []))
    assert "lossy repair" in desc.lower()
    comments = " ".join(meta.get("COMMENT", []))
    assert "lossy repair" in comments.lower()
    log = list(out.rglob("*.convert.log"))
    assert log
    text = log[0].read_text(encoding="utf-8")
    assert "Lossy repair" in text or "lossy repair" in text.lower()
    assert "[100%]" in text


def test_cli_repair_command(tmp_path: Path):
    wav = tmp_path / "r.wav"
    sf.write(str(wav), _bandlimited_tone(4410), 44100, subtype="PCM_16")
    out = tmp_path / "rips"
    rc = main(
        [
            "repair",
            str(wav),
            "-o",
            str(out),
            "--bits",
            "24",
            "--repair-strength",
            "light",
            "--no-expand",
        ]
    )
    assert rc == 0
    flacs = list(out.rglob("*.flac"))
    assert len(flacs) == 1
    meta = FLAC(str(flacs[0]))
    desc = " ".join(meta.get("DESCRIPTION", []))
    assert "lossy repair" in desc.lower()
