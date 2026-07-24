"""Repair lossy-encoded audio (MP3/AAC/etc.) toward a cleaner 24-bit FLAC listen.

This cannot restore a discarded studio master. It reduces common codec artifacts
and synthesizes plausible high-frequency content (bandwidth extension) so many
poor encodes sound closer to a clean lossless file for casual listening.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from scipy import signal

from .ml_upscaler import ml_available

SAMPLE_RATE = 44100
ProgressCb = Callable[[str, float], None]
Strength = Literal["light", "medium", "strong"]

LOSSY_SUFFIXES = {
    ".mp3",
    ".m4a",
    ".aac",
    ".wma",
    ".opus",
    ".ogg",
    ".oga",
}

STRENGTH_PARAMS = {
    # (artifact_suppress, bwe_gain, hf_tilt, stereo_widen)
    "light": (0.35, 0.45, 0.55, 0.08),
    "medium": (0.55, 0.75, 0.70, 0.14),
    "strong": (0.75, 1.00, 0.85, 0.22),
}


def is_likely_lossy(path_suffix: str) -> bool:
    return path_suffix.lower() in LOSSY_SUFFIXES


def repair_backend(*, prefer_torch: bool = True) -> str:
    if prefer_torch and ml_available():
        return "spectral+torch-refine"
    return "spectral-bwe"


def _estimate_cutoff_hz(freqs: np.ndarray, mag: np.ndarray) -> float:
    """Estimate codec bandwidth limit from average magnitude envelope."""
    env = np.maximum(mag.mean(axis=1), 1e-12)
    # Smooth in log-ish frequency bins
    kernel = np.ones(5) / 5.0
    smooth = np.convolve(env, kernel, mode="same")
    peak = float(np.max(smooth)) + 1e-12
    # Cutoff: last bin still within -40 dB of peak, scanning from high freqs down
    thresh = peak * (10 ** (-40 / 20))
    cutoff_bin = 1
    for i in range(len(smooth) - 2, 0, -1):
        if smooth[i] >= thresh and freqs[i] > 3000:
            cutoff_bin = i
            break
    # Typical MP3 walls ~15–18 kHz; clamp
    hz = float(freqs[cutoff_bin])
    return float(np.clip(hz, 8000.0, 19000.0))


def _suppress_codec_artifacts(
    z: np.ndarray,
    freqs: np.ndarray,
    cutoff_hz: float,
    strength: float,
) -> np.ndarray:
    """Soften unstable high-band bins typical of lossy coding."""
    mag = np.abs(z)
    phase = np.angle(z)
    # Temporal median-ish smooth in HF (mean of neighbors in time)
    hf = freqs >= max(6000.0, cutoff_hz * 0.75)
    if not np.any(hf):
        return z
    m = mag.copy()
    # 3-tap temporal average on HF bins
    left = np.roll(m, 1, axis=1)
    right = np.roll(m, -1, axis=1)
    avg = (left + m + right) / 3.0
    # Also damp bins with extreme frame-to-frame jumps
    jump = np.abs(m - avg) / (avg + 1e-9)
    damp = 1.0 / (1.0 + strength * 2.5 * jump)
    m[hf, :] = (1.0 - strength) * m[hf, :] + strength * avg[hf, :] * damp[hf, :]
    return m * np.exp(1j * phase)


def _bandwidth_extend(
    z: np.ndarray,
    freqs: np.ndarray,
    cutoff_hz: float,
    *,
    bwe_gain: float,
    hf_tilt: float,
) -> np.ndarray:
    """Mirror mid/high band into missing HF with decaying envelope."""
    mag = np.abs(z)
    phase = np.angle(z)
    n_freq, n_frames = mag.shape
    # Source band just below cutoff
    src_hi = int(np.searchsorted(freqs, cutoff_hz))
    src_hi = int(np.clip(src_hi, 8, n_freq - 2))
    src_lo = int(np.clip(src_hi - max(8, src_hi // 3), 1, src_hi - 1))
    src = mag[src_lo:src_hi, :]
    if src.size == 0:
        return z

    # Destination: above cutoff to Nyquist
    dst_lo = src_hi
    dst_hi = n_freq
    n_dst = dst_hi - dst_lo
    if n_dst <= 1:
        return z

    # Mirror source repeatedly to fill destination length
    mirrored = np.flipud(src)
    tiles = int(np.ceil(n_dst / mirrored.shape[0]))
    fill = np.tile(mirrored, (tiles, 1))[:n_dst, :]

    # Envelope decay toward Nyquist
    decay = np.linspace(1.0, max(0.05, 1.0 - hf_tilt), n_dst).reshape(-1, 1)
    # Match level at cutoff seam
    seam = np.maximum(mag[src_hi - 1 : src_hi, :], 1e-9)
    scale = (seam / np.maximum(fill[:1, :], 1e-9)) * bwe_gain
    fill = fill * scale * decay

    # Keep a little of any existing HF, blend in extension
    existing = mag[dst_lo:dst_hi, :]
    mag[dst_lo:dst_hi, :] = np.maximum(existing * 0.25, fill)

    # Phase: continue with randomized but stable HF phase from source mirror
    rng = np.random.default_rng(0xB10E)
    src_phase = phase[src_lo:src_hi, :]
    phase_fill = np.flipud(src_phase)
    phase_tiles = int(np.ceil(n_dst / max(phase_fill.shape[0], 1)))
    pfill = np.tile(phase_fill, (phase_tiles, 1))[:n_dst, :]
    pfill = pfill + rng.uniform(-0.35, 0.35, size=pfill.shape)
    # Preserve existing phase where energy already existed
    use_exist = existing > (fill * 0.5)
    phase[dst_lo:dst_hi, :] = np.where(use_exist, phase[dst_lo:dst_hi, :], pfill)

    return mag * np.exp(1j * phase)


def _stereo_widen(left: np.ndarray, right: np.ndarray, amount: float) -> tuple[np.ndarray, np.ndarray]:
    """Mild M/S widen to counteract joint-stereo collapse."""
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right)
    side = side * (1.0 + amount)
    l = mid + side
    r = mid - side
    peak = max(np.max(np.abs(l)), np.max(np.abs(r)), 1e-12)
    if peak > 0.98:
        g = 0.98 / peak
        l *= g
        r *= g
    return l, r


def _torch_hf_refine(x: np.ndarray) -> np.ndarray:
    """Optional light residual refine (reuses ML upscaler net scaled up)."""
    from .ml_upscaler import apply_ml_upscaler

    # apply_ml_upscaler adds sub-LSB residual; scale up for audible HF polish
    y = apply_ml_upscaler(x, prefer_torch=True)
    resid = y - x
    return np.clip(x + resid * 120.0, -1.0, 1.0)


def repair_channel(
    x: np.ndarray,
    *,
    strength: Strength = "medium",
    use_torch: bool = True,
) -> np.ndarray:
    """Repair one float channel in [-1, 1]."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    art, bwe_gain, hf_tilt, _widen = STRENGTH_PARAMS[strength]

    nperseg = 2048
    noverlap = 1536
    freqs, _times, z = signal.stft(
        x, fs=SAMPLE_RATE, nperseg=nperseg, noverlap=noverlap, boundary="zeros"
    )
    cutoff = _estimate_cutoff_hz(freqs, np.abs(z))
    z = _suppress_codec_artifacts(z, freqs, cutoff, art)
    z = _bandwidth_extend(z, freqs, cutoff, bwe_gain=bwe_gain, hf_tilt=hf_tilt)
    _, y = signal.istft(z, fs=SAMPLE_RATE, nperseg=nperseg, noverlap=noverlap, boundary=True)
    if y.size < n:
        y = np.pad(y, (0, n - y.size))
    else:
        y = y[:n]

    # Keep most of the original body; blend repaired HF/detail
    blend = 0.35 + 0.25 * {"light": 0, "medium": 1, "strong": 2}[strength]
    # Highpass residual so we don't smear the midrange
    b, a = signal.butter(2, 5000 / (SAMPLE_RATE / 2), btype="high")
    resid = signal.filtfilt(b, a, y - x)
    out = np.clip(x + blend * resid, -1.0, 1.0)

    if use_torch and ml_available():
        out = _torch_hf_refine(out)
    return out


def repair_lossy_pcm(
    pcm: np.ndarray,
    *,
    strength: Strength = "medium",
    use_torch: bool = True,
    progress: ProgressCb | None = None,
) -> np.ndarray:
    """Repair int16 stereo/mono PCM; returns int16 same shape."""
    if strength not in STRENGTH_PARAMS:
        raise ValueError(f"strength must be one of {tuple(STRENGTH_PARAMS)}")

    arr = np.asarray(pcm)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    float_x = arr.astype(np.float64) / 32768.0
    n_ch = float_x.shape[1]
    outs = []
    _, _, _, widen = STRENGTH_PARAMS[strength]

    for c in range(n_ch):
        if progress:
            progress(f"lossy repair ch{c + 1}/{n_ch}", 0.15 + 0.2 * (c / max(n_ch, 1)))
        outs.append(
            repair_channel(float_x[:, c], strength=strength, use_torch=use_torch)
        )

    if n_ch >= 2 and widen > 0:
        outs[0], outs[1] = _stereo_widen(outs[0], outs[1], widen)

    stacked = np.column_stack(outs) if n_ch > 1 else outs[0].reshape(-1, 1)
    # Match original channel count / layout
    if pcm.ndim == 1:
        stacked = stacked[:, 0]
    elif stacked.shape[1] > pcm.shape[1]:
        stacked = stacked[:, : pcm.shape[1]]

    peak = np.max(np.abs(stacked)) + 1e-12
    if peak > 0.99:
        stacked = stacked * (0.99 / peak)

    out_i16 = np.round(np.clip(stacked, -1.0, 1.0 - 1.0 / 32768.0) * 32768.0).astype(
        np.int16
    )
    if progress:
        progress("lossy repair done", 0.38)
    return np.ascontiguousarray(out_i16)


def repair_description(strength: Strength, backend: str) -> str:
    return (
        f"SuperMap lossy repair ({strength}): codec artifact cleanup + bandwidth "
        f"extension ({backend}). Approximate enhancement from a lossy source — "
        f"not a studio master or CD-quality archive."
    )
