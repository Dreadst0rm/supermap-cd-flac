"""SBM-inspired expand (16+16 -> 32-bit internal) then quantize to 20/24-bit.

Super Bit Mapping is encode-only. We synthesize 16 extra bits inside each
16-bit quantization bin (32-bit working precision), then SBM-style
noise-shaped quantize down to a user-selected output depth (20 or 24).

Hot loops are Numba-accelerated when available (required for full-track speed).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import signal

SAMPLE_RATE = 44100
INT16_SCALE = 32768.0
OutputBits = int  # 20 or 24
PreferFn = Callable[[np.ndarray], np.ndarray]
ProgressCb = Callable[[str, float], None]

try:
    from numba import njit

    _HAS_NUMBA = True
except ImportError:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*_a, **_k):  # type: ignore[misc]
        def deco(fn):
            return fn

        return deco


def sbm_noise_shaping_filter(order: int = 3) -> np.ndarray:
    """FIR residual filter that pushes quantization error toward high bands."""
    if order == 1:
        return np.array([1.0, -0.85], dtype=np.float64)
    if order == 2:
        return np.array([1.0, -1.6, 0.7], dtype=np.float64)
    return np.array([1.0, -2.0, 1.5, -0.45], dtype=np.float64)


def bit_scale(bits: int) -> float:
    return float(1 << (bits - 1))


def float_to_int_bits(x: np.ndarray, bits: int) -> np.ndarray:
    scale = bit_scale(bits)
    max_q = (1 << (bits - 1)) - 1
    clipped = np.clip(x, -1.0, max_q / scale)
    return np.round(clipped * scale).astype(np.int32)


def int_bits_to_float(x: np.ndarray, bits: int) -> np.ndarray:
    return x.astype(np.float64) / bit_scale(bits)


def int16_to_float(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float64) / INT16_SCALE


def float_to_int16(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -1.0, 1.0 - 1.0 / INT16_SCALE)
    return np.round(clipped * INT16_SCALE).astype(np.int16)


def float_to_int20(x: np.ndarray) -> np.ndarray:
    return float_to_int_bits(x, 20)


def int20_to_float(x: np.ndarray) -> np.ndarray:
    return int_bits_to_float(x, 20)


def pack_bits_to_int24(samples: np.ndarray, bits: int) -> np.ndarray:
    """Pack signed `bits`-depth samples into a 24-bit PCM word (left-aligned)."""
    if bits not in (20, 24):
        raise ValueError("bits must be 20 or 24")
    shift = 24 - bits
    return (samples.astype(np.int32) << shift).astype(np.int32)


@njit(cache=True)
def _project_sbm_numba(x_pref: np.ndarray, target16: np.ndarray) -> np.ndarray:
    """SBM-consistent projection (3rd-order shaping taps baked in)."""
    n = target16.shape[0]
    x = np.empty(n, dtype=np.float64)
    # taps[1:] for [1, -2, 1.5, -0.45] -> -2, 1.5, -0.45
    a1, a2, a3 = -2.0, 1.5, -0.45
    e0 = 0.0
    e1 = 0.0
    e2 = 0.0
    inv = 1.0 / 32768.0
    half_safe = 0.5 * inv * (1.0 - 1e-10)
    for i in range(n):
        shaped_err = a1 * e0 + a2 * e1 + a3 * e2
        t = int(target16[i])
        y_center = t * inv
        y_lo = y_center - half_safe
        y_hi = y_center + half_safe
        x_lo = y_lo - shaped_err
        x_hi = y_hi - shaped_err
        pref = x_pref[i]
        if pref < x_lo:
            xi = x_lo
        elif pref > x_hi:
            xi = x_hi
        else:
            xi = pref
        y = xi + shaped_err
        # mid-tread quantize to 16-bit
        y_c = y
        if y_c < -1.0:
            y_c = -1.0
        elif y_c > 1.0 - inv:
            y_c = 1.0 - inv
        q = int(np.floor(y_c * 32768.0 + 0.5))
        if q < -32768:
            q = -32768
        elif q > 32767:
            q = 32767
        if q != t:
            y = y_center
            xi = y - shaped_err
            q = t
        x[i] = xi
        e = y - (q * inv)
        e2 = e1
        e1 = e0
        e0 = e
    return x


@njit(cache=True)
def _forward_quantize_numba(x: np.ndarray, out_bits: int) -> np.ndarray:
    n = x.shape[0]
    out = np.empty(n, dtype=np.int32)
    a1, a2, a3 = -2.0, 1.5, -0.45
    e0 = 0.0
    e1 = 0.0
    e2 = 0.0
    scale = float(1 << (out_bits - 1))
    inv = 1.0 / scale
    qmin = -(1 << (out_bits - 1))
    qmax = (1 << (out_bits - 1)) - 1
    for i in range(n):
        shaped_err = a1 * e0 + a2 * e1 + a3 * e2
        y = x[i] + shaped_err
        y_c = y
        if y_c < -1.0:
            y_c = -1.0
        elif y_c > 1.0 - inv:
            y_c = 1.0 - inv
        q = int(np.floor(y_c * scale + 0.5))
        if q < qmin:
            q = qmin
        elif q > qmax:
            q = qmax
        out[i] = q
        e = y - (q * inv)
        e2 = e1
        e1 = e0
        e0 = e
    return out


def _quantize_sample(y: float, bits: int) -> int:
    scale = bit_scale(bits)
    lsb = 1.0 / scale
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    y_c = float(np.clip(y, -1.0, 1.0 - lsb))
    q = int(np.floor(y_c * scale + 0.5))
    return int(np.clip(q, qmin, qmax))


def sbm_forward_quantize(
    x_float: np.ndarray,
    *,
    out_bits: int = 16,
    rng: np.random.Generator | None = None,
    shaped: bool = True,
) -> np.ndarray:
    """Noise-shaped quantize float audio to signed integer at out_bits."""
    if out_bits < 8 or out_bits > 32:
        raise ValueError("out_bits out of range")
    x = np.ascontiguousarray(np.asarray(x_float, dtype=np.float64).ravel())
    if rng is None and shaped and out_bits in (16, 20, 24):
        out = _forward_quantize_numba(x, out_bits)
        return out.astype(np.int16) if out_bits == 16 else out

    # Slow path (dither / unusual bit depths)
    n = x.size
    out = np.empty(n, dtype=np.int32)
    taps = sbm_noise_shaping_filter()
    err_hist = np.zeros(len(taps) - 1, dtype=np.float64)
    scale = bit_scale(out_bits)
    lsb = 1.0 / scale
    a = taps[1:].copy()
    for i in range(n):
        shaped_err = float(a @ err_hist) if shaped else 0.0
        dither = (rng.random() - rng.random()) * lsb if rng is not None else 0.0
        y = float(x[i]) + shaped_err + dither
        q = _quantize_sample(y, out_bits)
        out[i] = q
        e = y - (q / scale)
        err_hist[1:] = err_hist[:-1]
        err_hist[0] = e
    if out_bits == 16:
        return out.astype(np.int16)
    return out


def _smoothness_step(x: np.ndarray, strength: float) -> np.ndarray:
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16.0
    smooth = signal.lfilter(kernel, [1.0], x)
    smooth = np.roll(smooth, -2)
    return (1.0 - strength) * x + strength * smooth


def _project_sbm_consistent(
    x_pref: np.ndarray,
    target16: np.ndarray,
    taps: np.ndarray | None = None,
) -> np.ndarray:
    """Nearest SBM-consistent float to x_pref (feasible width = one 16-bit LSB)."""
    target16 = np.ascontiguousarray(np.asarray(target16, dtype=np.int16).ravel())
    x_pref = np.ascontiguousarray(np.asarray(x_pref, dtype=np.float64).ravel())
    return _project_sbm_numba(x_pref, target16)


def expand_to_32bit_float(
    pcm16: np.ndarray,
    *,
    iterations: int = 3,
    smooth_strength: float = 0.2,
    prefer_fn: PreferFn | None = None,
) -> np.ndarray:
    """Synthesize 16 extra bits -> float approximating 32-bit PCM, CD-consistent."""
    target = np.asarray(pcm16, dtype=np.int16).ravel()
    x = int16_to_float(target)

    if prefer_fn is not None:
        x = prefer_fn(x)
        x = _project_sbm_consistent(x, target)

    for _ in range(max(1, iterations)):
        x = _smoothness_step(x, smooth_strength)
        x = _project_sbm_consistent(x, target)

    x = int_bits_to_float(float_to_int_bits(x, 32), 32)
    x = _project_sbm_consistent(x, target)
    return x


def gap_fill_channel(
    pcm16: np.ndarray,
    *,
    iterations: int = 3,
    smooth_strength: float = 0.2,
    prefer_fn: PreferFn | None = None,
) -> np.ndarray:
    """Backward-compatible alias: expand one channel to high-res float."""
    return expand_to_32bit_float(
        pcm16,
        iterations=iterations,
        smooth_strength=smooth_strength,
        prefer_fn=prefer_fn,
    )


def quantize_sbm_output(
    x_float: np.ndarray,
    target16: np.ndarray,
    bits: int,
) -> np.ndarray:
    """Project to CD-consistent float, then quantize to `bits`."""
    if bits not in (20, 24):
        raise ValueError("output bits must be 20 or 24")
    target16 = np.asarray(target16, dtype=np.int16).ravel()
    x = np.asarray(x_float, dtype=np.float64).ravel()
    xq = _project_sbm_consistent(x, target16)
    return float_to_int_bits(xq, bits)


def gap_fill(
    pcm16: np.ndarray,
    *,
    output_bits: int = 24,
    iterations: int = 3,
    prefer_fn: PreferFn | None = None,
    left_align_20_in_24: bool | None = None,
    progress: ProgressCb | None = None,
) -> np.ndarray:
    """Expand 16-bit PCM to int32 samples packed in a 24-bit container."""
    if left_align_20_in_24 is True:
        output_bits = 20
    elif left_align_20_in_24 is False and output_bits == 20:
        output_bits = 24

    if output_bits not in (20, 24):
        raise ValueError("output_bits must be 20 or 24")

    # Warm up Numba kernels once (avoids first-call stall mid-progress)
    if _HAS_NUMBA:
        _z = np.zeros(8, dtype=np.float64)
        _t = np.zeros(8, dtype=np.int16)
        _project_sbm_numba(_z, _t)
        _forward_quantize_numba(_z, 16)

    arr = np.asarray(pcm16)

    def _one(ch: np.ndarray, frac0: float, frac1: float) -> np.ndarray:
        if progress:
            progress("expand channel", frac0)
        hi = expand_to_32bit_float(ch, iterations=iterations, prefer_fn=prefer_fn)
        if progress:
            progress("quantize channel", (frac0 + frac1) * 0.5)
        q = quantize_sbm_output(hi, ch, output_bits)
        if progress:
            progress("pack channel", frac1)
        return pack_bits_to_int24(q, output_bits)

    if arr.ndim == 1:
        return _one(arr, 0.42, 0.68)
    if arr.ndim != 2:
        raise ValueError("pcm16 must be 1-D or 2-D (samples, channels)")
    chans = arr.shape[1]
    outs = []
    for c in range(chans):
        f0 = 0.42 + 0.26 * (c / max(chans, 1))
        f1 = 0.42 + 0.26 * ((c + 1) / max(chans, 1))
        outs.append(_one(arr[:, c], f0, f1))
    return np.column_stack(outs)


def process_description(
    *,
    gap_fill_enabled: bool,
    output_bits: int | None,
    ml_upscaler: bool,
) -> str:
    """Human-readable DESCRIPTION / COMMENT process blurb."""
    if not gap_fill_enabled:
        return (
            "SuperMap bit-perfect 16-bit CD rip. No expansion applied. "
            "Suitable for archival / AccurateRip-preserving storage."
        )
    bits = output_bits or 24
    ml = (
        " ML residual upscaler enabled as the 32-bit preference prior."
        if ml_upscaler
        else " Analytic SBM consistency projector (no ML)."
    )
    return (
        f"SuperMap SBM-style expand: synthesized +16 bits inside each 16-bit "
        f"quantization bin (32-bit working precision), then noise-shaped "
        f"quantize to {bits}-bit output packed in 24-bit FLAC."
        f"{ml} Approximate reconstruction - does not recover a lost studio master."
    )


def consistency_error(pcm16: np.ndarray, filled_float: np.ndarray) -> int:
    """Max absolute int16 mismatch after SBM forward quantize to 16-bit."""
    q = sbm_forward_quantize(filled_float, out_bits=16, rng=None, shaped=True)
    target = np.asarray(pcm16, dtype=np.int16).ravel().astype(np.int32)
    return int(np.max(np.abs(q.astype(np.int32) - target)))
