"""Optional ML residual upscaler for SuperMap 32-bit preference.

Builds a high-resolution preference signal from 16-bit-embedded float audio.
The result is always projected back into the SBM-feasible set by gapfill.

Backends:
  - torch (optional extra ``supermap-cd[ml]``): small 1-D residual CNN
  - numpy spectral fallback: STFT envelope residual when torch is absent

Long channels are processed in overlapping chunks to bound peak memory.
"""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np
from scipy import signal

SAMPLE_RATE = 44100
PreferFn = Callable[[np.ndarray], np.ndarray]

# Chunk sizes chosen so STFT/CNN work fits comfortably in RAM for album tracks.
_NP_CHUNK = 65536  # ~1.5 s
_NP_OVERLAP = 2048
_TORCH_CHUNK = 262144  # ~6 s
# 3x Conv1d k=9 → receptive field ~25; pad generously for OLA edges
_TORCH_OVERLAP = 64


def ml_available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


def _ola_process(
    x: np.ndarray,
    *,
    chunk: int,
    overlap: int,
    fn: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Overlap-add ``fn`` over ``x``; ``fn`` returns a same-length residual/signal."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    if n <= chunk:
        return fn(x)

    hop = max(1, chunk - overlap)
    out = np.zeros(n, dtype=np.float64)
    weight = np.zeros(n, dtype=np.float64)
    win = np.hanning(chunk).astype(np.float64)
    # Avoid zero endpoints killing the seam
    win = np.maximum(win, 1e-3)

    for start in range(0, n, hop):
        end = min(start + chunk, n)
        piece = x[start:end]
        length = piece.size
        if length < 8:
            break
        y = fn(piece)
        if y.size != length:
            y = y[:length] if y.size > length else np.pad(y, (0, length - y.size))
        w = win[:length] if length == chunk else np.hanning(length).astype(np.float64)
        w = np.maximum(w, 1e-3)
        out[start:end] += y * w
        weight[start:end] += w

    weight = np.maximum(weight, 1e-12)
    return out / weight


def _spectral_residual_chunk(x: np.ndarray) -> np.ndarray:
    """STFT residual for one chunk (returns residual only, not x+resid)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n = x.size
    nperseg = 512
    noverlap = 384
    _f, _t, z = signal.stft(x, fs=SAMPLE_RATE, nperseg=nperseg, noverlap=noverlap)
    mag = np.abs(z)
    freqs = _f.reshape(-1, 1)
    weight = np.clip((freqs - 2000.0) / 10000.0, 0.0, 1.0)
    low = np.maximum(mag[: max(1, mag.shape[0] // 4)].mean(axis=0, keepdims=True), 1e-9)
    high = mag[mag.shape[0] // 4 :].mean(axis=0, keepdims=True)
    ratio = float(np.mean(high / low) * float(np.mean(weight[mag.shape[0] // 4 :])))
    phase = np.angle(z)
    noise = np.random.default_rng(0).standard_normal(mag.shape)
    shaped = np.zeros_like(mag)
    hf = mag.shape[0] // 3
    shaped[hf:] = mag[hf:] * 0.02 * max(ratio, 0.0)
    shaped *= 0.5 + 0.5 * np.tanh(noise)
    z_res = shaped * np.exp(1j * phase)
    _, resid = signal.istft(z_res, fs=SAMPLE_RATE, nperseg=nperseg, noverlap=noverlap)
    if resid.size < n:
        resid = np.pad(resid, (0, n - resid.size))
    else:
        resid = resid[:n]
    lsb = 1.0 / 32768.0
    peak = np.max(np.abs(resid)) + 1e-12
    return resid * (0.35 * lsb / peak)


def _spectral_residual_prefer(x: np.ndarray) -> np.ndarray:
    """STFT-based residual preference (numpy fallback / always-available)."""
    x = np.asarray(x, dtype=np.float64).ravel()
    resid = _ola_process(
        x, chunk=_NP_CHUNK, overlap=_NP_OVERLAP, fn=_spectral_residual_chunk
    )
    return x + resid


class _TorchResidualNet:
    """Tiny residual CNN; deterministic init for reproducible preference."""

    def __init__(self) -> None:
        import torch
        from torch import nn

        class Net(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv1d(1, 16, kernel_size=9, padding=4),
                    nn.GELU(),
                    nn.Conv1d(16, 16, kernel_size=9, padding=4),
                    nn.GELU(),
                    nn.Conv1d(16, 1, kernel_size=9, padding=4),
                    nn.Tanh(),
                )
                torch.manual_seed(0x5B10D)
                for m in self.modules():
                    if isinstance(m, nn.Conv1d):
                        nn.init.normal_(m.weight, mean=0.0, std=0.02)
                        if m.bias is not None:
                            nn.init.zeros_(m.bias)

            def forward(self, x):  # type: ignore[no-untyped-def]
                return self.net(x) * (0.4 / 32768.0)

        self.torch = torch
        self.model = Net()
        self.model.eval()
        self._lock = threading.Lock()

    def _infer_f32(self, x32: np.ndarray) -> np.ndarray:
        """Run CNN on float32 mono; returns float32 residual."""
        torch = self.torch
        with self._lock, torch.no_grad():
            t = torch.from_numpy(np.ascontiguousarray(x32)).view(1, 1, -1)
            return self.model(t).view(-1).cpu().numpy()

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x64 = np.asarray(x, dtype=np.float64).ravel()
        x32 = x64.astype(np.float32, copy=False)

        def _chunk_resid(piece: np.ndarray) -> np.ndarray:
            p32 = np.asarray(piece, dtype=np.float32).ravel()
            r32 = self._infer_f32(p32)
            return r32.astype(np.float64, copy=False)

        if x32.size <= _TORCH_CHUNK:
            resid = _chunk_resid(x32)
        else:
            resid = _ola_process(
                x64,
                chunk=_TORCH_CHUNK,
                overlap=_TORCH_OVERLAP,
                fn=_chunk_resid,
            )
        return x64 + resid


_TORCH_NET: _TorchResidualNet | None = None


def make_prefer_fn(*, use_torch: bool = True) -> PreferFn:
    """Return a prefer_fn(x)->x for expand_to_32bit_float."""
    global _TORCH_NET
    if use_torch and ml_available():
        if _TORCH_NET is None:
            _TORCH_NET = _TorchResidualNet()
        net = _TORCH_NET

        def _prefer(x: np.ndarray) -> np.ndarray:
            return net(x)

        return _prefer

    def _prefer_np(x: np.ndarray) -> np.ndarray:
        return _spectral_residual_prefer(x)

    return _prefer_np


def apply_ml_upscaler(x_float: np.ndarray, *, prefer_torch: bool = True) -> np.ndarray:
    return make_prefer_fn(use_torch=prefer_torch)(x_float)


def backend_name(*, prefer_torch: bool = True) -> str:
    if prefer_torch and ml_available():
        return "torch-residual-cnn"
    return "numpy-spectral-stft"
