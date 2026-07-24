"""Tests for AccurateRip CRC + multi-pass majority vote."""

from __future__ import annotations

import numpy as np

from supermap_cd.rip import (
    _majority_vote_bytes,
    accuraterip_crc_v1,
    accuraterip_crc_v2,
)


def _ar_v2_reference(pcm: np.ndarray) -> int:
    """Original stepwise Python accumulator (correctness oracle)."""
    data = np.ascontiguousarray(np.asarray(pcm, dtype=np.int16).reshape(-1))
    if data.size % 2:
        data = data[:-1]
    words = data.view(np.uint32)
    crc = 0
    mul = 1
    for w in words:
        crc = (crc + (int(w) * mul)) & 0xFFFFFFFF
        mul = (mul + 1) & 0xFFFFFFFF
    return crc


def test_accuraterip_v2_matches_reference():
    rng = np.random.default_rng(0)
    pcm = rng.integers(-32768, 32767, size=(8000, 2), dtype=np.int16)
    assert accuraterip_crc_v2(pcm) == _ar_v2_reference(pcm)


def test_accuraterip_v1_and_v2_agree_on_cd_length():
    # For tracks shorter than 2^32 frames the two fingerprints coincide.
    rng = np.random.default_rng(1)
    pcm = rng.integers(-32768, 32767, size=(4410, 2), dtype=np.int16)
    assert accuraterip_crc_v1(pcm) == accuraterip_crc_v2(pcm)


def test_majority_vote_three_pass_clear_majority():
    # Positions with a clear 2-of-3 winner
    a = bytearray([10, 20, 30, 40])
    b = bytearray([10, 99, 30, 41])
    c = bytearray([11, 20, 31, 41])
    out = _majority_vote_bytes([a, b, c])
    assert list(out) == [10, 20, 30, 41]


def test_majority_vote_three_pass_ties_prefer_first():
    # All distinct → earliest pass
    a = bytearray([1, 2])
    b = bytearray([3, 4])
    c = bytearray([5, 6])
    assert _majority_vote_bytes([a, b, c]) == a


def test_majority_vote_two_pass_prefers_first():
    a = bytearray(b"\x00\x01\x02\x03")
    b = bytearray(b"\x00\xff\x02\xfe")
    assert _majority_vote_bytes([a, b]) == a
