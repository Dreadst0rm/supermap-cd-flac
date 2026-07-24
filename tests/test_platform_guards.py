"""Platform gating for Windows-only optical CD commands."""

from __future__ import annotations

import sys

import pytest

from supermap_cd.cli import main
from supermap_cd.rip import _require_windows_spti, list_cd_drives


def test_list_cd_drives_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert list_cd_drives() == []


def test_require_windows_spti_rejects_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "darwin")
    with pytest.raises(OSError, match="only supported on Windows"):
        _require_windows_spti()


@pytest.mark.skipif(sys.platform == "win32", reason="exercises non-Windows CLI guard")
def test_cli_drives_rejected_off_windows(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["drives"]) == 1
    err = capsys.readouterr().err
    assert "only available on Windows" in err


def test_cli_optical_guard_via_monkeypatch(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    assert main(["toc", "--drive", "/dev/sr0"]) == 1
    assert "only available on Windows" in capsys.readouterr().err
