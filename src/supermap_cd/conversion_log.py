"""Per-conversion step log files written next to output FLACs."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


class ConversionLog:
    """Append-only step log for one audio conversion."""

    def __init__(self, path: Path, *, source: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._closed = False
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.path.write_text(
            f"SuperMap conversion log\n"
            f"Started: {stamp}\n"
            f"Source:  {source}\n"
            f"{'=' * 60}\n",
            encoding="utf-8",
        )

    def step(self, pct: int, message: str) -> None:
        if self._closed:
            return
        clamped = max(0, min(100, int(pct)))
        line = f"[{clamped:3d}%] {message}\n"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line)

    def complete(self, *, outputs: list[Path], ok: bool = True) -> None:
        if self._closed:
            return
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        status = "COMPLETED" if ok else "FAILED"
        with self.path.open("a", encoding="utf-8") as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"{status} at {stamp}\n")
            for p in outputs:
                f.write(f"Output: {p}\n")
            f.write("[100%] Conversion finished\n")
        self._closed = True
