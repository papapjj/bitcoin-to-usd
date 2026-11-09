"""
src/self_growing_trader/progress_reader.py

Real-time progress-tracking binary file reader for large model checkpoints.
Eliminates silent console hangs when unpickling multi-gigabyte models.
"""

from __future__ import annotations

import hashlib
import time
from typing import BinaryIO, Optional


class ProgressFileReader:
    """Wraps a binary file stream to render live unpickling progress in the console."""

    def __init__(
        self,
        stream: BinaryIO,
        total_bytes: int,
        label: str = "Loading",
        min_interval: float = 0.25,
        track_sha256: bool = False,
    ) -> None:
        self._stream = stream
        self._total_bytes = max(1, int(total_bytes))
        self._label = label
        self._bytes_read = 0
        self._last_report = 0.0
        self._min_interval = min_interval
        self._hasher = hashlib.sha256() if track_sha256 else None

    def read(self, size: int = -1) -> bytes:
        chunk = self._stream.read(size)
        if chunk:
            self._bytes_read += len(chunk)
            if self._hasher is not None:
                self._hasher.update(chunk)
            self._report()
        return chunk

    def readinto(self, b) -> int:
        n = getattr(self._stream, "readinto", None)
        if n is not None:
            res = n(b)
            if res:
                self._bytes_read += res
                if self._hasher is not None:
                    self._hasher.update(memoryview(b)[:res])
                self._report()
            return res
        chunk = self.read(len(b))
        b[:len(chunk)] = chunk
        return len(chunk)

    def read1(self, size: int = -1) -> bytes:
        r1 = getattr(self._stream, "read1", self._stream.read)
        chunk = r1(size)
        if chunk:
            self._bytes_read += len(chunk)
            if self._hasher is not None:
                self._hasher.update(chunk)
            self._report()
        return chunk

    def readline(self, size: int = -1) -> bytes:
        line = self._stream.readline(size)
        if line:
            self._bytes_read += len(line)
            if self._hasher is not None:
                self._hasher.update(line)
            self._report()
        return line

    def seek(self, offset: int, whence: int = 0) -> int:
        res = self._stream.seek(offset, whence)
        self._bytes_read = self._stream.tell()
        self._report()
        return res

    def tell(self) -> int:
        return self._stream.tell()

    @property
    def sha256_hexdigest(self) -> str:
        return self._hasher.hexdigest().upper() if self._hasher is not None else ""

    def _report(self) -> None:
        now = time.monotonic()
        if now - self._last_report >= self._min_interval or self._bytes_read >= self._total_bytes:
            self._last_report = now
            pct = min(100.0, (self._bytes_read / self._total_bytes) * 100.0)
            mb_read = self._bytes_read / (1024 * 1024)
            mb_total = self._total_bytes / (1024 * 1024)
            width = 25
            filled = int(width * pct / 100.0)
            bar = "=" * filled + (">" if filled < width else "") + " " * (width - filled - (1 if filled < width else 0))
            print(
                f"\r      -> {self._label}: [{bar}] {mb_read:6.1f}/{mb_total:6.1f} MB ({pct:5.1f}%)",
                end="",
                flush=True,
            )

    def finish(self) -> None:
        mb_total = self._total_bytes / (1024 * 1024)
        print(
            f"\r      -> {self._label}: [=========================] {mb_total:6.1f}/{mb_total:6.1f} MB (100.0%) [OK]    ",
            flush=True,
        )

    def __getattr__(self, name: str):
        return getattr(self._stream, name)

