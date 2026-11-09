from __future__ import annotations

from pathlib import Path
import sys

def _configure_utf8_stdio() -> None:
    # Windows may inherit cp1252 from the parent UI. The tool emits JSON and
    # symbol descriptions that can contain arbitrary Unicode, so pin both
    # streams to UTF-8. errors=backslashreplace guarantees that a status line
    # can never abort a long MT5 download.
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="backslashreplace")
            except Exception:
                pass


_configure_utf8_stdio()

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from zigzag_miner.cli import main

if __name__ == "__main__":
    main()
