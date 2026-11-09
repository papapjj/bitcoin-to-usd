from __future__ import annotations

import argparse
import json
from pathlib import Path

from self_growing_trader.model1_replay import Model1ReplayManifest, Model1TickDataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Pin and validate an MT5 model-1 tick capture")
    parser.add_argument("--ticks", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--manifest", default=None)
    args = parser.parse_args()
    ticks = Path(args.ticks)
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8-sig"))
    manifest_path = Path(args.manifest) if args.manifest else ticks.with_suffix(ticks.suffix + ".manifest.json")
    manifest = Model1ReplayManifest.from_capture_metadata(tick_path=ticks, metadata=metadata)
    manifest.validate()
    manifest.write(manifest_path)
    dataset = Model1TickDataset(ticks, manifest_path)
    print(json.dumps({
        "ticks": str(ticks.resolve()), "manifest": str(manifest_path.resolve()),
        "tick_count": dataset.manifest.tick_count, "sha256": dataset.manifest.tick_sha256,
        "tester_model": dataset.manifest.tester_model,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
