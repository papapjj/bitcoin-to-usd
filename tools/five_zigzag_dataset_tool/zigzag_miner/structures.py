from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
import json
import math
import yaml


@dataclass(frozen=True)
class LegRule:
    direction: int  # +1, -1, or 0 for either
    min_atr: float = 0.0
    max_atr: float = math.inf
    min_bars: int = 1
    max_bars: int = 10**9
    min_ratio_prev: float = 0.0
    max_ratio_prev: float = math.inf

    @classmethod
    def from_dict(cls, d: dict) -> "LegRule":
        direction = d.get("direction", 0)
        if isinstance(direction, str):
            direction = {"up": 1, "down": -1, "any": 0}[direction.lower()]
        return cls(
            direction=int(direction),
            min_atr=float(d.get("min_atr", 0.0)),
            max_atr=float(d.get("max_atr", math.inf)),
            min_bars=int(d.get("min_bars", 1)),
            max_bars=int(d.get("max_bars", 10**9)),
            min_ratio_prev=float(d.get("min_ratio_prev", 0.0)),
            max_ratio_prev=float(d.get("max_ratio_prev", math.inf)),
        )


@dataclass(frozen=True)
class PatternSpec:
    name: str
    legs: Sequence[LegRule]
    history_bars: int = 256
    future_bars: int = 128
    min_gap_bars: int = 0

    @classmethod
    def from_dict(cls, d: dict) -> "PatternSpec":
        legs = [LegRule.from_dict(x) for x in d["legs"]]
        if not legs:
            raise ValueError("pattern must contain at least one leg")
        return cls(
            name=str(d.get("name", "unnamed_pattern")),
            legs=legs,
            history_bars=int(d.get("history_bars", 256)),
            future_bars=int(d.get("future_bars", 128)),
            min_gap_bars=int(d.get("min_gap_bars", 0)),
        )


def load_pattern(path: str | Path) -> PatternSpec:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return PatternSpec.from_dict(data)
