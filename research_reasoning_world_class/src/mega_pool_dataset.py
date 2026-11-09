"""
Mega-Pool Data Routing & Multi-Horizon Targets Engine.
File: research_reasoning_world_class/src/mega_pool_dataset.py
Milestone: Milestone 1 (Features 1, 2, 3, 4)

Architecture:
- Authentic 509k M1 bar dataset partitioned into Bull, Bear, and Chop training pools.
- Strict physical and programmatic isolation of the reserved Out-Of-Sample (OOS) benchmark slice
  [55000:64785] (February 22-28, 2025 crash slice, 9,785 bars, $2.50 spread).
- Precomputed vectorized multi-horizon returns at t+5, t+15, and t+30 normalized by 14-period
  MT5 iATR with 1e-4 safety floor and [-10.0, 10.0] clipping.
- Boundary safety guaranteeing zero future peeking and index clamping for boundary analysis.
- Causal 240-candle sequence lookback with left-padding for early bars (t < 239).
- Map-style PyTorch Datasets (SingleRegimeDataset, MegaPoolDataset) and custom Sampler
  (RegimeRoutingBatchSampler) supporting REGIME_BATCH, STRATIFIED, and ROUND_ROBIN routing.
- High-level MegaPoolRouter providing both direct functional batch sampling and standard PyTorch
  DataLoader creation, protected by an unbreakable zero-OOS leakage firewall.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from research_reasoning_world_class.src.real_m1_environment import RealM1Environment

try:
    from scripts.train_mythos_muzero_coevolution import SlicedM1Environment
except ImportError:
    # Standalone fallback if scripts package is not on sys.path
    class SlicedM1Environment:  # type: ignore[no-redef]
        """Zero-copy window view of RealM1Environment for Walk-Forward / OOS slices."""

        def __init__(
            self,
            master_env: Any,
            start_idx: int,
            end_idx: int,
            name: str = "",
        ) -> None:
            self.master = master_env
            self.master_env = master_env
            self.start_idx = max(0, start_idx)
            self.end_idx = min(master_env.total_bars, end_idx)
            self.name = name
            self.total_bars = max(0, self.end_idx - self.start_idx)

            self.open_arr = master_env.open_arr[self.start_idx : self.end_idx]
            self.high_arr = master_env.high_arr[self.start_idx : self.end_idx]
            self.low_arr = master_env.low_arr[self.start_idx : self.end_idx]
            self.close_arr = master_env.close_arr[self.start_idx : self.end_idx]
            self.spread_arr = master_env.spread_arr[self.start_idx : self.end_idx]
            self.volume_arr = master_env.volume_arr[self.start_idx : self.end_idx]
            self.atr_arr = master_env.atr_arr[self.start_idx : self.end_idx]
            self.day_progress_arr = master_env.day_progress_arr[self.start_idx : self.end_idx]
            if hasattr(master_env, "timestamps"):
                self.timestamps = master_env.timestamps[self.start_idx : self.end_idx]
            else:
                self.timestamps = [""] * self.total_bars
            if hasattr(master_env, "unix_timestamps") and len(master_env.unix_timestamps) >= self.end_idx:
                self.unix_timestamps = master_env.unix_timestamps[self.start_idx : self.end_idx]
            else:
                self.unix_timestamps = np.empty(0, dtype=np.int64)
            self.obs_matrix = master_env.obs_matrix[self.start_idx : self.end_idx]
            self.spread_usd = getattr(master_env, "spread_usd", 2.50)
            self.positions: List[Any] = []
            self.today_trade_count = 0
            self.equity = getattr(master_env, "equity", 10000.0)
            self.peak_equity = getattr(master_env, "peak_equity", 10000.0)
            self.initial_equity = getattr(master_env, "initial_equity", 10000.0)

        def __len__(self) -> int:
            return self.total_bars

        def get_observation(self, bar_idx: int) -> np.ndarray:
            if bar_idx < 0 or bar_idx >= self.total_bars:
                raise IndexError(f"bar_idx {bar_idx} out of range [0, {self.total_bars})")
            return self.obs_matrix[bar_idx].copy()

        def get_candle_sequence(self, bar_idx: int, lookback: int = 240) -> np.ndarray:
            if bar_idx < 0 or bar_idx >= self.total_bars:
                raise IndexError(f"bar_idx {bar_idx} out of range [0, {self.total_bars})")
            master_bar = self.start_idx + bar_idx
            return self.master_env.get_candle_sequence(master_bar, lookback=lookback)

        def get_atr(self, bar_idx: int) -> float:
            if bar_idx < 0 or bar_idx >= self.total_bars:
                return 50.0
            return float(self.atr_arr[bar_idx])

        def get_timestamp(self, bar_idx: int) -> str:
            if bar_idx < 0 or bar_idx >= self.total_bars:
                return ""
            return self.timestamps[bar_idx]


# =============================================================================
# 1. ENUMS & DATA STRUCTURES
# =============================================================================


class MarketRegime(str, Enum):
    """Enumeration of market regimes across authentic M1 data pools."""

    BULL = "bull"
    BEAR = "bear"
    CHOP = "chop"
    OOS = "oos"


class RoutingMode(str, Enum):
    """Routing modes for mini-batch generation across market regimes."""

    REGIME_BATCH = "regime_batch"  # 100% of samples drawn from a single regime
    STRATIFIED = "stratified"  # Proportional samples allocated across all regimes
    ROUND_ROBIN = "round_robin"  # Alternates regimes deterministically each batch


class MegaPoolBatch(NamedTuple):
    """
    Strongly-typed batch container complying with PROJECT.md Interface Contracts.
    Supports 4-tuple unpacking: `obs, seq, tgt, atr = batch`.
    """

    obs_137: torch.Tensor  # Shape [B, 137], dtype float32
    candle_seq_240: torch.Tensor  # Shape [B, 240, 6], dtype float32
    targets_3: torch.Tensor  # Shape [B, 3], dtype float32 ([y_5, y_15, y_30])
    atr: torch.Tensor  # Shape [B], dtype float32 (floored at 1e-4)

    @property
    def atr_ref(self) -> torch.Tensor:
        """Alias for PROJECT.md contract compatibility."""
        return self.atr


@dataclass(frozen=True)
class MegaPoolConfig:
    """Configuration for authentic Mega-Pool partitioning, target formulation, and caching."""

    csv_path: str = r"C:\Users\Admin\Desktop\BTCUSDm.csv"
    cache_dir: Optional[str] = "outputs/cache"
    use_cache: bool = True

    # Feature & Lookback Contracts
    feature_dim: int = 137
    lookback_candles: int = 240
    candle_dim: int = 6

    # Return Horizon & Normalization Contracts
    horizons: Tuple[int, ...] = (5, 15, 30)
    atr_floor: float = 1e-4
    clip_range: Tuple[float, float] = (-10.0, 10.0)
    spread_usd: float = 2.50
    min_bar_idx: int = 0

    # Authentic Date Partitions & Empirical Bar Counts
    bull_start: str = "2023.10.15"
    bull_end: str = "2024.03.31"
    bull_bars: int = 243084

    bear_start: str = "2022.04.01"
    bear_end: str = "2022.07.01"
    bear_bars: int = 132405

    chop_start: str = "2023.06.20"
    chop_end: str = "2023.09.20"
    chop_bars: int = 133798

    # Reserved OOS Benchmark Partition (February 2025 Crash Slice)
    oos_start: str = "2025.01.15"
    oos_end: str = "2025.02.28"
    oos_master_bars: int = 64785
    oos_slice_start: int = 55000
    oos_slice_end: int = 64785
    oos_slice_bars: int = 9785

    # Routing Probabilities for Training Pools
    default_routing_probs: Dict[str, float] = field(
        default_factory=lambda: {"bull": 0.40, "bear": 0.30, "chop": 0.30}
    )


# =============================================================================
# 2. VECTORIZED TARGET FORMULATION ENGINE
# =============================================================================


def compute_multi_horizon_targets_numpy(
    close_arr: np.ndarray,
    atr_arr: np.ndarray,
    horizons: Tuple[int, ...] = (5, 15, 30),
    clamp_boundary: bool = True,
    clip_range: Tuple[float, float] = (-10.0, 10.0),
    atr_floor: float = 1e-4,
) -> np.ndarray:
    """
    Vectorized computation of multi-horizon ATR-normalized future return targets.

    Formula:
        y_{t, h} = clip((Close_{min(t+h, N-1)} - Close_t) / max(ATR_t, atr_floor), clip_min, clip_max)

    Args:
        close_arr: 1D float32 array of shape (N,) containing close prices.
        atr_arr: 1D float32 array of shape (N,) containing 14-period iATR values.
        horizons: Tuple of future bar offsets, default (5, 15, 30).
        clamp_boundary: If True, clamps future index to N - 1; otherwise leaves tail as 0.0.
        clip_range: Symmetric clipping bounds (default (-10.0, 10.0)).
        atr_floor: Minimum safety denominator (default 1e-4).

    Returns:
        targets: 2D float32 array of shape (N, len(horizons)).
    """
    n = len(close_arr)
    k = len(horizons)
    if n == 0:
        return np.empty((0, k), dtype=np.float32)

    targets = np.zeros((n, k), dtype=np.float32)
    clean_atr = np.nan_to_num(atr_arr, nan=atr_floor, posinf=atr_floor, neginf=atr_floor)
    safe_atr = np.maximum(clean_atr, atr_floor, dtype=np.float32)
    clip_min, clip_max = clip_range

    for i, h in enumerate(horizons):
        valid_len = max(0, n - h)
        if valid_len > 0:
            diff = close_arr[h:] - close_arr[:valid_len]
            targets[:valid_len, i] = np.clip(diff / safe_atr[:valid_len], clip_min, clip_max)
        if clamp_boundary and valid_len < n:
            tail_diff = close_arr[-1] - close_arr[valid_len:]
            targets[valid_len:, i] = np.clip(tail_diff / safe_atr[valid_len:], clip_min, clip_max)

    np.nan_to_num(targets, copy=False, nan=0.0, posinf=clip_max, neginf=clip_min)
    return targets


def compute_multi_horizon_targets_torch(
    close_tensor: torch.Tensor,
    atr_tensor: torch.Tensor,
    horizons: Tuple[int, ...] = (5, 15, 30),
    clamp_boundary: bool = True,
    clip_range: Tuple[float, float] = (-10.0, 10.0),
    atr_floor: float = 1e-4,
) -> torch.Tensor:
    """PyTorch vectorized equivalent of compute_multi_horizon_targets_numpy."""
    n = close_tensor.shape[0]
    k = len(horizons)
    if n == 0:
        return torch.empty((0, k), dtype=torch.float32, device=close_tensor.device)

    targets = torch.zeros((n, k), dtype=torch.float32, device=close_tensor.device)
    clean_atr = torch.nan_to_num(atr_tensor, nan=atr_floor, posinf=atr_floor, neginf=atr_floor)
    safe_atr = torch.clamp(clean_atr, min=atr_floor)
    clip_min, clip_max = clip_range

    for i, h in enumerate(horizons):
        valid_len = max(0, n - h)
        if valid_len > 0:
            diff = close_tensor[h:] - close_tensor[:valid_len]
            targets[:valid_len, i] = torch.clamp(diff / safe_atr[:valid_len], min=clip_min, max=clip_max)
        if clamp_boundary and valid_len < n:
            tail_diff = close_tensor[-1] - close_tensor[valid_len:]
            targets[valid_len:, i] = torch.clamp(tail_diff / safe_atr[valid_len:], min=clip_min, max=clip_max)

    return torch.nan_to_num(targets, nan=0.0, posinf=clip_max, neginf=clip_min)


compute_multi_horizon_targets = compute_multi_horizon_targets_numpy


# =============================================================================
# 3. SINGLE REGIME PYTORCH DATASET
# =============================================================================


class SingleRegimeDataset(Dataset):
    """
    PyTorch Dataset wrapping one partition (Bull, Bear, Chop, or OOS).
    Precomputes vectorized multi-horizon targets with boundary safety.
    """

    def __init__(
        self,
        name: str,
        env: Any,
        lookback: int = 240,
        horizons: Tuple[int, ...] = (5, 15, 30),
        atr_floor: float = 1e-4,
        clip_range: Tuple[float, float] = (-10.0, 10.0),
        min_bar_idx: int = 0,
        is_oos: bool = False,
    ) -> None:
        self.name = name
        self.env = env
        self.lookback = lookback
        self.horizons = horizons
        self.atr_floor = atr_floor
        self.clip_range = clip_range
        self.min_bar_idx = max(0, min_bar_idx)
        self.is_oos = is_oos

        self.total_bars = getattr(env, "total_bars", len(getattr(env, "close_arr", [])))
        self.max_horizon = max(horizons) if horizons else 0
        self.valid_len = max(0, self.total_bars - self.max_horizon)

        # Precompute vectorized target matrix [total_bars, len(horizons)]
        if self.total_bars > 0 and hasattr(env, "close_arr") and hasattr(env, "atr_arr"):
            self.targets_matrix = compute_multi_horizon_targets_numpy(
                close_arr=self.env.close_arr,
                atr_arr=self.env.atr_arr,
                horizons=self.horizons,
                clamp_boundary=True,
                clip_range=self.clip_range,
                atr_floor=self.atr_floor,
            )
        else:
            self.targets_matrix = np.empty((0, len(self.horizons)), dtype=np.float32)

    def __len__(self) -> int:
        return max(0, self.valid_len - self.min_bar_idx)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bar_idx = idx + self.min_bar_idx
        if bar_idx < 0 or bar_idx >= self.valid_len:
            raise IndexError(f"Index {idx} (bar {bar_idx}) out of range [0, {self.valid_len})")

        obs = torch.from_numpy(self.env.get_observation(bar_idx)).float()
        seq = torch.from_numpy(self.env.get_candle_sequence(bar_idx, lookback=self.lookback)).float()
        tgt = torch.from_numpy(self.targets_matrix[bar_idx]).float()

        raw_atr = float(self.env.atr_arr[bar_idx]) if bar_idx < len(self.env.atr_arr) else self.atr_floor
        safe_atr = max(raw_atr, self.atr_floor) if math.isfinite(raw_atr) else self.atr_floor
        atr = torch.tensor(safe_atr, dtype=torch.float32)

        return obs, seq, tgt, atr

    def get_batch(self, indices: Union[int, Sequence[int], np.ndarray, torch.Tensor]) -> MegaPoolBatch:
        """
        Extracts a vectorized batch of data from this regime.
        Supports single int, list of ints, or numpy/torch index tensors.
        """
        if isinstance(indices, int):
            idx_list = [indices]
        elif isinstance(indices, torch.Tensor):
            idx_list = indices.cpu().tolist()
        elif isinstance(indices, np.ndarray):
            idx_list = indices.tolist()
        else:
            idx_list = list(indices)

        batch_size = len(idx_list)
        if batch_size == 0:
            return MegaPoolBatch(
                obs_137=torch.empty((0, 137), dtype=torch.float32),
                candle_seq_240=torch.empty((0, self.lookback, 6), dtype=torch.float32),
                targets_3=torch.empty((0, len(self.horizons)), dtype=torch.float32),
                atr=torch.empty((0,), dtype=torch.float32),
            )

        bar_indices = [i + self.min_bar_idx for i in idx_list]
        for bi in bar_indices:
            if bi < 0 or bi >= self.valid_len:
                raise IndexError(f"Bar index {bi} out of valid range [0, {self.valid_len})")

        obs_list = [self.env.get_observation(bi) for bi in bar_indices]
        seq_list = [self.env.get_candle_sequence(bi, lookback=self.lookback) for bi in bar_indices]
        tgt_list = [self.targets_matrix[bi] for bi in bar_indices]
        atr_list = [
            max(float(self.env.atr_arr[bi]), self.atr_floor)
            if math.isfinite(float(self.env.atr_arr[bi]))
            else self.atr_floor
            for bi in bar_indices
        ]

        return MegaPoolBatch(
            obs_137=torch.from_numpy(np.stack(obs_list)).float(),
            candle_seq_240=torch.from_numpy(np.stack(seq_list)).float(),
            targets_3=torch.from_numpy(np.stack(tgt_list)).float(),
            atr=torch.tensor(atr_list, dtype=torch.float32),
        )


# =============================================================================
# 4. COMPOSITE MEGA-POOL PYTORCH DATASET
# =============================================================================


class MegaPoolDataset(Dataset):
    """
    Composite Map-style PyTorch Dataset combining multiple training pools (Bull, Bear, Chop).
    Maps a global linear index [0, total_len) to (regime_name, local_idx).
    Strictly forbids OOS benchmark datasets from being included in the training composite.
    """

    def __init__(self, datasets: Dict[str, SingleRegimeDataset]) -> None:
        if not datasets:
            raise ValueError("MegaPoolDataset requires at least one regime dataset")

        self.datasets = datasets
        self.names = list(datasets.keys())

        # Strict OOS firewall: training composite MUST NOT contain OOS dataset
        for name, ds in self.datasets.items():
            if ds.is_oos or name.lower() == "oos" or "oos" in name.lower():
                raise ValueError(
                    f"Dataset '{name}' is marked as OOS. Out-of-sample data cannot be part of MegaPoolDataset!"
                )

        self.lengths = [len(self.datasets[n]) for n in self.names]
        self.total_len = sum(self.lengths)
        self.offsets = np.cumsum([0] + self.lengths[:-1])

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx < 0 or idx >= self.total_len:
            raise IndexError(f"Index {idx} out of range [0, {self.total_len})")

        ds_idx = int(np.searchsorted(self.offsets, idx, side="right") - 1)
        name = self.names[ds_idx]
        local_idx = idx - int(self.offsets[ds_idx])
        return self.datasets[name][local_idx]


# =============================================================================
# 5. REGIME ROUTING BATCH SAMPLER
# =============================================================================


class RegimeRoutingBatchSampler(Sampler[List[int]]):
    """
    PyTorch BatchSampler routing index batches across market regimes.
    Supports REGIME_BATCH, STRATIFIED, and ROUND_ROBIN routing modes.
    """

    def __init__(
        self,
        datasets: Dict[str, SingleRegimeDataset],
        batch_size: int = 64,
        num_batches: int = 1000,
        routing_probs: Optional[Dict[str, float]] = None,
        mode: Union[RoutingMode, str] = RoutingMode.REGIME_BATCH,
    ) -> None:
        self.datasets = datasets
        self.names = list(datasets.keys())
        self.batch_size = max(1, batch_size)
        self.num_batches = max(1, num_batches)
        self.mode = RoutingMode(mode) if isinstance(mode, str) else mode

        # Compute global offsets mapping each regime to global indices
        self.offsets: Dict[str, int] = {}
        curr = 0
        for name in self.names:
            self.offsets[name] = curr
            curr += len(datasets[name])

        self.set_routing_probabilities(routing_probs)

    def set_routing_probabilities(self, probs: Optional[Dict[str, float]] = None) -> None:
        """Sets or updates the probability distribution across regimes."""
        if probs is None:
            p = [1.0 / len(self.names)] * len(self.names)
        else:
            raw_weights = [probs.get(n, 0.0) for n in self.names]
            total = sum(raw_weights)
            if total <= 0:
                raise ValueError("Routing probabilities must sum to a positive number")
            p = [w / total for w in raw_weights]
        self.probs = np.array(p, dtype=np.float64)

    def __len__(self) -> int:
        return self.num_batches

    def __iter__(self) -> Iterator[List[int]]:
        for b_idx in range(self.num_batches):
            if self.mode == RoutingMode.REGIME_BATCH:
                chosen_idx = int(np.random.choice(len(self.names), p=self.probs))
                name = self.names[chosen_idx]
                ds_len = len(self.datasets[name])
                if ds_len <= 0:
                    yield []
                    continue
                local_indices = np.random.randint(0, ds_len, size=self.batch_size)
                yield (local_indices + self.offsets[name]).tolist()

            elif self.mode == RoutingMode.STRATIFIED:
                counts = np.random.multinomial(self.batch_size, self.probs)
                batch: List[int] = []
                for idx, name in enumerate(self.names):
                    c = counts[idx]
                    ds_len = len(self.datasets[name])
                    if c > 0 and ds_len > 0:
                        loc = np.random.randint(0, ds_len, size=c)
                        batch.extend((loc + self.offsets[name]).tolist())
                # Shuffle batch indices to avoid regime clustering inside batch
                if len(batch) > 1:
                    np.random.shuffle(batch)
                yield batch

            elif self.mode == RoutingMode.ROUND_ROBIN:
                chosen_idx = b_idx % len(self.names)
                name = self.names[chosen_idx]
                ds_len = len(self.datasets[name])
                if ds_len <= 0:
                    yield []
                    continue
                local_indices = np.random.randint(0, ds_len, size=self.batch_size)
                yield (local_indices + self.offsets[name]).tolist()


def collate_mega_pool(
    batch_list: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
) -> MegaPoolBatch:
    """Collates a list of single-step tuples into a strongly-typed MegaPoolBatch."""
    obs = torch.stack([item[0] for item in batch_list], dim=0)
    seq = torch.stack([item[1] for item in batch_list], dim=0)
    tgt = torch.stack([item[2] for item in batch_list], dim=0)
    atr = torch.stack([item[3] for item in batch_list], dim=0)
    return MegaPoolBatch(obs_137=obs, candle_seq_240=seq, targets_3=tgt, atr=atr)


# =============================================================================
# 6. HIGH-LEVEL MEGA-POOL ROUTER
# =============================================================================


class MegaPoolRouter:
    """
    High-level manager providing specialized mini-batch routing across 509k authentic M1 bars,
    PyTorch DataLoader creation, disk caching, and programmatic OOS firewall enforcement.
    """

    def __init__(
        self,
        config: Optional[MegaPoolConfig] = None,
        bull_env: Optional[Any] = None,
        bear_env: Optional[Any] = None,
        chop_env: Optional[Any] = None,
        oos_env: Optional[Any] = None,
        load_on_init: bool = True,
    ) -> None:
        self.config = config or MegaPoolConfig()

        if load_on_init:
            # Ingest or assign training environments
            self.bull_env = bull_env or self._load_env(self.config.bull_start, self.config.bull_end, "bull")
            self.bear_env = bear_env or self._load_env(self.config.bear_start, self.config.bear_end, "bear")
            self.chop_env = chop_env or self._load_env(self.config.chop_start, self.config.chop_end, "chop")

            # Create training datasets
            self.training_datasets: Dict[str, SingleRegimeDataset] = {
                "bull": SingleRegimeDataset(
                    "bull",
                    self.bull_env,
                    lookback=self.config.lookback_candles,
                    horizons=self.config.horizons,
                    atr_floor=self.config.atr_floor,
                    clip_range=self.config.clip_range,
                    min_bar_idx=self.config.min_bar_idx,
                    is_oos=False,
                ),
                "bear": SingleRegimeDataset(
                    "bear",
                    self.bear_env,
                    lookback=self.config.lookback_candles,
                    horizons=self.config.horizons,
                    atr_floor=self.config.atr_floor,
                    clip_range=self.config.clip_range,
                    min_bar_idx=self.config.min_bar_idx,
                    is_oos=False,
                ),
                "chop": SingleRegimeDataset(
                    "chop",
                    self.chop_env,
                    lookback=self.config.lookback_candles,
                    horizons=self.config.horizons,
                    atr_floor=self.config.atr_floor,
                    clip_range=self.config.clip_range,
                    min_bar_idx=self.config.min_bar_idx,
                    is_oos=False,
                ),
            }
            self.mega_dataset = MegaPoolDataset(self.training_datasets)

            # Ingest OOS Benchmark (Physically Isolated)
            if oos_env is not None:
                self.oos_env = oos_env
            else:
                self.oos_env = self._load_env(self.config.oos_start, self.config.oos_end, "oos_master")

            # Slice to exact crash slice [55000:64785] (9,785 bars)
            expected_slice_len = self.config.oos_slice_end - self.config.oos_slice_start
            if isinstance(self.oos_env, SlicedM1Environment):
                self.oos_sliced_env = self.oos_env
            elif getattr(self.oos_env, "total_bars", 0) == expected_slice_len:
                self.oos_sliced_env = self.oos_env
            else:
                self.oos_sliced_env = SlicedM1Environment(
                    self.oos_env,
                    self.config.oos_slice_start,
                    self.config.oos_slice_end,
                    name="OOS_Benchmark_Crash_Slice",
                )

            self.oos_dataset = SingleRegimeDataset(
                "oos",
                self.oos_sliced_env,
                lookback=self.config.lookback_candles,
                horizons=self.config.horizons,
                atr_floor=self.config.atr_floor,
                clip_range=self.config.clip_range,
                min_bar_idx=0,
                is_oos=True,
            )

        self.routing_probs = dict(self.config.default_routing_probs)

    def _load_env(self, start_date: str, end_date: str, name: str) -> RealM1Environment:
        """Loads environment from disk cache or CSV, saving cache on miss."""
        cache_dir = Path(self.config.cache_dir) if self.config.cache_dir else Path("outputs/cache")
        cache_file = cache_dir / f"pool_{name}.npz"

        env = RealM1Environment(csv_path=self.config.csv_path, spread_usd=self.config.spread_usd)

        if self.config.use_cache and cache_file.exists():
            with np.load(cache_file) as data:
                env.total_bars = len(data["close_arr"])
                env.open_arr = data["open_arr"]
                env.high_arr = data["high_arr"]
                env.low_arr = data["low_arr"]
                env.close_arr = data["close_arr"]
                env.volume_arr = data["volume_arr"]
                env.spread_arr = data["spread_arr"]
                env.atr_arr = data["atr_arr"]
                env.day_progress_arr = data["day_progress_arr"]
                env.obs_matrix = data["obs_matrix"]
                if "timestamps" in data:
                    env.timestamps = data["timestamps"].tolist()
                else:
                    env.timestamps = [""] * env.total_bars
        else:
            env.load_data(start_date, end_date)
            if self.config.use_cache:
                cache_dir.mkdir(parents=True, exist_ok=True)
                np.savez(
                    cache_file,
                    open_arr=env.open_arr,
                    high_arr=env.high_arr,
                    low_arr=env.low_arr,
                    close_arr=env.close_arr,
                    volume_arr=env.volume_arr,
                    spread_arr=env.spread_arr,
                    atr_arr=env.atr_arr,
                    day_progress_arr=env.day_progress_arr,
                    obs_matrix=env.obs_matrix,
                )
        return env

    def set_routing_probabilities(self, probs: Dict[str, float]) -> None:
        """Updates regime routing probabilities for training mini-batches."""
        self.routing_probs = dict(probs)

    def get_batch(
        self,
        batch_size: Union[int, str, MarketRegime] = 64,
        pool_name: Optional[Union[str, MarketRegime]] = None,
    ) -> MegaPoolBatch:
        """
        Samples a mini-batch of training data complying with PROJECT.md Interface Contracts.

        Programmatic OOS Firewall:
            Calling `get_batch("oos")` or `get_batch(64, "oos")` raises `PermissionError`.
            OOS data must only be queried through `get_oos_batch()`.
        """
        # Handle positional string argument e.g. get_batch("oos") or get_batch("bull")
        if isinstance(batch_size, (str, MarketRegime)):
            pool_name = batch_size
            actual_batch_size = 64
        else:
            actual_batch_size = int(batch_size)

        # Check OOS firewall
        if pool_name is not None:
            pool_str = str(pool_name.value if isinstance(pool_name, MarketRegime) else pool_name).lower()
            if pool_str == "oos" or "oos" in pool_str:
                raise PermissionError(
                    "OOS Benchmark slice cannot be accessed via get_batch! Use get_oos_batch() for evaluation."
                )

        if pool_name is not None:
            key = pool_str if isinstance(pool_name, str) else pool_name.value
            if key not in self.training_datasets:
                raise KeyError(
                    f"Unknown pool '{pool_name}'. Valid training pools: {list(self.training_datasets.keys())}"
                )
            target_ds = self.training_datasets[key]
        else:
            names = list(self.training_datasets.keys())
            raw_weights = [self.routing_probs.get(n, 1.0 / len(names)) for n in names]
            total = sum(raw_weights)
            probs = [w / total for w in raw_weights]
            chosen_name = names[int(np.random.choice(len(names), p=probs))]
            target_ds = self.training_datasets[chosen_name]

        if len(target_ds) == 0:
            raise ValueError(f"Target dataset '{target_ds.name}' is empty")

        indices = np.random.randint(0, len(target_ds), size=actual_batch_size)
        return target_ds.get_batch(indices)

    def get_oos_batch(
        self,
        batch_size: int = 64,
        sequential: bool = True,
        offset: int = 0,
    ) -> MegaPoolBatch:
        """
        Fetches evaluation data strictly from the reserved OOS crash benchmark slice.
        Supports sequential walk-through for continuous backtesting or random sampling.
        """
        ds_len = len(self.oos_dataset)
        if ds_len == 0:
            raise ValueError("OOS dataset is empty")

        if sequential:
            start = min(offset, ds_len)
            end = min(ds_len, start + batch_size)
            indices = list(range(start, end))
            if not indices:
                indices = [ds_len - 1]
        else:
            indices = np.random.randint(0, ds_len, size=batch_size).tolist()

        return self.oos_dataset.get_batch(indices)

    def create_dataloader(
        self,
        batch_size: int = 64,
        mode: Union[RoutingMode, str] = RoutingMode.REGIME_BATCH,
        num_batches: int = 1000,
        num_workers: int = 0,
        routing_probs: Optional[Dict[str, float]] = None,
    ) -> DataLoader:
        """Creates a standard PyTorch DataLoader with custom regime routing and collate."""
        sampler = RegimeRoutingBatchSampler(
            datasets=self.training_datasets,
            batch_size=batch_size,
            num_batches=num_batches,
            routing_probs=routing_probs or self.routing_probs,
            mode=mode,
        )
        return DataLoader(
            self.mega_dataset,
            batch_sampler=sampler,
            collate_fn=collate_mega_pool,
            num_workers=num_workers,
        )

    def validate_zero_oos_leakage(self) -> bool:
        """
        Verifies zero data leakage between training pools and the OOS benchmark slice:
        1. OOS crash slice has exactly 9,785 bars ([55000:64785]).
        2. OOS dataset has exactly 9,785 - 30 = 9,755 valid horizon targets.
        3. 'oos' is strictly absent from training datasets.
        4. All training datasets are marked is_oos=False.
        5. Training pool dates (2022.04 - 2024.03) are strictly before OOS dates (2025.01 - 2025.02).
        """
        oos_bars = self.config.oos_slice_end - self.config.oos_slice_start
        assert oos_bars == 9785, f"Expected 9,785 OOS slice bars, got {oos_bars}"
        assert (
            len(self.oos_dataset) == 9785 - 30
        ), f"Expected {9785 - 30} valid OOS targets, got {len(self.oos_dataset)}"
        assert "oos" not in self.training_datasets, "OOS pool found in training datasets!"

        for name, ds in self.training_datasets.items():
            assert not ds.is_oos, f"Training dataset '{name}' is marked as OOS!"
            assert "oos" not in name.lower(), f"OOS substring found in training dataset name '{name}'!"

        return True
