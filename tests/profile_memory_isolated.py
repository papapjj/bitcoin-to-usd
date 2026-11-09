"""
Isolated Process Resident RAM Profiler across 100 Training Steps.
File: tests/profile_memory_isolated.py
"""

import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RRWC_PATH = REPO_ROOT / "research_reasoning_world_class"
if str(RRWC_PATH) not in sys.path:
    sys.path.insert(0, str(RRWC_PATH))

import torch

from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    MultiHorizonProjectionHead,
)
from research_reasoning_world_class.src.mega_pool_dataset import (
    MegaPoolRouter,
    MegaPoolConfig,
    RoutingMode,
)
from scripts.train_tca_multi_horizon import (
    build_lr_scheduler,
    build_phase1_optimizer,
    execute_training_step,
    set_seed,
)

CACHE_DIR = REPO_ROOT / "outputs" / "cache"


def get_resident_ram_mb() -> float:
    """Returns current process Working Set Size in MB."""
    class PROCESS_MEMORY_COUNTERS_EX(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
            ("PrivateUsage", ctypes.c_size_t),
        ]

    handle = ctypes.windll.kernel32.OpenProcess(0x0410, False, os.getpid())
    if not handle:
        return 0.0
    try:
        counters = PROCESS_MEMORY_COUNTERS_EX()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS_EX)
        if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.WorkingSetSize / (1024.0 * 1024.0)
        return 0.0
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def main():
    set_seed(42)
    device = torch.device("cpu")

    initial_ram = get_resident_ram_mb()

    cfg = MegaPoolConfig(cache_dir=str(CACHE_DIR), use_cache=True)
    router = MegaPoolRouter(config=cfg)
    router_loaded_ram = get_resident_ram_mb()

    wm = LatentDynamicsWorldModel().to(device)
    wm.enable_temporal_stream(candle_len=240)
    head = MultiHorizonProjectionHead(d_in=768, num_horizons=3).to(device)
    trainable = list(wm.temporal_encoder.parameters()) + list(head.parameters())
    opt = build_phase1_optimizer(wm.temporal_encoder, head)
    sched = build_lr_scheduler(opt, total_steps=100)

    models_loaded_ram = get_resident_ram_mb()

    num_steps = 100
    batch_size = 16
    dataloader = router.create_dataloader(
        batch_size=batch_size, mode=RoutingMode.STRATIFIED, num_batches=num_steps
    )

    ram_samples = []
    for step, batch in enumerate(dataloader, 1):
        success, _ = execute_training_step(
            batch=batch,
            world_model=wm,
            head=head,
            optimizer=opt,
            scheduler=sched,
            trainable_params=trainable,
            device=device,
            beta=1.0,
        )
        assert success is True
        current_ram = get_resident_ram_mb()
        ram_samples.append(current_ram)
        if step in [1, 10, 25, 50, 75, 100]:
            print(f"[Step {step:3d}] Current RAM: {current_ram:.2f} MB", flush=True)

    peak_ram = max(ram_samples)
    min_ram = min(ram_samples)
    mid_ram = ram_samples[49]
    final_ram = ram_samples[-1]
    steady_state_delta = final_ram - mid_ram

    result = {
        "initial_ram_mb": round(initial_ram, 2),
        "router_loaded_ram_mb": round(router_loaded_ram, 2),
        "models_loaded_ram_mb": round(models_loaded_ram, 2),
        "step_1_ram_mb": round(ram_samples[0], 2),
        "step_50_ram_mb": round(mid_ram, 2),
        "step_100_ram_mb": round(final_ram, 2),
        "peak_ram_mb": round(peak_ram, 2),
        "steady_state_delta_mb": round(steady_state_delta, 2),
        "bounded_under_1gb": bool(peak_ram < 1024.0),
        "zero_memory_leak": bool(abs(steady_state_delta) < 75.0),
    }

    print("\n--- ISOLATED PROFILE RESULTS ---")
    print(json.dumps(result, indent=2))

    assert result["bounded_under_1gb"], f"Peak RAM {peak_ram:.2f} MB >= 1024.0 MB (1.0 GB)"
    assert result["zero_memory_leak"], f"Memory leak delta {steady_state_delta:.2f} MB >= 75.0 MB"


if __name__ == "__main__":
    main()
