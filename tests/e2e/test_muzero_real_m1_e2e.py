"""
E2E Test Suite Bridge: tests/e2e/test_muzero_real_m1_e2e.py
Imports the complete 4-tier E2E test suite from research_reasoning_world_class/tests/test_muzero_real_m1_e2e.py
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

# Lock single CPU thread for sub-50ms tick execution SLA
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
torch.set_num_threads(1)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
REASONING_DIR = PROJECT_ROOT / "research_reasoning_world_class"
if str(REASONING_DIR) not in sys.path:
    sys.path.insert(0, str(REASONING_DIR))

from research_reasoning_world_class.tests.test_muzero_real_m1_e2e import (
    TestTier1FeatureCoverage,
    TestTier2BoundaryAndCornerCases,
    TestTier3CrossFeatureCombinations,
    TestTier4RealWorldApplicationScenarios,
)
import unittest

if __name__ == "__main__":
    unittest.main()
