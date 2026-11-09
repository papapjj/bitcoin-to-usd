import pytest
import torch
import torch.nn as nn
from research_reasoning_world_class.src.metacognitive_reasoning import (
    ReasoningConfig,
    CoTReasoningHead,
    MetacognitiveReasoningEngine,
    MetacognitiveOutput,
)


class MockConstantDir(nn.Module):
    def __init__(self, val: float):
        super().__init__()
        self.val = val

    def forward(self, x):
        # x shape: [B, D] or [B, P, D]
        if x.dim() == 2:
            return torch.full((x.size(0), 1), self.val, device=x.device, dtype=x.dtype)
        elif x.dim() == 3:
            return torch.full((x.size(0), x.size(1), 1), self.val, device=x.device, dtype=x.dtype)
        return torch.full((1, 1), self.val, device=x.device, dtype=x.dtype)


def test_dynamic_fault_step_backtracking_hop2():
    """
    Test verifying that when a contradiction occurs between
    Hop 2 (index 1) and Hop 4 (index 3), the engine attributes fault to Hop 2
    (faulty_step = 1).
    """
    torch.manual_seed(42)
    config = ReasoningConfig(
        input_dim=64,
        hidden_dim=32,
        num_hops=4,
        max_hops=4,
        enable_cross_hop=True,
        enable_backtrack=True,
        enable_act=False,
    )
    cot_head = CoTReasoningHead(config)
    cot_head.eval()

    # Create dummy input
    x = torch.randn(1, 64)
    smc = torch.randn(1, 64)

    class SelectiveHop2TrapDetector(nn.Module):
        def forward(self, inputs):
            if inputs.dim() == 3:
                # inputs: [B, P, 2D+S]
                # candidate_pairs order for 4 hops:
                # (0,1)=0, (0,2)=1, (0,3)=2, (1,2)=3, (1,3)=4, (2,3)=5
                B, P, _ = inputs.shape
                confs = torch.zeros(B, P, device=inputs.device)
                confs[:, 4] = 0.98  # Pair (1, 3) -> Hop 2 vs Hop 4
                return confs
            else:
                # inputs: [B, 2D+S]
                B, _ = inputs.shape
                return torch.full((B, 1), 0.98, device=inputs.device)

    cot_head.backtrack_module.detector = SelectiveHop2TrapDetector()
    cot_head.backtrack_module.contradiction_detector.detector = cot_head.backtrack_module.detector
    cot_head.backtrack_module.contradiction_detector.dir_h1 = MockConstantDir(1.0)
    cot_head.backtrack_module.contradiction_detector.dir_hk = MockConstantDir(-1.0)

    # Run forward
    all_hops, final_thought = cot_head(x, smc_context=smc)

    assert cot_head.last_was_backtracked is True
    assert cot_head.last_faulty_step == 1, f"Expected faulty_step == 1, got {cot_head.last_faulty_step}"


def test_selective_attribution_preserves_hop0():
    """
    Ensure that when faulty_step = 1 (Hop 2), hop_states[0] (Hop 1) is preserved before and after.
    """
    torch.manual_seed(42)
    config = ReasoningConfig(
        input_dim=64,
        hidden_dim=32,
        num_hops=4,
        max_hops=4,
        enable_cross_hop=False,
        enable_backtrack=True,
        enable_act=False,
    )
    cot_head = CoTReasoningHead(config)
    cot_head.eval()

    class Hop1TrapDetector(nn.Module):
        def forward(self, inputs):
            if inputs.dim() == 3:
                B, P, _ = inputs.shape
                confs = torch.zeros(B, P, device=inputs.device)
                confs[:, 4] = 0.95  # Pair (1, 3) has contradiction
                return confs
            else:
                B, _ = inputs.shape
                return torch.full((B, 1), 0.95, device=inputs.device)

    cot_head.backtrack_module.detector = Hop1TrapDetector()
    cot_head.backtrack_module.contradiction_detector.detector = cot_head.backtrack_module.detector
    cot_head.backtrack_module.contradiction_detector.dir_h1 = MockConstantDir(1.0)
    cot_head.backtrack_module.contradiction_detector.dir_hk = MockConstantDir(-1.0)

    # We also run without backtrack to capture original hop 0
    cot_head.enable_backtrack = False
    x = torch.randn(1, 64)
    orig_hops, _ = cot_head(x)
    orig_hop0 = orig_hops[:, 0, :].clone()

    cot_head.enable_backtrack = True
    revised_hops, _ = cot_head(x)

    assert cot_head.last_was_backtracked is True
    assert cot_head.last_faulty_step == 1
    # Hop 0 must be 100% identical!
    diff = torch.norm(revised_hops[:, 0, :] - orig_hop0).item()
    assert diff < 1e-6, f"Hop 0 was mutated! Difference = {diff}"


def test_metacognitive_engine_propagates_faulty_step():
    """
    Test MetacognitiveReasoningEngine outputs include faulty_step in MetacognitiveOutput and ReasoningVerdict.
    """
    config = ReasoningConfig(
        input_dim=64,
        hidden_dim=32,
        num_hops=4,
        max_hops=4,
        enable_backtrack=True,
    )
    engine = MetacognitiveReasoningEngine(config)
    engine.eval()

    x = torch.randn(1, 64)
    out = engine(x)
    assert hasattr(out, "faulty_step")
    verdict = out.to_verdict(0)
    assert hasattr(verdict, "faulty_step")
    assert verdict.metadata.get("faulty_step") is not None
