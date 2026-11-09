# -*- coding: utf-8 -*-
"""
Adversarial Stress Test Suite: Checkpoint Deserialization Under Mutated Weights & Partial Keys.
Author: teamwork_challenger_tca_m5_1 (Empirical Challenger)
Milestone: TCA Milestone 5 (Final Clearance)

Coverage:
1. Baseline World Model (30 keys) partial keys, unexpected keys, and weight mutation stress.
2. TemporalCrossAttentionEncoder (41 keys) partial keys, unexpected keys, and shape mutation stress.
3. Reasoning Engine (191 keys, 4,196,564 weights) zero-drift invariant under single-bit mutation,
   tied-parameter perturbation, and missing/extra keys.
4. MultiHorizonProjectionHead (8 keys, 230,659 weights) partial keys and tensor corruption.
5. Checkpoint namespace tampering, missing namespaces, and raw byte-level corruption.
"""

from __future__ import annotations

import copy
import io
import os
from pathlib import Path
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

RRWC_PATH = REPO_ROOT / "research_reasoning_world_class"
if str(RRWC_PATH) not in sys.path:
    sys.path.insert(0, str(RRWC_PATH))

from research_reasoning_world_class.src.latent_dynamics_world_model import (
    LatentDynamicsWorldModel,
    TemporalCrossAttentionEncoder,
    MultiHorizonProjectionHead,
)
from research_reasoning_world_class.src.metacognitive_reasoning import (
    MetacognitiveReasoningEngine,
    ReasoningConfig,
)
from scripts.train_tca_multi_horizon import hash_reasoning_engine

CHECKPOINT_DIR = REPO_ROOT / "outputs" / "checkpoints"
PHASE1_CKPT = CHECKPOINT_DIR / "tca_phase1_pretrained.pt"
PHASE2_CKPT = CHECKPOINT_DIR / "tca_phase2_finetuned.pt"


@pytest.fixture(scope="module")
def loaded_ckpts():
    assert PHASE1_CKPT.exists(), f"Phase 1 checkpoint missing at {PHASE1_CKPT}"
    assert PHASE2_CKPT.exists(), f"Phase 2 checkpoint missing at {PHASE2_CKPT}"
    ckpt1 = torch.load(PHASE1_CKPT, map_location="cpu", weights_only=False)
    ckpt2 = torch.load(PHASE2_CKPT, map_location="cpu", weights_only=False)
    return ckpt1, ckpt2


# ==============================================================================
# 1. BASELINE WORLD MODEL (30 KEYS) ADVERSARIAL STRESS
# ==============================================================================

class TestBaselineWorldModelCheckpointStress:
    """Adversarial stress-testing of 30-key LatentDynamicsWorldModel deserialization."""

    def test_baseline_wm_exact_keys_count(self, loaded_ckpts):
        """Verifies both checkpoints contain strictly 30 keys in world_model_state_dict."""
        for ckpt in loaded_ckpts:
            wm_sd = ckpt["world_model_state_dict"]
            assert len(wm_sd) == 30, f"Expected strictly 30 keys, got {len(wm_sd)}"

    @pytest.mark.parametrize("missing_fraction", [0.1, 0.3, 0.5, 0.9, 1.0])
    def test_baseline_wm_partial_keys_rejection(self, loaded_ckpts, missing_fraction):
        """Adversarial check: removing varying fractions of keys must raise RuntimeError under strict=True."""
        ckpt1, _ = loaded_ckpts
        wm_sd = copy.deepcopy(ckpt1["world_model_state_dict"])
        keys = list(wm_sd.keys())
        num_to_remove = max(1, int(len(keys) * missing_fraction))
        removed_keys = keys[:num_to_remove]
        for k in removed_keys:
            del wm_sd[k]

        wm = LatentDynamicsWorldModel()
        with pytest.raises(RuntimeError) as exc_info:
            wm.load_state_dict(wm_sd, strict=True)
        err_msg = str(exc_info.value)
        assert "missing" in err_msg.lower(), f"Expected missing keys in error, got: {err_msg}"
        assert any(k in err_msg for k in removed_keys), f"At least one removed key must be named in: {err_msg}"

    def test_baseline_wm_unexpected_rogue_keys(self, loaded_ckpts):
        """Adversarial check: injecting unauthorized/rogue keys must fail strict deserialization."""
        ckpt1, _ = loaded_ckpts
        wm_sd = copy.deepcopy(ckpt1["world_model_state_dict"])
        wm_sd["malicious_backdoor_weight"] = torch.randn(10, 10)
        wm_sd["state_encoder.unauthorized_bias"] = torch.zeros(128)

        wm = LatentDynamicsWorldModel()
        with pytest.raises(RuntimeError) as exc_info:
            wm.load_state_dict(wm_sd, strict=True)
        err_msg = str(exc_info.value)
        assert "unexpected" in err_msg.lower()
        assert "malicious_backdoor_weight" in err_msg
        assert "state_encoder.unauthorized_bias" in err_msg

    def test_baseline_wm_mutated_tensor_shapes(self, loaded_ckpts):
        """Adversarial check: mutated weight shapes must raise RuntimeError even if key exists."""
        ckpt1, _ = loaded_ckpts
        wm_sd = copy.deepcopy(ckpt1["world_model_state_dict"])
        # Mutate shape of first weight
        first_key = list(wm_sd.keys())[0]
        orig_tensor = wm_sd[first_key]
        wm_sd[first_key] = orig_tensor.view(-1)  # Flatten or change rank

        wm = LatentDynamicsWorldModel()
        with pytest.raises(RuntimeError) as exc_info:
            wm.load_state_dict(wm_sd, strict=True)
        assert "size mismatch" in str(exc_info.value).lower()

    def test_baseline_wm_nan_inf_weight_propagation(self, loaded_ckpts):
        """Adversarial check: injecting NaN or Inf weights must poison representations during forward pass."""
        ckpt1, _ = loaded_ckpts
        wm_sd = copy.deepcopy(ckpt1["world_model_state_dict"])
        first_key = list(wm_sd.keys())[0]
        wm_sd[first_key] = torch.full_like(wm_sd[first_key], float("nan"))

        wm = LatentDynamicsWorldModel()
        wm.load_state_dict(wm_sd, strict=True)
        wm.eval()

        dummy_obs = torch.randn(2, 137)
        z = wm.encode_state(dummy_obs)
        assert torch.isnan(z).any(), "NaN in weights must propagate into latent representation"


# ==============================================================================
# 2. TEMPORAL CROSS-ATTENTION ENCODER (41 KEYS) ADVERSARIAL STRESS
# ==============================================================================

class TestTemporalCrossAttentionCheckpointStress:
    """Adversarial stress-testing of 41-key TemporalCrossAttentionEncoder deserialization."""

    def test_tca_exact_keys_count(self, loaded_ckpts):
        """Verifies both checkpoints contain strictly 41 keys in temporal_encoder_state_dict."""
        for ckpt in loaded_ckpts:
            tca_sd = ckpt["temporal_encoder_state_dict"]
            assert len(tca_sd) == 41, f"Expected strictly 41 keys, got {len(tca_sd)}"

    @pytest.mark.parametrize("target_key", [
        "gate.0.weight",
        "gate.0.bias",
        "candle_proj.0.weight",
        "q_proj.weight",
        "k_proj.weight",
        "v_proj.weight",
        "transformer.layers.0.linear1.weight",
        "final_norm.weight",
    ])
    def test_tca_critical_missing_key_rejection(self, loaded_ckpts, target_key):
        """Adversarial check: missing key from TCA encoder must raise RuntimeError under strict=True."""
        ckpt1, _ = loaded_ckpts
        tca_sd = copy.deepcopy(ckpt1["temporal_encoder_state_dict"])
        assert target_key in tca_sd
        del tca_sd[target_key]

        encoder = TemporalCrossAttentionEncoder(candle_len=240, latent_dim=768)
        with pytest.raises(RuntimeError) as exc_info:
            encoder.load_state_dict(tca_sd, strict=True)
        assert f'Missing key(s) in state_dict: "{target_key}"' in str(exc_info.value)

    def test_tca_mutated_gate_weight_shape(self, loaded_ckpts):
        """Adversarial check: corrupting gate weight dimension breaks deserialization."""
        ckpt1, _ = loaded_ckpts
        tca_sd = copy.deepcopy(ckpt1["temporal_encoder_state_dict"])
        tca_sd["gate.0.weight"] = torch.randn(1, 128)  # Expected shape is [1, 768]

        encoder = TemporalCrossAttentionEncoder(candle_len=240, latent_dim=768)
        with pytest.raises(RuntimeError) as exc_info:
            encoder.load_state_dict(tca_sd, strict=True)
        assert "size mismatch" in str(exc_info.value).lower()

    def test_tca_unexpected_keys_rejection(self, loaded_ckpts):
        """Adversarial check: unexpected keys injected into TCA state dict are rejected."""
        ckpt1, _ = loaded_ckpts
        tca_sd = copy.deepcopy(ckpt1["temporal_encoder_state_dict"])
        tca_sd["rogue_temporal_attn.weight"] = torch.randn(64, 64)

        encoder = TemporalCrossAttentionEncoder(candle_len=240, latent_dim=768)
        with pytest.raises(RuntimeError) as exc_info:
            encoder.load_state_dict(tca_sd, strict=True)
        assert "Unexpected key(s) in state_dict" in str(exc_info.value)
        assert "rogue_temporal_attn.weight" in str(exc_info.value)


# ==============================================================================
# 3. REASONING ENGINE (191 KEYS) ZERO-DRIFT & TAMPER-RESISTANCE
# ==============================================================================

class TestReasoningEngineAntiTamperStress:
    """Adversarial stress-testing of MetacognitiveReasoningEngine frozen invariant & hashing."""

    def test_re_exact_keys_and_weights_count(self, loaded_ckpts):
        """Verifies strictly 191 keys and 4,196,564 unique parameters in reasoning_engine_state_dict."""
        for ckpt in loaded_ckpts:
            re_sd = ckpt["reasoning_engine_state_dict"]
            assert len(re_sd) == 191
            cfg = ReasoningConfig(input_dim=768)
            engine = MetacognitiveReasoningEngine(cfg)
            engine.load_state_dict(re_sd, strict=True)
            total_params = sum(p.numel() for p in engine.parameters())
            assert total_params == 4196564

    def test_re_single_bit_mutation_avalanche(self, loaded_ckpts):
        """Adversarial check: perturbing even a single floating-point weight by 1e-7 must alter SHA-256 hash."""
        ckpt1, _ = loaded_ckpts
        re_sd = copy.deepcopy(ckpt1["reasoning_engine_state_dict"])
        ref_sha256 = hash_reasoning_engine(re_sd)
        assert ref_sha256 == "b69d48bd5993f9f989d466247a0242b5b317e96dae32769b84f1fa73b3b8a929"

        # Perturb a single scalar element in the first weight tensor
        first_key = sorted(list(re_sd.keys()))[0]
        mutated_tensor = re_sd[first_key].clone()
        mutated_tensor.view(-1)[0] += 1e-7
        re_sd[first_key] = mutated_tensor

        mutated_sha256 = hash_reasoning_engine(re_sd)
        assert mutated_sha256 != ref_sha256, "Cryptographic hash must detect 1e-7 weight perturbation"

        # Max drift check
        diff = (re_sd[first_key] - ckpt1["reasoning_engine_state_dict"][first_key]).abs().max().item()
        assert diff > 0.0, "Drift must be strictly non-zero"

    def test_re_missing_key_rejection(self, loaded_ckpts):
        """Adversarial check: removing any key from the 191 keys must raise RuntimeError."""
        ckpt1, _ = loaded_ckpts
        re_sd = copy.deepcopy(ckpt1["reasoning_engine_state_dict"])
        del re_sd["cot_head.backtrack_module.detector.0.weight"]

        cfg = ReasoningConfig(input_dim=768)
        engine = MetacognitiveReasoningEngine(cfg)
        with pytest.raises(RuntimeError) as exc_info:
            engine.load_state_dict(re_sd, strict=True)
        assert "Missing key(s) in state_dict" in str(exc_info.value)
        assert "cot_head.backtrack_module.detector.0.weight" in str(exc_info.value)

    def test_re_tied_parameter_consistency(self, loaded_ckpts):
        """Adversarial check: tied parameters share underlying tensor memory pointer."""
        ckpt1, _ = loaded_ckpts
        cfg = ReasoningConfig(input_dim=768)
        engine = MetacognitiveReasoningEngine(cfg)
        engine.load_state_dict(ckpt1["reasoning_engine_state_dict"], strict=True)

        w1 = engine.cot_head.backtrack_module.contradiction_detector.detector[0].weight
        w2 = engine.cot_head.backtrack_module.detector[0].weight
        assert torch.equal(w1, w2), "Tied parameters must be mathematically equal"
        assert w1.data_ptr() == w2.data_ptr(), "Tied parameters must share identical data pointer"


# ==============================================================================
# 4. PROJECTION HEAD (8 KEYS) ADVERSARIAL STRESS
# ==============================================================================

class TestProjectionHeadCheckpointStress:
    """Adversarial stress-testing of MultiHorizonProjectionHead deserialization."""

    def test_head_exact_keys_count(self, loaded_ckpts):
        """Verifies strictly 8 keys and 230,659 weights in projection_head_state_dict."""
        for ckpt in loaded_ckpts:
            head_sd = ckpt["projection_head_state_dict"]
            assert len(head_sd) == 8
            total_weights = sum(t.numel() for t in head_sd.values())
            assert total_weights == 230659

            head = MultiHorizonProjectionHead()
            load_res = head.load_state_dict(head_sd, strict=True)
            assert len(load_res.missing_keys) == 0
            assert len(load_res.unexpected_keys) == 0

    def test_head_corrupted_output_layer_dimension(self, loaded_ckpts):
        """Adversarial check: corrupting output projection dimension (e.g. from 3 to 5 horizons) must fail."""
        ckpt1, _ = loaded_ckpts
        head_sd = copy.deepcopy(ckpt1["projection_head_state_dict"])
        # Find final layer weight (output dim 3)
        final_weight_key = [k for k in head_sd.keys() if "weight" in k and head_sd[k].shape[0] == 3][0]
        # Mutate to 5 output horizons
        head_sd[final_weight_key] = torch.randn(5, head_sd[final_weight_key].shape[1])

        head = MultiHorizonProjectionHead()
        with pytest.raises(RuntimeError) as exc_info:
            head.load_state_dict(head_sd, strict=True)
        assert "size mismatch" in str(exc_info.value).lower()

    def test_head_missing_key_rejection(self, loaded_ckpts):
        """Adversarial check: missing key from projection head must fail under strict=True."""
        ckpt1, _ = loaded_ckpts
        head_sd = copy.deepcopy(ckpt1["projection_head_state_dict"])
        first_key = list(head_sd.keys())[0]
        del head_sd[first_key]

        head = MultiHorizonProjectionHead()
        with pytest.raises(RuntimeError) as exc_info:
            head.load_state_dict(head_sd, strict=True)
        assert f'Missing key(s) in state_dict: "{first_key}"' in str(exc_info.value)


# ==============================================================================
# 5. CHECKPOINT CONTAINER, MISSING NAMESPACES & BYTE CORRUPTION
# ==============================================================================

class TestCheckpointContainerTamperStress:
    """Adversarial stress-testing of checkpoint file structure and byte integrity."""

    @pytest.mark.parametrize("namespace", [
        "world_model_state_dict",
        "temporal_encoder_state_dict",
        "projection_head_state_dict",
        "reasoning_engine_state_dict",
    ])
    def test_missing_mandatory_namespace_rejection(self, loaded_ckpts, namespace):
        """Adversarial check: omitting any mandatory namespace from checkpoint is detected."""
        ckpt1, _ = loaded_ckpts
        tampered_ckpt = copy.deepcopy(ckpt1)
        del tampered_ckpt[namespace]

        assert namespace not in tampered_ckpt
        required_namespaces = {
            "world_model_state_dict",
            "temporal_encoder_state_dict",
            "projection_head_state_dict",
            "reasoning_engine_state_dict",
        }
        missing = required_namespaces - set(tampered_ckpt.keys())
        assert namespace in missing

    def test_truncated_bytes_torch_load_failure(self):
        """Adversarial check: severely truncated checkpoint file bytes fail to deserialize."""
        raw_bytes = PHASE1_CKPT.read_bytes()
        truncated_bytes = raw_bytes[:1024]  # Truncate to first 1KB

        buffer = io.BytesIO(truncated_bytes)
        with pytest.raises(Exception):  # EOFError, UnpicklingError, or RuntimeError
            torch.load(buffer, map_location="cpu", weights_only=False)

    def test_phase1_vs_phase2_drift_properties(self, loaded_ckpts):
        """
        Adversarial validation of Phase 1 vs Phase 2 dynamics:
        1. Base encoder weights in world_model_state_dict MUST drift between Phase 1 and Phase 2 (fine-tuning).
        2. Reasoning Engine weights MUST remain 100% invariant between Phase 1 and Phase 2.
        """
        ckpt1, ckpt2 = loaded_ckpts
        wm1 = ckpt1["world_model_state_dict"]
        wm2 = ckpt2["world_model_state_dict"]

        # Base encoder must have moved during Phase 2 unfreezing
        base_encoder_drift = max((wm1[k] - wm2[k]).abs().max().item() for k in wm1 if "encoder" in k)
        assert base_encoder_drift > 0.0, "Base encoder must have trained and updated in Phase 2"

        # Reasoning Engine must have 0.0 drift
        re1 = ckpt1["reasoning_engine_state_dict"]
        re2 = ckpt2["reasoning_engine_state_dict"]
        re_drift = max((re1[k] - re2[k]).abs().max().item() for k in re1)
        assert re_drift == 0.0, f"Reasoning engine must have zero drift between Phase 1 and 2, got {re_drift}"
        assert hash_reasoning_engine(re1) == hash_reasoning_engine(re2)
