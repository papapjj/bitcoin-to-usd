from __future__ import annotations

import numpy as np

from self_growing_trader.child_supernetwork_manager import ChildSupernetworkManager
from self_growing_trader.m1_graph_module import M1GraphModule
from self_growing_trader.self_growing_runner import SelfGrowingPolicy


def test_child_manager_spawns_nonclone_warms_and_requires_mt5_gate(tmp_path):
    policy = SelfGrowingPolicy.seed(input_size=5, output_size=3, hidden_size=5, seed=501)
    policy.convert_m1_to_graph_backend()
    parent = policy.modules[0].m1_component.graph
    manager = ChildSupernetworkManager.root(network_id="root", max_children=2, max_generation=2)
    manager.on_root_promotion()

    journal = tmp_path / "journal.jsonl"
    journal.write_text(
        '{"memory_key_schema":6,"reward":1.0,"td_delta":0.5,"trace":{"execution_details":{"memory_eligible":true},"active_edge_ids":[],"active_neuron_ids":[]}}\n',
        encoding="utf-8",
    )
    child = manager.spawn_candidate(
        parent_graph=parent,
        input_size=5,
        output_size=3,
        seed=42,
        journal_path=journal,
    )
    assert child is not None
    assert child.network.graph is not parent
    assert child.warmup_rows == 1
    assert not child.selectable

    assert not manager.validate_child(
        child.network_id,
        replay_validator=lambda _graph: 1.0,
        mt5_validator=lambda _graph: (True, 2.0, False),
    )
    assert child.status == "rejected"


def test_policy_spawns_candidate_child_only_after_promoted_graph_trial():
    policy = SelfGrowingPolicy.seed(input_size=5, output_size=3, hidden_size=5, seed=502)
    policy.convert_m1_to_graph_backend()
    policy.graph_supernetwork_config.update({
        "enable_child_supernet": True,
        "max_child_networks": 2,
        "max_generation": 2,
        "max_children_per_network": 2,
        "diversity_threshold": 0.01,
    })
    policy.configure_graph_supernetwork(type("Cfg", (), policy.graph_supernetwork_config)())
    candidate = policy.propose(np.random.default_rng(7), mutation_sigma=0.03, max_modules=3)
    candidate.record_graph_candidate_outcome(promoted=True)

    assert candidate.child_supernetwork_manager is not None
    assert candidate.child_supernetwork_manager.children
    child = next(iter(candidate.child_supernetwork_manager.children.values()))
    assert child.status == "candidate"
    selection = candidate.select_graph_policy(baseline_confidence=0.5, baseline_utility=0.1)
    assert selection["selected_policy_id"] == "m1"
    assert selection["used_fallback"] is False or selection["reason"] in {"baseline", "best_score"}
