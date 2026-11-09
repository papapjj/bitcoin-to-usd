import numpy as np
import torch

from scripts.audit_temporal_head_bottleneck import collect
from scripts.train_temporal_v3_ablation_candidate import _forward_window_batch
from self_growing_trader.five_shared_policy import FactorizedActionHeads, PostLstmCrossAgentTransformer


def test_bottleneck_rows_match_production_role_logits_after_burn_in():
    torch.manual_seed(9)
    network = torch.nn.Module()
    network.hidden_size = 8
    network.post_lstm_transformer = PostLstmCrossAgentTransformer(8, model_width=8)
    network.heads = FactorizedActionHeads(8)
    network.eval()
    hidden = torch.randn(1, 130, 5, 8)
    sampled = {"actions": np.zeros((1, 130, 5), dtype=np.int64),
               "owner_masks": np.ones((1, 130, 5), dtype=bool)}
    sampled["actions"][0, 128, 0] = 1
    sampled["actions"][0, 129, 1] = 2
    rows = collect(network, hidden, sampled)
    with torch.inference_mode():
        logits, labels, roles = _forward_window_batch(
            network, hidden, sampled, np.array([0]), burn_in=128
        )
    for role in (0, 1):
        selected = roles.numpy() == role
        np.testing.assert_array_equal(rows[str(role)]["labels"], labels.numpy()[selected])
        np.testing.assert_allclose(rows[str(role)]["margin"],
                                   (logits[:, 1] - logits[:, 0]).numpy()[selected], atol=1e-6)
        np.testing.assert_allclose(rows[str(role)]["raw_lstm"], hidden[0, 128:, role].numpy())
