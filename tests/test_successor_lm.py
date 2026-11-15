import torch

from snnformer import SNNConfig
from snnformer.successor_lm import SuccessorDendriteLM


def test_successor_model_is_causal_and_streams_the_same_distribution():
    cfg = SNNConfig(attn="transformer_successor", local_window=5, successor_orders=(2, 3, 5),
                    successor_slots=16, successor_scaled_votes=True)
    model = SuccessorDendriteLM(cfg, dim=16, heads=2, depth=2).eval()
    tokens = torch.tensor([list(b"ab cab dab cab dab cab"), list(b"de fde gde fde gde fde")])
    with torch.no_grad():
        expected = model(tokens)
        modified = tokens.clone()
        modified[:, 12:] = 255
        torch.testing.assert_close(model(modified)[:, :12], expected[:, :12])
        torch.testing.assert_close(expected.exp().sum(-1), torch.ones_like(expected[..., 0]))
        state, streamed = None, []
        for token in tokens.unbind(1):
            logits, state = model.step(token, state)
            streamed.append(logits)
        torch.testing.assert_close(torch.stack(streamed, dim=1), expected, atol=2e-6, rtol=1e-5)
