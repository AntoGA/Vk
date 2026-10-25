import tempfile

import torch

from ml.contracts import ModelInput
from ml.models.interest_predictor import InterestPredictor, ModelConfig


def make_input(batch_size=2):
    return ModelInput(
        action_sequence=torch.randn(batch_size, 6, 8),
        action_mask=torch.ones(batch_size, 6, dtype=torch.bool),
        graph_features=torch.randn(batch_size, 4, 5),
        adjacency_matrix=torch.eye(4).repeat(batch_size, 1, 1),
    )


def test_forward_contract_shapes():
    config = ModelConfig(n_interests=7, action_dim=8, node_dim=5, d_model=16, n_heads=4, n_layers=1, gat_hidden_dim=8, gat_n_layers=2, gat_n_heads=2, embedding_dim=6)
    model = InterestPredictor(config).eval()
    inputs = make_input()
    output = model(inputs)
    output.validate()
    assert output.interest_logits.shape == (2, 7)
    assert output.interest_probabilities.shape == (2, 7)
    assert output.confidence_scores.shape == (2,)
    assert output.user_embeddings.shape == (2, 6)


def test_checkpoint_round_trip():
    config = ModelConfig(n_interests=7, action_dim=8, node_dim=5, d_model=16, n_heads=4, n_layers=1, gat_hidden_dim=8, gat_n_layers=2, gat_n_heads=2, embedding_dim=6)
    model = InterestPredictor(config).eval()
    inputs = make_input(1)
    with torch.inference_mode():
        before = model(inputs).interest_logits
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save({"model_state_dict": model.state_dict()}, f.name)
        restored = InterestPredictor(config).eval()
        restored.load_state_dict(torch.load(f.name, map_location="cpu")["model_state_dict"])
        with torch.inference_mode():
            after = restored(inputs).interest_logits
    assert torch.allclose(before, after)
