import subprocess
import sys

import pytest
import torch

from ml.contracts import TrainingBatch
from ml.models.interest_predictor import InterestPredictor, ModelConfig
from ml.training import Trainer, OnlineLearner
from test_model_contract import make_input


def small_model():
    return InterestPredictor(ModelConfig(n_interests=7, action_dim=8, node_dim=5,
        d_model=16, n_heads=4, n_layers=1, gat_hidden_dim=8,
        gat_n_layers=2, gat_n_heads=2, embedding_dim=6, dropout=0))


def test_padded_batch_retains_sequence_length():
    model = small_model().eval()
    x = make_input()
    x.action_mask[:, 3:] = False
    with torch.inference_mode():
        original = model(x).interest_logits
        x.action_sequence[:, 3:] = 10000
        changed = model(x).interest_logits
    assert torch.isfinite(original).all()
    torch.testing.assert_close(original, changed)


@pytest.mark.parametrize('kind', ['empty', 'masked', 'float_mask'])
def test_invalid_inputs_rejected(kind):
    x = make_input()
    if kind == 'empty':
        x.action_sequence = x.action_sequence[:, :0]
    elif kind == 'masked':
        x.action_mask[:] = False
    else:
        x.action_mask = x.action_mask.float()
    with pytest.raises(ValueError):
        x.validate()


def test_trainer_checkpoint_loadable_for_inference(tmp_path):
    model = small_model()
    trainer = Trainer(model, torch.optim.Adam(model.parameters()),
        torch.nn.BCEWithLogitsLoss(), checkpoint_dir=str(tmp_path))
    batch = TrainingBatch(make_input(), torch.ones(2, 7))
    assert trainer.train_epoch([batch])['loss'] > 0
    path = trainer.save_checkpoint('model.pt')
    restored = InterestPredictor.load_from_checkpoint(str(path)).eval()
    model.eval()
    with torch.inference_mode():
        torch.testing.assert_close(model(batch.inputs).interest_logits,
                                   restored(batch.inputs).interest_logits)
    trainer.load_checkpoint(str(path))
    assert trainer.global_step == 1


def test_online_update_changes_weights():
    model = small_model()
    learner = OnlineLearner(model, batch_size=2, update_frequency=2)
    before = model.interest_head[-1].weight.detach().clone()
    learner.add_sample(make_input(1), torch.ones(7))
    learner.add_sample(make_input(1), torch.zeros(7))
    assert learner.total_updates == 1
    assert not torch.equal(before, model.interest_head[-1].weight)


@pytest.mark.parametrize('script,args', [
    ('benchmark.py', ['--num-runs', '0']),
    ('benchmark.py', ['--warmup', '-1']),
    ('run_e2e_pipeline.py', ['--num-users', '1']),
    ('run_e2e_pipeline.py', ['--epochs', '0']),
])
def test_cli_rejects_invalid_arguments(script, args):
    result = subprocess.run([sys.executable, 'scripts/' + script, *args],
        capture_output=True, text=True)
    assert result.returncode == 2
    assert 'error:' in result.stderr


def test_graph_attention_matches_explicit_neighbor_sum():
    from ml.models.graph_attention import GraphAttentionLayer
    layer = GraphAttentionLayer(3, 2, n_heads=2, dropout=0, concat=False).eval()
    x = torch.randn(2, 3, 3)
    adj = torch.eye(3).repeat(2, 1, 1)
    adj[:, 0, 1] = 1
    expected = torch.zeros(2, 3, 2)
    for b in range(2):
        for h in range(2):
            transformed = x[b] @ layer.W[h]
            for i in range(3):
                scores = torch.stack([layer.leaky_relu(
                    torch.cat((transformed[i], transformed[j])) @ layer.a[h]
                ).squeeze() for j in range(3)])
                weights = scores.masked_fill(adj[b, i] == 0, -9e15).softmax(0)
                expected[b, i] += (weights[:, None] * transformed).sum(0) / 2
    torch.testing.assert_close(layer(x, adj), layer.elu(expected))


def test_single_result_service_rejects_multi_item_batch():
    from ml.inference.predictor import InterestPredictorService
    service = InterestPredictorService(small_model())
    with pytest.raises(ValueError):
        service.predict_input(make_input(2))


def test_cache_uses_input_and_invalidates_on_checkpoint_load(tmp_path):
    from ml.inference.predictor import InterestPredictorService
    service = InterestPredictorService(small_model(), cache_enabled=True)
    x = make_input(1)
    service.predict(x, user_id=42)
    assert len(service.cache) == 1
    assert service.predict(x, user_id=42).latency_ms == 0
    x.action_sequence += 1
    service.predict(x, user_id=42)
    assert len(service.cache) == 2
    path = tmp_path / 'model.pt'
    torch.save(service.model.checkpoint_payload(), path)
    service.load_checkpoint(str(path))
    assert not service.cache
