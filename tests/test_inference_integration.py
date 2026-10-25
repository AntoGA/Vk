import math
import pytest
import torch
from ml.inference.batch_processor import BatchProcessor, BatchRequest, collate_inputs
from ml.inference.predictor import InterestPredictorService
from ml.inference.optimizer import ModelOptimizer
from test_model_contract import make_input
from test_regressions import small_model


def test_batch_matches_individual_predictions_and_drains_on_stop():
    model = small_model().eval()
    service = InterestPredictorService(model)
    processor = BatchProcessor(service, batch_size=8, max_wait_ms=100)
    a, b = make_input(1), make_input(1)
    b.action_sequence = b.action_sequence[:, :3]
    b.action_mask = b.action_mask[:, :3]
    expected = [service.predict_batch(x).interest_logits for x in (a, b)]
    callbacks = []
    processor.start()
    processor.start()  # Must not spawn another worker.
    processor.add_request(BatchRequest('a', a, callbacks.append))
    processor.add_request(BatchRequest('b', b))
    processor.stop()  # Flushes partial batch.
    for key, truth in zip(('a', 'b'), expected):
        result = processor.get_result(key)
        torch.testing.assert_close(result.interest_logits, truth, atol=1e-6, rtol=1e-5)
        assert result.confidence_scores.shape == (1,)
    assert len(callbacks) == 1
    assert not processor.worker_thread.is_alive()


def test_worker_propagates_model_error_and_can_restart():
    class Broken:
        def predict_batch(self, inputs):
            raise ValueError('model failed')
    processor = BatchProcessor(Broken(), max_wait_ms=10)
    processor.start()
    processor.add_request(BatchRequest('bad', make_input(1)))
    with pytest.raises(ValueError, match='model failed'):
        processor.get_result('bad')
    processor.stop()
    processor.predictor = InterestPredictorService(small_model())
    processor.start()
    processor.add_request(BatchRequest('ok', make_input(1)))
    assert processor.get_result('ok').interest_logits.shape == (1, 7)
    processor.stop()


def test_duplicate_and_stopped_submissions_rejected():
    processor = BatchProcessor(InterestPredictorService(small_model()), max_wait_ms=10)
    request = BatchRequest('same', make_input(1))
    with pytest.raises(RuntimeError):
        processor.add_request(request)
    processor.start()
    try:
        processor.add_request(request)
        with pytest.raises(ValueError, match='duplicate'):
            processor.add_request(request)
        processor.get_result('same')
    finally:
        processor.stop()


def test_callback_failure_does_not_kill_worker():
    def fail(output):
        raise RuntimeError('callback failed')
    processor = BatchProcessor(InterestPredictorService(small_model()), max_wait_ms=10)
    processor.start()
    try:
        processor.add_request(BatchRequest('one', make_input(1), fail))
        assert processor.get_result('one').interest_logits.shape == (1, 7)
        processor.add_request(BatchRequest('two', make_input(1)))
        assert processor.get_result('two').interest_logits.shape == (1, 7)
    finally:
        processor.stop()


def test_different_graph_sizes_are_rejected():
    a, b = make_input(1), make_input(1)
    b.graph_features = b.graph_features[:, :2]
    b.adjacency_matrix = b.adjacency_matrix[:, :2, :2]
    with pytest.raises(ValueError, match='graph sizes'):
        collate_inputs([a, b])


def test_optimizer_prepares_and_measures_canonical_input():
    model = small_model()
    inputs = make_input(1)
    model.eval()
    with torch.inference_mode():
        expected = model(inputs).interest_logits
    optimizer = ModelOptimizer(model)
    prepared = optimizer.optimize_for_inference(example_input=inputs)
    assert optimizer.optimized
    with torch.inference_mode():
        torch.testing.assert_close(prepared(inputs).interest_logits, expected)
    report = optimizer.benchmark(n_runs=3, warmup=1, example_input=inputs)
    assert report['measured_runs'] == 3
    assert math.isfinite(report['p95_ms']) and report['p95_ms'] >= 0
    # Inference preparation must not disable gradients for later training.
    assert all(p.requires_grad for p in model.parameters())


def test_optimizer_default_example_uses_model_configuration():
    optimizer = ModelOptimizer(small_model())
    optimizer.optimize_for_inference()
    assert optimizer.benchmark(n_runs=1, warmup=0)['measured_runs'] == 1


@pytest.mark.parametrize('runs', [0, -1])
def test_optimizer_invalid_runs(runs):
    with pytest.raises(ValueError):
        ModelOptimizer(small_model()).benchmark(runs)


def test_unsupported_modes_are_explicit():
    optimizer = ModelOptimizer(small_model())
    with pytest.raises(ValueError, match='CUDA'):
        optimizer.optimize_for_inference(use_fp16=True)
    assert not optimizer.optimized
    with pytest.raises(NotImplementedError, match='not validated'):
        optimizer.export_onnx('unused.onnx')
