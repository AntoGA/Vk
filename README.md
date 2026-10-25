# VK Interest Predictor

Research / proof-of-concept system for multi-label interest prediction with a hybrid Transformer + Graph Attention architecture.

> **Status: research PoC.** The repository currently uses synthetic data for the reproducible E2E path. It has not been validated on real VK user data, at production traffic, or against a production SLA.

## What is implemented

- shared `ModelInput` / `ModelOutput` contract across the model-facing PoC path;
- Transformer encoder for action sequences;
- graph-attention encoder for synthetic social-graph features;
- cross-attention fusion and multi-label output;
- deterministic synthetic-data training/evaluation smoke pipeline;
- checkpoint save/load with model configuration;
- measured inference benchmark that writes a JSON artifact;
- API, dashboard, Docker/Kubernetes and online-learning experiments from earlier iterations of the project.

The last group should be treated as **experimental/integration work**, not evidence of production deployment.

## Reproducible PoC path

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
python -m pip install numpy pytest
# Run from the repository root so scripts can import ml.
export PYTHONPATH="$PWD"        # PowerShell: $env:PYTHONPATH = (Get-Location).Path

python scripts/run_e2e_pipeline.py --device cpu
python scripts/benchmark.py --device cpu
```

The E2E command generates deterministic synthetic tensors, trains a small model, evaluates it, saves/reloads a checkpoint and writes:

```text
artifacts/e2e/model.pt
artifacts/e2e/results.json
```

The benchmark reads that checkpoint and writes:

```text
artifacts/benchmark/results.json
```

Benchmark output contains the commit SHA when Git is available, runtime/software information, device information, warm-up count, measured-run count, latency percentiles and measured throughput.

**There are deliberately no benchmark numbers in this README.** Results depend on the exact commit, checkpoint, hardware and runtime. Publish a number only together with the generated artifact that supports it.

## Architecture

```text
ACTION SEQUENCE ──> projection ──> Transformer ──┐
                                                 ├─> cross-attention ─> pooled representation ─> interests
SOCIAL GRAPH ─────────────────────> GAT ─────────┘                                  ├─> confidence
                                                                                     └─> embedding
```

The shared PoC contract is defined in `ml/contracts.py`:

```python
ModelInput(
    action_sequence,      # [B, T, action_dim]
    action_mask,          # [B, T]
    graph_features,       # [B, N, node_dim]
    adjacency_matrix,     # [B, N, N]
)
```

`InterestPredictor` returns logits, sigmoid probabilities, confidence scores and user embeddings through `ModelOutput`.

## Evaluation scope

The current E2E metrics are useful for checking that the training/evaluation path works on a deterministic synthetic task. They are **not estimates of quality on VK users**.

MAP@K and NDCG@K are not claimed by the reproducible path until their implementation and evaluation protocol are verified. No real-data F1, latency SLA, throughput SLA or accuracy claim is made here.

## Limitations

1. **Synthetic data only** — no validated real VK dataset is included in the reproducible path.
2. **No production validation** — load balancing, autoscaling, failure recovery and sustained traffic have not been demonstrated by the benchmark.
3. **Online learning is experimental** — it requires monitoring, drift controls and rollback before production use.
4. **Legacy/integration surfaces may lag the unified PoC contract** — API/dashboard/deployment code should be validated independently before being presented as deployable.
5. **No fixed hardware claims** — hardware shown by a benchmark comes from the machine that actually ran it.

## Repository map

```text
ml/contracts.py                  shared model-facing contract
ml/models/                       Transformer, GAT and hybrid predictor
ml/training/                     training experiments
ml/inference/                    inference experiments
scripts/run_e2e_pipeline.py      reproducible synthetic PoC
scripts/benchmark.py             measured checkpoint benchmark
api/                             API integration work
dashboard/                       dashboard experiments
k8s/                             deployment manifests (not production-validated)
tests/                           test suite
```

## Evidence standard

For any performance or quality result attached to this project, record at minimum:

- commit SHA;
- dataset/source and split;
- seed and model/checkpoint;
- software/runtime versions;
- hardware/device;
- warm-up and measured-run counts for latency;
- raw machine-readable result artifact.

If those are absent, treat the number as an example or target, not as a measured result.

## Roadmap

- validate the unified contract across API/inference/training integration surfaces;
- expand integration coverage beyond the synthetic PoC tests;
- implement and test MAP@K / NDCG@K if ranking metrics are needed;
- validate against an appropriately licensed/consented real dataset before making real-world quality claims;
- add load testing and operational monitoring before using production terminology.

## License

The project metadata declares MIT; a LICENSE file is not currently included.

## Inference integration validation

`InterestPredictorService`, `BatchProcessor` and the CPU preparation/benchmark
path of `ModelOptimizer` use `ModelInput` / `ModelOutput`.
Submit `BatchRequest(request_id, features=single_item_model_input)` after
`processor.start()`; `get_result(id)` returns a single-item `ModelOutput` or
raises the worker error. `stop()` drains accepted requests; if draining exceeds
five seconds it raises `TimeoutError` and the worker continues draining.
Sequence lengths may differ; graph node counts and feature dimensions must match
within a batch. Mixed graph sizes are rejected because graph-node padding masks
are not implemented. A failed batch propagates the error to every request in it.
Do not mutate submitted inputs while the worker is processing them.

CPU inference preparation and timing are tested. CUDA FP16 and `torch.compile`
remain unvalidated optional paths; compile/warmup failures propagate to the caller.
`export_onnx` explicitly raises `NotImplementedError` until an exporter and
runtime parity tests are implemented. No sub-100ms guarantee is made.

Validation history:
- [Core audit](docs/reports/2026-10-25-pr6-audit.md)
- [Batch/optimizer integration](docs/reports/2026-10-25-inference-integration.md)
