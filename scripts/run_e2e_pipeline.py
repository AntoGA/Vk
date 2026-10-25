#!/usr/bin/env python3
"""Reproducible synthetic-data E2E smoke pipeline (research/PoC only)."""
import argparse, json, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from ml.contracts import ModelInput
from ml.models.interest_predictor import InterestPredictor, ModelConfig


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def make_data(n, c, seed):
    g = torch.Generator().manual_seed(seed); t, nodes = 24, 12
    actions = torch.randn(n, t, c.action_dim, generator=g)
    graph = torch.randn(n, nodes, c.node_dim, generator=g)
    adj = (torch.rand(n, nodes, nodes, generator=g) > .7).float()
    adj = torch.maximum(adj, torch.eye(nodes).unsqueeze(0))
    labels = torch.zeros(n, c.n_interests)
    signal = actions[:, :, :min(c.n_interests, c.action_dim)].mean(1)
    labels[:, :signal.shape[1]] = (signal > 0).float()
    return ModelInput(actions, torch.ones(n, t, dtype=torch.bool), graph, adj), labels


def take(x, idx, device):
    return ModelInput(x.action_sequence[idx].to(device), x.action_mask[idx].to(device), x.graph_features[idx].to(device), x.adjacency_matrix[idx].to(device))


def score(probs, labels):
    pred = (probs >= .5).float(); tp = (pred * labels).sum().item()
    p = tp / max(pred.sum().item(), 1.); r = tp / max(labels.sum().item(), 1.)
    return {"precision": p, "recall": r, "f1": 2*p*r/max(p+r, 1e-12)}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--num-users", type=int, default=256); p.add_argument("--epochs", type=int, default=3); p.add_argument("--batch-size", type=int, default=32); p.add_argument("--seed", type=int, default=42); p.add_argument("--device", choices=["cpu","cuda"], default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--output-dir", default="artifacts/e2e"); a = p.parse_args()
    if a.num_users < 2 or a.epochs < 1 or a.batch_size < 1:
        p.error("num-users must be >= 2; epochs and batch-size must be positive")
    if a.device == "cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA requested but unavailable")
    seed_all(a.seed); device = torch.device(a.device); config = ModelConfig(); x, y = make_data(a.num_users, config, a.seed)
    split = int(a.num_users*.8); train = torch.arange(split); test = torch.arange(split, a.num_users)
    model = InterestPredictor(config).to(device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(a.epochs):
        model.train()
        for s in range(0, len(train), a.batch_size):
            idx=train[s:s+a.batch_size]; batch=take(x,idx,device); target=y[idx].to(device); opt.zero_grad(); out=model(batch); loss=F.binary_cross_entropy_with_logits(out.interest_logits,target); loss.backward(); opt.step()
    outdir=Path(a.output_dir); outdir.mkdir(parents=True,exist_ok=True); ckpt=outdir/"model.pt"; torch.save(model.checkpoint_payload(),ckpt)
    model=InterestPredictor.load_from_checkpoint(str(ckpt),map_location=device).to(device).eval(); test_input=take(x,test,device)
    with torch.no_grad(): quality=score(model(test_input).interest_probabilities.cpu(),y[test])
    one=take(x,test[:1],device)
    for _ in range(10):
        with torch.no_grad(): model(one)
    lat=[]
    for _ in range(50):
        if device.type=="cuda": torch.cuda.synchronize()
        t0=time.perf_counter()
        with torch.no_grad(): model(one)
        if device.type=="cuda": torch.cuda.synchronize()
        lat.append((time.perf_counter()-t0)*1000)
    result={"status":"synthetic-poc","seed":a.seed,"device":str(device),"num_users":a.num_users,"epochs":a.epochs,"test_samples":len(test),"quality":quality,"latency_ms":{"p50":float(np.percentile(lat,50)),"p95":float(np.percentile(lat,95)),"p99":float(np.percentile(lat,99))},"checkpoint":str(ckpt)}
    (outdir/"results.json").write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
