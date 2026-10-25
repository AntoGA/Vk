#!/usr/bin/env python3
"""Measure checkpoint inference performance and emit a machine-readable report.

No benchmark numbers are hard-coded. Quality metrics are intentionally not reported
here; use E2E/evaluation artifacts produced from an explicit dataset instead.
"""
import argparse, json, platform, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from ml.contracts import ModelInput
from ml.models.interest_predictor import InterestPredictor


def git_sha():
    try: return subprocess.check_output(["git","rev-parse","HEAD"], text=True).strip()
    except Exception: return None


def dummy(config, batch, device, seed):
    g=torch.Generator().manual_seed(seed); t,n=24,12
    x=ModelInput(torch.randn(batch,t,config.action_dim,generator=g),torch.ones(batch,t,dtype=torch.bool),torch.randn(batch,n,config.node_dim,generator=g),(torch.rand(batch,n,n,generator=g)>.7).float())
    x.adjacency_matrix=torch.maximum(x.adjacency_matrix,torch.eye(n).unsqueeze(0)); return x.to(device)


def main():
    p=argparse.ArgumentParser(); p.add_argument("--model",default="artifacts/e2e/model.pt"); p.add_argument("--device",choices=["cpu","cuda"],default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--num-runs",type=int,default=100); p.add_argument("--warmup",type=int,default=10); p.add_argument("--batch-size",type=int,default=1); p.add_argument("--seed",type=int,default=42); p.add_argument("--output",default="artifacts/benchmark/results.json"); a=p.parse_args()
    if a.num_runs < 1 or a.warmup < 0 or a.batch_size < 1:
        p.error("num-runs and batch-size must be positive; warmup must be nonnegative")
    if a.device=="cuda" and not torch.cuda.is_available(): raise SystemExit("CUDA requested but unavailable")
    device=torch.device(a.device); model=InterestPredictor.load_from_checkpoint(a.model,map_location=device).to(device).eval(); x=dummy(model.config,a.batch_size,device,a.seed)
    for _ in range(a.warmup):
        with torch.no_grad(): model(x)
    lat=[]
    for _ in range(a.num_runs):
        if device.type=="cuda": torch.cuda.synchronize()
        t0=time.perf_counter()
        with torch.no_grad(): model(x)
        if device.type=="cuda": torch.cuda.synchronize()
        lat.append((time.perf_counter()-t0)*1000)
    arr=np.asarray(lat); elapsed=arr.sum()/1000
    report={"schema_version":1,"timestamp":datetime.now(timezone.utc).isoformat(),"commit_sha":git_sha(),"checkpoint":a.model,"seed":a.seed,"device":str(device),"batch_size":a.batch_size,"warmup_runs":a.warmup,"measured_runs":a.num_runs,"latency_ms":{"mean":float(arr.mean()),"p50":float(np.percentile(arr,50)),"p95":float(np.percentile(arr,95)),"p99":float(np.percentile(arr,99))},"throughput_items_per_sec":float((a.num_runs*a.batch_size)/elapsed),"model":{"parameters":model.get_num_params()},"system":{"platform":platform.platform(),"python":platform.python_version(),"torch":torch.__version__,"cuda_runtime":torch.version.cuda,"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},"quality_metrics":None,"note":"Synthetic-input inference benchmark; not a production SLA."}
    path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(report,indent=2),encoding="utf-8"); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
