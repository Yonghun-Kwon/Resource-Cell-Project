# -*- coding: utf-8 -*-
import time, os, psutil
import torch
from dataclasses import dataclass
from typing import List

DTYPE = torch.float32
DEVICE = "cpu"
BATCH = 1
IMG_H, IMG_W, IMG_C = 224, 224, 3
NUM_THREADS = 4
WARMUP = 10
ITERS = 50
HIDDEN_DIM = 2048   # 고정 크기

def set_threads(n: int):
    os.environ["OMP_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)

def bytes_to_mb(x: int) -> float:
    return x / (1024**2)

def tensor_bytes(n: int, dtype: torch.dtype = torch.float32) -> int:
    return n * torch.tensor([], dtype=dtype).element_size()

class SliceFirstH(torch.nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.h = h
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = torch.flatten(x, start_dim=1)
        return x[:, :self.h]

def build_mlp(hidden_dim: int, num_layers: int) -> torch.nn.Module:
    layers = [SliceFirstH(hidden_dim)]
    for _ in range(num_layers):
        layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
        layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)

def per_layer_flops(h: int) -> int:
    return 2 * h * h + h  # mul+add + bias

def total_flops(h: int, L: int, batch: int = 1) -> int:
    return per_layer_flops(h) * L * batch

def param_bytes(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())

@dataclass
class Result:
    layers: int
    flops: int
    time_ms: float
    gflops: float
    param_mb: float
    rss_delta_mb: float

def measure_latency(model, x):
    model.eval()
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(x)
        start = time.perf_counter()
        for _ in range(ITERS):
            _ = model(x)
        end = time.perf_counter()
    avg_ms = (end - start) / ITERS * 1000.0
    return avg_ms, end - start

def run() -> List[Result]:
    set_threads(NUM_THREADS)
    x = torch.randn(BATCH, IMG_C, IMG_H, IMG_W, dtype=DTYPE, device=DEVICE)
    results = []
    for L in range(1, 11):
        model = build_mlp(HIDDEN_DIM, L).to(DEVICE)
        rss_before = psutil.Process().memory_info().rss
        avg_ms, total_elapsed = measure_latency(model, x)
        rss_after = psutil.Process().memory_info().rss
        rss_delta = max(0, rss_after - rss_before)
        fl = total_flops(HIDDEN_DIM, L, BATCH)
        gflops = (fl / total_elapsed) / 1e9
        results.append(Result(
            layers=L,
            flops=fl,
            time_ms=avg_ms,
            gflops=gflops,
            param_mb=bytes_to_mb(param_bytes(model)),
            rss_delta_mb=bytes_to_mb(rss_delta)
        ))
    return results

def main():
    results = run()
    print(f"{'L':>2} | {'Actual FLOPs':>15} | {'Time(ms)':>10} | {'GFLOPs/s':>10} | {'Params(MB)':>10} | {'RSSΔ(MB)':>10}")
    print("-"*70)
    for r in results:
        print(f"{r.layers:>2d} | {r.flops:>15,d} | {r.time_ms:>10.3f} | {r.gflops:>10.3f} | {r.param_mb:>10.3f} | {r.rss_delta_mb:>10.3f}")

if __name__ == "__main__":
    main()
