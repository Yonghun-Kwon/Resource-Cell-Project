# -*- coding: utf-8 -*-
import time, os, psutil
import torch
from dataclasses import dataclass
from typing import List

# =========================
# 설정 (MLP 코드와 동일한 출력 포맷)
# =========================
DTYPE = torch.float32
DEVICE = "cpu"
BATCH = 1
IMG_H, IMG_W, IMG_C = 224, 224, 3
NUM_THREADS = 4
WARMUP = 10
ITERS = 50

# Pointwise Conv(1x1) 설정
C_IN = IMG_C
C_OUT = IMG_C          # 필요시 변경 가능 (예: 채널 확장/축소)
KERNEL = 1
STRIDE = 1
PADDING = 0
USE_BIAS = True
COUNT_RELU_FLOPS = True  # ReLU 연산도 FLOPs에 포함할지

# =========================
# 유틸
# =========================
def set_threads(n: int):
    os.environ["OMP_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)

def bytes_to_mb(x: int) -> float:
    return x / (1024**2)

def param_bytes(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())

# =========================
# 모델 구성
# =========================
def build_pointwise_net(num_layers: int) -> torch.nn.Module:
    """
    (B, C_in=3, 224, 224) -> [Conv1x1(C_in->C_out) + ReLU] * L
    공간 크기는 유지(224x224).
    """
    layers = []
    for _ in range(num_layers):
        layers.append(
            torch.nn.Conv2d(
                in_channels=C_IN,
                out_channels=C_OUT,
                kernel_size=KERNEL,   # 1x1
                stride=STRIDE,
                padding=PADDING,
                bias=USE_BIAS
            )
        )
        layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)

# =========================
# FLOPs / 파라미터
# =========================
def pointwise_per_layer_flops(
    cin: int, cout: int, h: int, w: int, bias: bool = True, count_relu: bool = True, batch: int = 1
) -> int:
    """
    1x1 Conv FLOPs (batch=B):
    - per-output FLOPs = 2 * cin  (mul+add, K=1)
    - 출력 요소 수 = B * cout * h * w
    - 총 FLOPs = B * cout * h * w * (2*cin) + (bias ? +B*cout*h*w : 0) + (ReLU ? +B*cout*h*w : 0)
    """
    out_elems = batch * cout * h * w
    flops = out_elems * (2 * cin)
    if bias:
        flops += out_elems
    if count_relu:
        flops += out_elems
    return flops

def total_flops_per_inference(L: int, batch: int = 1) -> int:
    per = pointwise_per_layer_flops(
        cin=C_IN, cout=C_OUT, h=IMG_H, w=IMG_W,
        bias=USE_BIAS, count_relu=COUNT_RELU_FLOPS, batch=batch
    )
    return per * L

# =========================
# 측정 루틴
# =========================
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
        model = build_pointwise_net(L).to(DEVICE)

        rss_before = psutil.Process().memory_info().rss
        avg_ms, total_elapsed = measure_latency(model, x)
        rss_after = psutil.Process().memory_info().rss
        rss_delta = max(0, rss_after - rss_before)

        fl = total_flops_per_inference(L, BATCH)
        gflops = (fl / total_elapsed) / 1e9 if total_elapsed > 0 else 0.0

        results.append(
            Result(
                layers=L,
                flops=fl,
                time_ms=avg_ms,
                gflops=gflops,
                param_mb=bytes_to_mb(param_bytes(model)),
                rss_delta_mb=bytes_to_mb(rss_delta),
            )
        )
    return results

# =========================
# 메인
# =========================
def main():
    results = run()
    print(f"{'L':>2} | {'Actual FLOPs':>15} | {'Time(ms)':>10} | {'GFLOPs/s':>10} | {'Params(MB)':>10} | {'RSSΔ(MB)':>10}")
    print("-"*70)
    for r in results:
        print(f"{r.layers:>2d} | {r.flops:>15,d} | {r.time_ms:>10.3f} | {r.gflops:>10.3f} | {r.param_mb:>10.6f} | {r.rss_delta_mb:>10.6f}")

if __name__ == "__main__":
    main()
