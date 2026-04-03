# -*- coding: utf-8 -*-
import time, os, psutil
import torch
from dataclasses import dataclass
from typing import List

# =========================
# 설정 (MLP/Conv 코드와 동일한 스타일)
# =========================
DTYPE = torch.float32
DEVICE = "cpu"
BATCH = 1
IMG_H, IMG_W, IMG_C = 224, 224, 3
NUM_THREADS = 4
WARMUP = 10
ITERS = 50

# Depthwise Conv 설정 (고정)
C_IN = IMG_C                 # 입력 채널
CHANNEL_MULTIPLIER = 1       # depthwise: out_ch = C_IN * multiplier
C_OUT = C_IN * CHANNEL_MULTIPLIER
KERNEL = 3
PADDING = 1
STRIDE = 1
USE_BIAS = True
COUNT_RELU_FLOPS = True      # ReLU FLOPs 포함 여부

# =========================
# 유틸
# =========================
def set_threads(n: int):
    os.environ["OMP_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)

def bytes_to_mb(x: int) -> float:
    return x / (1024**2)

def tensor_bytes(n: int, dtype: torch.dtype = torch.float32) -> int:
    return n * torch.tensor([], dtype=dtype).element_size()

# =========================
# 모델 구성
# =========================
def build_depthwise_net(num_layers: int) -> torch.nn.Module:
    """
    (B, C, 224, 224) -> [DW-Conv(groups=C, multiplier=1, k=3) + ReLU] * L
    출력 공간 크기는 유지(224x224).
    """
    layers = []
    for _ in range(num_layers):
        layers.append(
            torch.nn.Conv2d(
                in_channels=C_IN,
                out_channels=C_OUT,
                kernel_size=KERNEL,
                stride=STRIDE,
                padding=PADDING,
                groups=C_IN,           # depthwise
                bias=USE_BIAS,
            )
        )
        layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)

# =========================
# FLOPs / 파라미터
# =========================
def depthwise_per_layer_flops(
    c: int, h: int, w: int, k: int,
    bias: bool = True, count_relu: bool = True, batch: int = 1, multiplier: int = 1
) -> int:
    """
    Depthwise Conv FLOPs (배치 batch):
    - 각 채널은 독립적으로 k*k 커널 적용
    - 출력 요소 수 = batch * (c*multiplier) * h * w
    - MACs per output = 2 * (k*k)
    - Bias per output = +1 (선택)
    - ReLU per output = +1 (선택)
    """
    out_elems = batch * (c * multiplier) * h * w
    flops = out_elems * (2 * k * k)
    if bias:
        flops += out_elems
    if count_relu:
        flops += out_elems
    return flops

def total_flops_per_inference(L: int, batch: int = 1) -> int:
    per = depthwise_per_layer_flops(
        c=C_IN, h=IMG_H, w=IMG_W, k=KERNEL,
        bias=USE_BIAS, count_relu=COUNT_RELU_FLOPS,
        batch=batch, multiplier=CHANNEL_MULTIPLIER
    )
    return per * L

def param_bytes(model: torch.nn.Module) -> int:
    # 일반적으로 depthwise conv 파라미터 수는 c * k * k * multiplier (+ bias)
    return sum(p.numel() * p.element_size() for p in model.parameters())

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
        model = build_depthwise_net(L).to(DEVICE)

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
        print(f"{r.layers:>2d} | {r.flops:>15,d} | {r.time_ms:>10.3f} | {r.gflops:>10.3f} | {r.param_mb:>10.3f} | {r.rss_delta_mb:>10.3f}")

if __name__ == "__main__":
    main()
