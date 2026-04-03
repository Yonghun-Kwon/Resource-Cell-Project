# -*- coding: utf-8 -*-
import time, os, psutil
import torch
from dataclasses import dataclass
from typing import List

# =========================
# 설정 (MLP 코드와 동일한 스타일)
# =========================
DTYPE = torch.float32
DEVICE = "cpu"
BATCH = 1
IMG_H, IMG_W, IMG_C = 224, 224, 3
NUM_THREADS = 4
WARMUP = 10
ITERS = 50

# Conv 설정 (고정)
C_IN = C_OUT = IMG_C          # 3채널 고정 (입력과 동일하게)
KERNEL = 3
PADDING = 1
STRIDE = 1
USE_BIAS = True
COUNT_RELU_FLOPS = True       # ReLU FLOPs도 카운트(선택)

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
def build_conv_net(num_layers: int) -> torch.nn.Module:
    """
    (B, 3, 224, 224) -> [Conv2d(3->3, k=3, s=1, p=1) + ReLU] * L
    공간 크기(224x224) 유지.
    """
    layers = []
    for _ in range(num_layers):
        layers.append(torch.nn.Conv2d(C_IN, C_OUT, kernel_size=KERNEL,
                                      stride=STRIDE, padding=PADDING, bias=USE_BIAS))
        layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)

# =========================
# FLOPs / 파라미터
# =========================
def conv2d_per_layer_flops(
    cin: int, cout: int, h: int, w: int, k: int,
    bias: bool = True, count_relu: bool = True
) -> int:
    """
    Conv2d FLOPs (배치=1 가정):
    - MACs: cout * H * W * (2 * cin * k * k)
    - bias: + cout * H * W
    - ReLU: + cout * H * W
    """
    mac = cout * h * w * (2 * cin * k * k)
    if bias:
        mac += cout * h * w
    if count_relu:
        mac += cout * h * w
    return mac

def total_flops_per_inference(L: int, batch: int = 1) -> int:
    per = conv2d_per_layer_flops(
        cin=C_IN, cout=C_OUT, h=IMG_H, w=IMG_W, k=KERNEL,
        bias=USE_BIAS, count_relu=COUNT_RELU_FLOPS
    )
    return per * L * batch

def param_bytes(model: torch.nn.Module) -> int:
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
        model = build_conv_net(L).to(DEVICE)

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
