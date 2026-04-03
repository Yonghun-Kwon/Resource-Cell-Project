# -*- coding: utf-8 -*-
"""
네 가지 연산(MLP / Conv3x3 / Pointwise1x1 / Depthwise3x3)을 순차 실행.
- 각 측정에서 ITERS번 중 '가장 빠른 절반'만 평균
- 실제 FLOPs, 이론적 메모리 접근 Bytes, OI(FLOPs/Byte) 계산
- Pointwise: per-layer 4배(블록 4회)
- Depthwise: per-layer 2배(블록 2회)
Python 3.7+ 호환 (typing.Union 사용)
"""

import time, os, psutil
import torch
from dataclasses import dataclass
from typing import List, Tuple, Union

# =========================
# 공통 설정
# =========================
DTYPE = torch.float32
DEVICE = "cpu"
BATCH = 1
IMG_H, IMG_W, IMG_C = 224, 224, 3
NUM_THREADS = 4
WARMUP = 10
ITERS = 50
DT_SIZE = torch.tensor([], dtype=DTYPE).element_size()  # float32 -> 4 bytes

# -------------------------
# Conv 설정 (3x3)
# -------------------------
CIN_CONV = IMG_C
COUT_CONV = IMG_C
K_CONV = 3
P_CONV = 1
S_CONV = 1
USE_BIAS_CONV = True
COUNT_RELU_CONV = True

# -------------------------
# Pointwise 설정 (1x1) — per-layer 4배 연산량
# -------------------------
C_PW = IMG_C                 # 모든 블록 Cin=Cout=C_PW
K_PW = 1
P_PW = 0
S_PW = 1
USE_BIAS_PW = True
COUNT_RELU_PW = True
PW_REPEAT_PER_LAYER = 4      # 레이어 내부 4회 반복

# -------------------------
# Depthwise 설정 (3x3, multiplier=1) — per-layer 2배 연산량
# -------------------------
C_DW = IMG_C
MULT_DW = 1
K_DW = 3
P_DW = 1
S_DW = 1
USE_BIAS_DW = True
COUNT_RELU_DW = True
DW_REPEAT_PER_LAYER = 2      # 레이어 내부 2회 반복

# -------------------------
# MLP 설정
# -------------------------
HIDDEN_DIM = 2048  # 고정

# =========================
# 공통 유틸
# =========================
def set_threads(n: int):
    os.environ["OMP_NUM_THREADS"] = str(n)
    torch.set_num_threads(n)

def bytes_to_mb(x: Union[int, float]) -> float:
    return float(x) / (1024**2)

def param_bytes(model: torch.nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())

@dataclass
class Row:
    layers: int
    flops: int
    bytes_used: int
    oi_flops_per_byte: float
    time_ms: float
    gflops: float
    param_mb: float
    rss_delta_mb: float

def measure_latency_fastest_half(model: torch.nn.Module, x: torch.Tensor) -> Tuple[float, float]:
    """
    각 반복의 개별 시간 측정 → 가장 빠른 절반만 골라 평균(초)과 ms 반환.
    """
    model.eval()
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(x)
        it_times: List[float] = []
        for _ in range(ITERS):
            t0 = time.perf_counter()
            _ = model(x)
            t1 = time.perf_counter()
            it_times.append(t1 - t0)
    it_times.sort()
    k = max(1, len(it_times) // 2)
    fastest = it_times[:k]
    avg_s = sum(fastest) / k
    return avg_s * 1000.0, avg_s  # (ms, s)

def print_table(title: str, rows: List[Row]):
    print(f"\n=== {title} ===")
    print(f"{'L':>2} | {'Actual FLOPs':>15} | {'Bytes':>12} | {'OI(F/B)':>10} | {'Time(ms)':>10} | {'GFLOPs/s':>10} | {'Params(MB)':>10} | {'RSSΔ(MB)':>10}")
    print("-"*98)
    for r in rows:
        print(
            f"{r.layers:>2d} | "
            f"{r.flops:>15,d} | "
            f"{r.bytes_used:>12,d} | "
            f"{r.oi_flops_per_byte:>10.3f} | "
            f"{r.time_ms:>10.3f} | "
            f"{r.gflops:>10.3f} | "
            f"{r.param_mb:>10.6f} | "
            f"{r.rss_delta_mb:>10.6f}"
        )

# =========================
# 1) MLP
# =========================
class SliceFirstH(torch.nn.Module):
    def __init__(self, h: int):
        super().__init__()
        self.h = h
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = torch.flatten(x, start_dim=1)
        return x[:, :self.h]

def build_mlp(num_layers: int) -> torch.nn.Module:
    layers = [SliceFirstH(HIDDEN_DIM)]
    for _ in range(num_layers):
        layers.append(torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM, bias=True))
        layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)

def mlp_per_layer_flops(h: int, count_relu: bool = True, bias: bool = True) -> int:
    fl = 2*h*h
    if bias: fl += h
    if count_relu: fl += h
    return fl

def mlp_per_layer_bytes(h: int, count_relu: bool = True, bias: bool = True) -> int:
    elems = h*h + h + h + (h if bias else 0) + (2*h if count_relu else 0)
    return elems * DT_SIZE

def run_mlp(x: torch.Tensor) -> List[Row]:
    rows = []
    per_flops = mlp_per_layer_flops(HIDDEN_DIM, count_relu=True, bias=True)
    per_bytes = mlp_per_layer_bytes(HIDDEN_DIM, count_relu=True, bias=True)
    for L in range(1, 11):
        model = build_mlp(L).to(DEVICE)
        rss_before = psutil.Process().memory_info().rss
        avg_ms, avg_s = measure_latency_fastest_half(model, x)
        rss_after = psutil.Process().memory_info().rss
        rss_delta = max(0, rss_after - rss_before)

        flops_total = per_flops * L * BATCH
        bytes_total = per_bytes * L * BATCH
        oi = flops_total / bytes_total if bytes_total > 0 else 0.0
        gflops = (flops_total / avg_s) / 1e9 if avg_s > 0 else 0.0

        rows.append(Row(
            layers=L,
            flops=flops_total,
            bytes_used=bytes_total,
            oi_flops_per_byte=oi,
            time_ms=avg_ms,
            gflops=gflops,
            param_mb=bytes_to_mb(param_bytes(model)),
            rss_delta_mb=bytes_to_mb(rss_delta),
        ))
    return rows

# =========================
# 2) Conv (3x3)
# =========================
def build_conv_net(num_layers: int) -> torch.nn.Module:
    layers = []
    for _ in range(num_layers):
        layers.append(torch.nn.Conv2d(CIN_CONV, COUT_CONV, kernel_size=K_CONV, stride=S_CONV, padding=P_CONV, bias=USE_BIAS_CONV))
        layers.append(torch.nn.ReLU())
    return torch.nn.Sequential(*layers)

def conv2d_per_block_flops(cin: int, cout: int, h: int, w: int, k: int, bias: bool=True, relu: bool=True, batch: int=1) -> int:
    out_elems = batch * cout * h * w
    fl = out_elems * (2 * cin * k * k)
    if bias: fl += out_elems
    if relu: fl += out_elems
    return fl

def conv2d_per_block_bytes(cin: int, cout: int, h: int, w: int, k: int, bias: bool=True, relu: bool=True, batch: int=1) -> int:
    read_w = cout * cin * k * k
    read_x = batch * cin * h * w
    write_y = batch * cout * h * w
    read_bias = (batch * cout * h * w) if bias else 0
    relu_rw = (2 * batch * cout * h * w) if relu else 0
    elems = read_w + read_x + write_y + read_bias + relu_rw
    return elems * DT_SIZE

def run_conv(x: torch.Tensor) -> List[Row]:
    rows = []
    per_flops = conv2d_per_block_flops(CIN_CONV, COUT_CONV, IMG_H, IMG_W, K_CONV, USE_BIAS_CONV, COUNT_RELU_CONV, BATCH)
    per_bytes = conv2d_per_block_bytes(CIN_CONV, COUT_CONV, IMG_H, IMG_W, K_CONV, USE_BIAS_CONV, COUNT_RELU_CONV, BATCH)
    for L in range(1, 11):
        model = build_conv_net(L).to(DEVICE)
        rss_before = psutil.Process().memory_info().rss
        avg_ms, avg_s = measure_latency_fastest_half(model, x)
        rss_after = psutil.Process().memory_info().rss
        rss_delta = max(0, rss_after - rss_before)

        flops_total = per_flops * L
        bytes_total = per_bytes * L
        oi = flops_total / bytes_total if bytes_total > 0 else 0.0
        gflops = (flops_total / avg_s) / 1e9 if avg_s > 0 else 0.0

        rows.append(Row(
            layers=L,
            flops=flops_total,
            bytes_used=bytes_total,
            oi_flops_per_byte=oi,
            time_ms=avg_ms,
            gflops=gflops,
            param_mb=bytes_to_mb(param_bytes(model)),
            rss_delta_mb=bytes_to_mb(rss_delta),
        ))
    return rows

# =========================
# 3) Pointwise (1x1) — per-layer 4배 연산량
# =========================
class PointwiseLayer(torch.nn.Module):
    """Conv1x1+ReLU 블록을 PW_REPEAT_PER_LAYER 번 반복 (Cin=Cout=C_PW)."""
    def __init__(self):
        super().__init__()
        blocks = []
        for _ in range(PW_REPEAT_PER_LAYER):
            blocks.append(torch.nn.Conv2d(C_PW, C_PW, kernel_size=K_PW, stride=S_PW, padding=P_PW, bias=USE_BIAS_PW))
            blocks.append(torch.nn.ReLU())
        self.seq = torch.nn.Sequential(*blocks)
    def forward(self, x):
        return self.seq(x)

def build_pointwise_net(num_layers: int) -> torch.nn.Module:
    layers = []
    for _ in range(num_layers):
        layers.append(PointwiseLayer())
    return torch.nn.Sequential(*layers)

def pointwise_block_flops(cin: int, cout: int, h: int, w: int, bias: bool=True, relu: bool=True, batch: int=1) -> int:
    out_elems = batch * cout * h * w
    fl = out_elems * (2 * cin)
    if bias: fl += out_elems
    if relu: fl += out_elems
    return fl

def pointwise_block_bytes(cin: int, cout: int, h: int, w: int, bias: bool=True, relu: bool=True, batch: int=1) -> int:
    read_w = cout * cin
    read_x = batch * cin * h * w
    write_y = batch * cout * h * w
    read_bias = (batch * cout * h * w) if bias else 0
    relu_rw = (2 * batch * cout * h * w) if relu else 0
    elems = read_w + read_x + write_y + read_bias + relu_rw
    return elems * DT_SIZE

def run_pointwise(x: torch.Tensor) -> List[Row]:
    rows = []
    per_block_flops = pointwise_block_flops(C_PW, C_PW, IMG_H, IMG_W, USE_BIAS_PW, COUNT_RELU_PW, BATCH)
    per_block_bytes = pointwise_block_bytes(C_PW, C_PW, IMG_H, IMG_W, USE_BIAS_PW, COUNT_RELU_PW, BATCH)
    per_layer_flops = per_block_flops * PW_REPEAT_PER_LAYER
    per_layer_bytes = per_block_bytes * PW_REPEAT_PER_LAYER

    for L in range(1, 11):
        model = build_pointwise_net(L).to(DEVICE)
        rss_before = psutil.Process().memory_info().rss
        avg_ms, avg_s = measure_latency_fastest_half(model, x)
        rss_after = psutil.Process().memory_info().rss
        rss_delta = max(0, rss_after - rss_before)

        flops_total = per_layer_flops * L
        bytes_total = per_layer_bytes * L
        oi = flops_total / bytes_total if bytes_total > 0 else 0.0
        gflops = (flops_total / avg_s) / 1e9 if avg_s > 0 else 0.0

        rows.append(Row(
            layers=L,
            flops=flops_total,
            bytes_used=bytes_total,
            oi_flops_per_byte=oi,
            time_ms=avg_ms,
            gflops=gflops,
            param_mb=bytes_to_mb(param_bytes(model)),
            rss_delta_mb=bytes_to_mb(rss_delta),
        ))
    return rows

# =========================
# 4) Depthwise (3x3) — per-layer 2배 연산량
# =========================
class DepthwiseLayer(torch.nn.Module):
    """DW-Conv3x3+ReLU 블록을 DW_REPEAT_PER_LAYER 번 반복 (Cin=C_DW, groups=C_DW)."""
    def __init__(self):
        super().__init__()
        blocks = []
        for _ in range(DW_REPEAT_PER_LAYER):
            blocks.append(torch.nn.Conv2d(C_DW, C_DW, kernel_size=K_DW, stride=S_DW, padding=P_DW, groups=C_DW, bias=USE_BIAS_DW))
            blocks.append(torch.nn.ReLU())
        self.seq = torch.nn.Sequential(*blocks)
    def forward(self, x):
        return self.seq(x)

def build_depthwise_net(num_layers: int) -> torch.nn.Module:
    layers = []
    for _ in range(num_layers):
        layers.append(DepthwiseLayer())
    return torch.nn.Sequential(*layers)

def depthwise_block_flops(c: int, h: int, w: int, k: int, bias: bool=True, relu: bool=True, batch: int=1, mult: int=1) -> int:
    out_elems = batch * (c * mult) * h * w
    fl = out_elems * (2 * k * k)
    if bias: fl += out_elems
    if relu: fl += out_elems
    return fl

def depthwise_block_bytes(c: int, h: int, w: int, k: int, bias: bool=True, relu: bool=True, batch: int=1, mult: int=1) -> int:
    read_w = c * mult * k * k
    read_x = batch * c * h * w
    write_y = batch * (c * mult) * h * w
    read_bias = (batch * (c * mult) * h * w) if bias else 0
    relu_rw = (2 * batch * (c * mult) * h * w) if relu else 0
    elems = read_w + read_x + write_y + read_bias + relu_rw
    return elems * DT_SIZE

def run_depthwise(x: torch.Tensor) -> List[Row]:
    rows = []
    per_block_flops = depthwise_block_flops(C_DW, IMG_H, IMG_W, K_DW, USE_BIAS_DW, COUNT_RELU_DW, BATCH, MULT_DW)
    per_block_bytes = depthwise_block_bytes(C_DW, IMG_H, IMG_W, K_DW, USE_BIAS_DW, COUNT_RELU_DW, BATCH, MULT_DW)
    per_layer_flops = per_block_flops * DW_REPEAT_PER_LAYER
    per_layer_bytes = per_block_bytes * DW_REPEAT_PER_LAYER

    for L in range(1, 11):
        model = build_depthwise_net(L).to(DEVICE)
        rss_before = psutil.Process().memory_info().rss
        avg_ms, avg_s = measure_latency_fastest_half(model, x)
        rss_after = psutil.Process().memory_info().rss
        rss_delta = max(0, rss_after - rss_before)

        flops_total = per_layer_flops * L
        bytes_total = per_layer_bytes * L
        oi = flops_total / bytes_total if bytes_total > 0 else 0.0
        gflops = (flops_total / avg_s) / 1e9 if avg_s > 0 else 0.0

        rows.append(Row(
            layers=L,
            flops=flops_total,
            bytes_used=bytes_total,
            oi_flops_per_byte=oi,
            time_ms=avg_ms,
            gflops=gflops,
            param_mb=bytes_to_mb(param_bytes(model)),
            rss_delta_mb=bytes_to_mb(rss_delta),
        ))
    return rows

# =========================
# 메인: 순차 실행
# =========================
def main():
    set_threads(NUM_THREADS)
    x = torch.randn(BATCH, IMG_C, IMG_W, IMG_H, dtype=DTYPE, device=DEVICE)  # NCHW (주의: IMG_W, IMG_H 순서 확인)
    # 위 한 줄이 헷갈리면 다음처럼 사용하세요:
    # x = torch.randn(BATCH, IMG_C, IMG_H, IMG_W, dtype=DTYPE, device=DEVICE)

    rows_mlp = run_mlp(x)
    print_table(f"MLP (Hidden={HIDDEN_DIM}->{HIDDEN_DIM})", rows_mlp)

    rows_conv = run_conv(x)
    print_table(f"Conv2D ({K_CONV}x{K_CONV}, Cin={CIN_CONV}, Cout={COUT_CONV})", rows_conv)

    rows_pw = run_pointwise(x)
    print_table(f"Pointwise 1x1 (Cin=Cout={C_PW}, per-layer x{PW_REPEAT_PER_LAYER} blocks)", rows_pw)

    rows_dw = run_depthwise(x)
    print_table(f"Depthwise ({K_DW}x{K_DW}, C={C_DW}, per-layer x{DW_REPEAT_PER_LAYER} blocks)", rows_dw)

if __name__ == "__main__":
    main()
