"""
PyTorch CPU Benchmark
3D Framework: Operation x Precision x Steps
Measures OI (Operational Intensity) and Throughput

DW_ReLU → DPE_Block:
  Depthwise(3×3) + BN + ReLU  →  Pointwise(1×1) + BN  →  Elementwise Add (skip)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import csv
import os
import math
from dataclasses import dataclass, field
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
WARMUP_RUNS = 5
MEASURE_RUNS = 30
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DTYPES = {
    "Float32": torch.float32,
    "Int8":    torch.int8,
}

# ─────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────
@dataclass
class BenchResult:
    benchmark:  str
    precision:  str
    step:       int
    param_desc: str
    flops:      float   # estimated FLOPs
    mem_bytes:  float   # estimated memory bytes accessed
    latency_ms: float
    throughput_gflops: float
    oi:         float   # Operational Intensity = FLOPs / bytes

# ─────────────────────────────────────────────
# Timer util
# ─────────────────────────────────────────────
def measure_latency(fn, warmup=WARMUP_RUNS, runs=MEASURE_RUNS) -> float:
    """Returns mean latency in milliseconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return float(np.mean(times))

# ─────────────────────────────────────────────
# 1. GEMM Benchmark
# ─────────────────────────────────────────────
GEMM_SIZES = [
    (64,  64,  64),
    (96,  96,  96),
    (128, 128, 128),
    (160, 160, 160),
    (192, 192, 192),
    (256, 256, 256),
    (320, 320, 320),
    (448, 448, 448),
    (576, 576, 576),
    (768, 768, 768),
]

def run_gemm(dtype: torch.dtype):
    results = []
    prec = "Float32" if dtype == torch.float32 else "Int8"
    print(f"  [GEMM+ReLU] {prec}")

    for step, (M, K, N) in enumerate(GEMM_SIZES, 1):
        if dtype == torch.int8:
            A = torch.randint(-127, 127, (M, K), dtype=torch.int8).float()
            B = torch.randint(-127, 127, (K, N), dtype=torch.int8).float()
            fn = lambda A=A, B=B: F.relu(torch.mm(A, B))
        else:
            A = torch.randn(M, K, dtype=dtype)
            B = torch.randn(K, N, dtype=dtype)
            fn = lambda A=A, B=B: F.relu(torch.mm(A, B))

        lat = measure_latency(fn)
        # Int8 케이스도 .float()로 캐스팅 후 float32로 실행하므로 메모리는 항상 4바이트
        elem_bytes = 4
        flops = 2.0 * M * K * N + M * N          # mm + ReLU
        mem   = (M*K + K*N + 3*M*N) * elem_bytes  # mm I/O + ReLU read+write
        tput  = (flops / 1e9) / (lat / 1000)
        oi    = flops / mem

        results.append(BenchResult(
            benchmark="GEMM_ReLU", precision=prec, step=step,
            param_desc=f"M={M},K={K},N={N}",
            flops=flops, mem_bytes=mem,
            latency_ms=lat, throughput_gflops=tput, oi=oi
        ))
        print(f"    Step {step}: {M}x{K}x{N}  lat={lat:.2f}ms  OI={oi:.2f}  Tput={tput:.2f} GFLOPS")
    return results

# ─────────────────────────────────────────────
# 2. Convolution Benchmark
# ─────────────────────────────────────────────
CONV_CONFIGS = [
    (1, 16,  32,  32,  3),
    (1, 24,  48,  48,  3),
    (1, 32,  48,  48,  3),
    (1, 32,  64,  64,  3),
    (1, 48,  64,  64,  3),
    (1, 64,  64,  64,  3),
    (1, 64,  80,  80,  3),
    (1, 64,  96,  96,  3),
    (1, 96,  112, 112, 3),
    (1, 128, 128, 128, 3),
]

def run_conv(dtype: torch.dtype):
    results = []
    prec = "Float32" if dtype == torch.float32 else "Int8"
    print(f"  [Conv+ReLU] {prec}")

    for step, (B, Cin, H, W, K) in enumerate(CONV_CONFIGS, 1):
        Cout = Cin * 2
        x = torch.randn(B, Cin, H, W, dtype=torch.float32)
        w = torch.randn(Cout, Cin, K, K, dtype=torch.float32)

        if dtype == torch.int8:
            x = x.clamp(-1, 1)
            w = w.clamp(-1, 1)

        fn = lambda x=x, w=w: F.relu(F.conv2d(x, w, padding=K//2))

        lat = measure_latency(fn)
        Hout, Wout = H, W
        out_elems  = B * Cout * Hout * Wout
        flops = 2.0 * out_elems * Cin * K * K + out_elems   # conv + ReLU
        elem_bytes = 4
        mem = (B*Cin*H*W + Cout*Cin*K*K + 3*out_elems) * elem_bytes  # conv I/O + ReLU read+write
        tput = (flops / 1e9) / (lat / 1000)
        oi   = flops / mem

        results.append(BenchResult(
            benchmark="Conv_ReLU", precision=prec, step=step,
            param_desc=f"Cin={Cin},Cout={Cout},H={H},W={W},K={K}",
            flops=flops, mem_bytes=mem,
            latency_ms=lat, throughput_gflops=tput, oi=oi
        ))
        print(f"    Step {step}: Cin={Cin} Cout={Cout} H={H} W={W} k={K}  lat={lat:.2f}ms  OI={oi:.2f}  Tput={tput:.2f} GFLOPS")
    return results

# ─────────────────────────────────────────────
# 3. DPE Block Benchmark
#
#  구조: Depthwise(3×3)+BN+ReLU → Pointwise(1×1)+BN → Elementwise Add (skip)
#
#  FLOPs 산정:
#    - DW conv  : 2 * B*C*H*W * K*K     (채널별 독립 3×3)
#    - BN(DW)   : 2 * B*C*H*W           (mean/var 정규화, 근사)
#    - ReLU     : 1 * B*C*H*W
#    - PW conv  : 2 * B*C*H*W * C       (1×1, Cin=Cout=C)
#    - BN(PW)   : 2 * B*C*H*W
#    - Add(skip): 1 * B*C*H*W
#
#  Memory 산정 (elem_bytes=4, float32):
#    - DW  입력/가중치/출력 : B*C*H*W + C*K*K + B*C*H*W
#    - PW  가중치/출력     : C*C     + B*C*H*W          (입력은 DW 출력 재사용)
#    - Add 출력(write)    : B*C*H*W
#    - skip 입력(read)    : B*C*H*W
# ─────────────────────────────────────────────
DPE_CONFIGS = [
    (1, 16,  32,  32,  3),
    (1, 24,  48,  48,  3),
    (1, 32,  48,  48,  3),
    (1, 32,  64,  64,  3),
    (1, 48,  64,  64,  3),
    (1, 64,  64,  64,  3),
    (1, 64,  80,  80,  3),
    (1, 64,  96,  96,  3),
    (1, 96,  112, 112, 3),
    (1, 128, 128, 128, 3),
]


class DPEBlock(nn.Module):
    """
    단순 DPE 블록:
      ① Depthwise Conv(3×3) + BN + ReLU  — 공간 패턴 추출 (MobileNet / Xception 방식)
      ② Pointwise Conv(1×1) + BN         — 채널 정보 결합 (MobileNet 방식, 활성화 없음)
      ③ Elementwise Add(skip)            — 잔차 연결      (ResNet 방식)
    """
    def __init__(self, C: int, K: int = 3):
        super().__init__()
        # ① Depthwise
        self.dw = nn.Conv2d(C, C, kernel_size=K, padding=K // 2, groups=C, bias=False)
        self.bn_dw = nn.BatchNorm2d(C)
        # ② Pointwise
        self.pw = nn.Conv2d(C, C, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn_dw(self.dw(x)))   # ① DW + BN + ReLU
        out = self.bn_pw(self.pw(out))          # ② PW + BN  (활성화 없음)
        return out + x                          # ③ Elementwise Add


def run_dpe(dtype: torch.dtype):
    """
    DPE Block 벤치마크.
    Int8 설정은 Float32 텐서로 실행하되 값 범위만 int8 수준으로 clamp
    (PyTorch CPU의 quantized grouped conv 지원 제한 때문).
    """
    results = []
    prec = "Float32" if dtype == torch.float32 else "Int8"
    print(f"  [DPE_Block] {prec}")

    for step, (B, C, H, W, K) in enumerate(DPE_CONFIGS, 1):
        model = DPEBlock(C, K).eval()

        x = torch.randn(B, C, H, W)
        if dtype == torch.int8:
            x = x.clamp(-1.0, 1.0)  # int8 범위 모사

        fn = lambda model=model, x=x: model(x)

        lat = measure_latency(fn)

        elems = B * C * H * W          # 공간 원소 수
        elem_bytes = 4                  # float32 기준 (int8 clamp 모사이므로 동일)

        # ── FLOPs ──────────────────────────────────────────
        flops_dw  = 2.0 * elems * K * K   # DW conv
        flops_bn1 = 2.0 * elems            # BN(DW)
        flops_relu = elems                 # ReLU
        flops_pw  = 2.0 * elems * C        # PW conv  (1×1, Cin=Cout=C)
        flops_bn2 = 2.0 * elems            # BN(PW)
        flops_add = elems                  # Elementwise Add
        flops = flops_dw + flops_bn1 + flops_relu + flops_pw + flops_bn2 + flops_add

        # ── Memory bytes ───────────────────────────────────
        mem_dw  = (elems + C * K * K + elems) * elem_bytes   # DW: in + weight + out
        mem_pw  = (C * C  + elems)            * elem_bytes    # PW: weight + out (in 재사용)
        mem_add = (elems  + elems)            * elem_bytes    # Add: skip_read + out_write
        mem = mem_dw + mem_pw + mem_add

        tput = (flops / 1e9) / (lat / 1000)
        oi   = flops / mem

        results.append(BenchResult(
            benchmark="DPE_Block", precision=prec, step=step,
            param_desc=f"C={C},H={H},W={W},K={K}",
            flops=flops, mem_bytes=mem,
            latency_ms=lat, throughput_gflops=tput, oi=oi
        ))
        print(f"    Step {step}: C={C} H={H} W={W} k={K}  lat={lat:.2f}ms  OI={oi:.2f}  Tput={tput:.2f} GFLOPS")
    return results

# ─────────────────────────────────────────────
# 4. Attention Benchmark
# ─────────────────────────────────────────────
ATTN_CONFIGS = [
    (1, 1,  64,  32),
    (1, 1,  128, 64),
    (1, 2,  128, 64),
    (1, 2,  256, 64),
    (1, 4,  256, 64),
    (1, 4,  384, 64),
    (1, 8,  256, 64),
    (1, 8,  384, 64),
    (1, 8,  512, 64),
    (1, 16, 512, 64),
]

def run_attention(dtype: torch.dtype):
    results = []
    prec = "Float32" if dtype == torch.float32 else "Int8"
    print(f"  [Attn+GELU] {prec}")

    for step, (B, H, S, D) in enumerate(ATTN_CONFIGS, 1):
        Q  = torch.randn(B, H, S, D, dtype=torch.float32)
        K_ = torch.randn(B, H, S, D, dtype=torch.float32)
        V  = torch.randn(B, H, S, D, dtype=torch.float32)
        scale = 1.0 / math.sqrt(D)

        def fn(Q=Q, K=K_, V=V, scale=scale):
            attn = torch.softmax(torch.matmul(Q, K.transpose(-2,-1)) * scale, dim=-1)
            return F.gelu(torch.matmul(attn, V))

        lat = measure_latency(fn)
        out_elems  = B * H * S * D
        flops = B * H * (2*S*S*D + S*S + 2*S*S*D) + 8 * out_elems
        elem_bytes = 4
        mem = (3*B*H*S*D + 2*B*H*S*S + 3*out_elems) * elem_bytes
        tput = (flops / 1e9) / (lat / 1000)
        oi   = flops / mem

        results.append(BenchResult(
            benchmark="Attn_GELU", precision=prec, step=step,
            param_desc=f"B={B},H={H},S={S},D={D}",
            flops=flops, mem_bytes=mem,
            latency_ms=lat, throughput_gflops=tput, oi=oi
        ))
        print(f"    Step {step}: B={B} H={H} S={S} D={D}  lat={lat:.2f}ms  OI={oi:.2f}  Tput={tput:.2f} GFLOPS")
    return results

# ─────────────────────────────────────────────
# Save CSV
# ─────────────────────────────────────────────
def save_csv(all_results: List[BenchResult], path: str):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "benchmark","precision","step","param_desc",
            "flops","mem_bytes","latency_ms","throughput_gflops","oi"
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow(r.__dict__)
    print(f"\n✅ CSV saved → {path}")

# ─────────────────────────────────────────────
# Regression Model: OI → Throughput (log-log linear)
# ─────────────────────────────────────────────
def build_oi_regression(all_results: List[BenchResult]):
    ois   = np.array([r.oi                for r in all_results])
    tputs = np.array([r.throughput_gflops for r in all_results])
    with np.errstate(divide="ignore"):
        log_oi   = np.log10(np.maximum(ois,   1e-9))
        log_tput = np.log10(np.maximum(tputs, 1e-9))
    slope, intercept = np.polyfit(log_oi, log_tput, 1)
    log_pred = slope * log_oi + intercept
    ss_res = np.sum((log_tput - log_pred) ** 2)
    ss_tot = np.sum((log_tput - log_tput.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, r2

def predict_throughput(oi: float, slope: float, intercept: float) -> float:
    return 10 ** (slope * math.log10(max(oi, 1e-9)) + intercept)


# ─────────────────────────────────────────────
# Combined Regression: (OI, mem_bytes) → latency_ms
# ─────────────────────────────────────────────
def build_combined_regression(results: List[BenchResult]):
    ois  = np.array([r.oi        for r in results])
    mems = np.array([r.mem_bytes for r in results])
    lats = np.array([r.latency_ms for r in results])
    log_oi  = np.log10(np.maximum(ois,  1e-9))
    log_mem = np.log10(np.maximum(mems, 1e-9))
    log_lat = np.log10(np.maximum(lats, 1e-9))

    X = np.column_stack([log_oi, log_mem, np.ones(len(results))])
    coeffs, _, _, _ = np.linalg.lstsq(X, log_lat, rcond=None)
    a, b, c = coeffs

    log_pred = X @ coeffs
    ss_res = np.sum((log_lat - log_pred) ** 2)
    ss_tot = np.sum((log_lat - log_lat.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(a), float(b), float(c), float(r2)

def predict_latency_combined(oi: float, mem_bytes: float,
                              a: float, b: float, c: float) -> float:
    log_pred = (a * math.log10(max(oi, 1e-9))
                + b * math.log10(max(mem_bytes, 1e-9))
                + c)
    return 10 ** log_pred


def save_combined_regression_csv(all_results: List[BenchResult], path: str):
    benchmarks = ["GEMM_ReLU", "Conv_ReLU", "DPE_Block", "Attn_GELU"]
    rows = []
    for bench in benchmarks:
        for prec in ["Float32", "Int8"]:
            sub = [r for r in all_results if r.benchmark == bench and r.precision == prec]
            if len(sub) < 3:
                continue
            a, b, c, r2 = build_combined_regression(sub)
            # 각 스텝별 예측 → RMSE / MAPE 로 검증 (평균 단일 점 오류 수정)
            preds   = [predict_latency_combined(r.oi, r.mem_bytes, a, b, c) for r in sub]
            actuals = [r.latency_ms for r in sub]
            rmse    = float(np.sqrt(np.mean([(p - ac) ** 2 for p, ac in zip(preds, actuals)])))
            mape    = float(np.mean([abs(p - ac) / max(ac, 1e-9) * 100 for p, ac in zip(preds, actuals)]))
            avg_lat = float(np.mean(actuals))
            avg_oi  = float(np.mean([r.oi        for r in sub]))
            avg_mem = float(np.mean([r.mem_bytes  for r in sub]))
            rows.append({
                "benchmark":     bench,
                "precision":     prec,
                "coef_oi":       round(a, 6),
                "coef_mem":      round(b, 6),
                "intercept":     round(c, 6),
                "r2":            round(r2, 6),
                "avg_oi":        round(avg_oi, 6),
                "avg_mem_bytes": round(avg_mem, 2),
                "rmse_ms":       round(rmse, 6),
                "mape_pct":      round(mape, 4),
                "avg_actual_ms": round(avg_lat, 6),
            })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Combined Regression CSV saved → {path}")

def save_regression_csv(all_results: List[BenchResult], path: str):
    benchmarks = ["GEMM_ReLU", "Conv_ReLU", "DPE_Block", "Attn_GELU"]
    rows = []
    for bench in benchmarks:
        for prec in ["Float32", "Int8"]:
            sub = [r for r in all_results if r.benchmark == bench and r.precision == prec]
            if len(sub) < 2:
                continue
            slope, intercept, r2 = build_oi_regression(sub)
            avg_oi     = float(np.mean([r.oi for r in sub]))
            actual_avg = float(np.mean([r.throughput_gflops for r in sub]))
            pred       = predict_throughput(avg_oi, slope, intercept)
            err_pct    = (pred - actual_avg) / actual_avg * 100
            rows.append({
                "benchmark":        bench,
                "precision":        prec,
                "slope":            round(slope, 6),
                "intercept":        round(intercept, 6),
                "r2":               round(r2, 6),
                "avg_oi":           round(avg_oi, 6),
                "pred_tput_gflops": round(pred, 6),
                "actual_tput_gflops": round(actual_avg, 6),
                "error_pct":        round(err_pct, 4),
            })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Regression CSV saved → {path}")

# ─────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────
BENCH_COLORS = {
    "GEMM_ReLU": "#e74c3c",
    "Conv_ReLU": "#f39c12",
    "DPE_Block": "#27ae60",   # 기존 DW_ReLU 색상 유지
    "Attn_GELU": "#8e44ad",
}

def plot_results(all_results: List[BenchResult], save_path: str):
    benchmarks = ["GEMM_ReLU", "Conv_ReLU", "DPE_Block", "Attn_GELU"]
    precisions = ["Float32", "Int8"]

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0f0f1a")
    fig.suptitle("PyTorch CPU Benchmark — Throughput & OI per Step",
                 color="white", fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(len(benchmarks), len(precisions),
                           figure=fig, hspace=0.55, wspace=0.45)

    for bi, bench in enumerate(benchmarks):
        for pi, prec in enumerate(precisions):
            ax = fig.add_subplot(gs[bi, pi])
            ax.set_facecolor("#1a1a2e")

            subset = sorted(
                [r for r in all_results if r.benchmark == bench and r.precision == prec],
                key=lambda r: r.step
            )
            if not subset:
                ax.text(0.5, 0.5, "N/A", color="gray", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            steps = [r.step for r in subset]
            tputs = [r.throughput_gflops for r in subset]
            ois   = [r.oi for r in subset]
            color = BENCH_COLORS[bench]

            ax.bar(steps, tputs, color=color, alpha=0.75, zorder=4,
                   edgecolor="white", linewidth=0.4)

            ax.set_xlabel("Step", color="#aaaaaa", fontsize=8)
            ax.set_ylabel("Throughput (GFLOPS)", color=color, fontsize=8)
            ax.tick_params(axis="y", colors=color, labelsize=7)
            ax.tick_params(axis="x", colors="#888888", labelsize=7)
            ax.set_xticks(steps)

            ax2 = ax.twinx()
            ax2.set_facecolor("#1a1a2e")
            oi_color = "#f0c040"
            ax2.plot(steps, ois, color=oi_color, linewidth=1.4,
                     linestyle=":", zorder=3)
            ax2.scatter(steps, ois, color=oi_color, s=40, zorder=4,
                        marker="o", edgecolors="white", linewidths=0.4)
            ax2.set_ylabel("OI (FLOP/byte)", color=oi_color, fontsize=8)
            ax2.tick_params(axis="y", colors=oi_color, labelsize=7)

            title_color = "#ff6b6b" if prec == "Float32" and bench == "GEMM_ReLU" else color
            label_suffix = " (Baseline)" if prec == "Float32" and bench == "GEMM_ReLU" else ""
            ax.set_title(f"{bench} · {prec}{label_suffix}",
                         color=title_color, fontsize=9, fontweight="bold")
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")
            for spine in ax2.spines.values():
                spine.set_edgecolor("#333355")
            ax.grid(True, color="#2a2a4a", linewidth=0.5)

            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            handles = [
                Patch(facecolor=color, edgecolor="white", alpha=0.75, label="Throughput"),
                Line2D([0], [0], color=oi_color, linewidth=1.4, linestyle=":", label="OI"),
            ]
            ax.legend(handles=handles, fontsize=7, facecolor="#0f0f1a",
                      edgecolor="gray", labelcolor="white", loc="upper left")

    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"✅ Plot saved → {save_path}")
    plt.close()

# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main():
    all_results: List[BenchResult] = []

    runners = {
        "GEMM_ReLU": run_gemm,
        "Conv_ReLU": run_conv,
        "DPE_Block": run_dpe,       # ← DW_ReLU 대체
        "Attn_GELU": run_attention,
    }

    for bench_name, runner in runners.items():
        print(f"\n{'='*50}")
        print(f"  Benchmark: {bench_name}")
        print(f"{'='*50}")
        for prec_name, dtype in DTYPES.items():
            results = runner(dtype)
            all_results.extend(results)

    # Save CSV
    csv_path = os.path.join(RESULTS_DIR, "benchmark_results.csv")
    save_csv(all_results, csv_path)

    reg_csv_path = os.path.join(RESULTS_DIR, "regression_results.csv")
    save_regression_csv(all_results, reg_csv_path)

    comb_csv_path = os.path.join(RESULTS_DIR, "combined_regression_results.csv")
    save_combined_regression_csv(all_results, comb_csv_path)

    plot_path = os.path.join(RESULTS_DIR, "benchmark_plot.png")
    plot_results(all_results, plot_path)

    # Summary table
    benchmarks = ["GEMM_ReLU", "Conv_ReLU", "DPE_Block", "Attn_GELU"]
    print("\n" + "="*70)
    print(f"{'Benchmark':<14} {'Precision':<10} {'Steps':>5} "
          f"{'Avg OI':>8} {'Avg Tput(GFLOPS)':>16} {'Avg Lat(ms)':>12}")
    print("-"*70)
    for bench in benchmarks:
        for prec in ["Float32","Int8"]:
            sub = [r for r in all_results if r.benchmark==bench and r.precision==prec]
            if not sub: continue
            avg_oi   = np.mean([r.oi for r in sub])
            avg_tput = np.mean([r.throughput_gflops for r in sub])
            avg_lat  = np.mean([r.latency_ms for r in sub])
            suffix = " ← Baseline" if bench=="GEMM_ReLU" and prec=="Float32" else ""
            print(f"{bench:<14} {prec:<10} {len(sub):>5} "
                  f"{avg_oi:>8.2f} {avg_tput:>16.4f} {avg_lat:>12.3f}{suffix}")
    print("="*70)

    # Regression summary
    print(f"\n{'='*70}")
    print("  OI → Throughput Regression (log-log linear)")
    print(f"{'='*70}")
    for bench in benchmarks:
        for prec in ["Float32", "Int8"]:
            sub = [r for r in all_results if r.benchmark == bench and r.precision == prec]
            if len(sub) < 2:
                continue
            slope, intercept, r2 = build_oi_regression(sub)
            avg_oi     = float(np.mean([r.oi for r in sub]))
            actual_avg = float(np.mean([r.throughput_gflops for r in sub]))
            pred       = predict_throughput(avg_oi, slope, intercept)
            err_pct    = (pred - actual_avg) / actual_avg * 100
            print(f"\n  [{bench} · {prec}]")
            print(f"    log10(Tput) = {slope:.4f} * log10(OI) + {intercept:.4f}  R²={r2:.4f}")
            print(f"    Avg OI={avg_oi:.4f}  →  Pred={pred:.4f} GFLOPS  "
                  f"Actual={actual_avg:.4f} GFLOPS  Error={err_pct:+.2f}%")
    print(f"\n{'='*70}")

    # Combined regression summary
    print(f"\n{'='*78}")
    print("  Combined Regression: log(lat) = a·log(OI) + b·log(mem) + c")
    print(f"{'='*78}")
    print(f"  {'Benchmark':<14} {'Prec':<8} {'coef_OI':>9} {'coef_mem':>10} "
          f"{'intercept':>10} {'R²':>7} {'RMSE(ms)':>10} {'MAPE%':>8}")
    print(f"  {'-'*76}")
    for bench in benchmarks:
        for prec in ["Float32", "Int8"]:
            sub = [r for r in all_results if r.benchmark == bench and r.precision == prec]
            if len(sub) < 3:
                continue
            a, b, c, r2 = build_combined_regression(sub)
            preds   = [predict_latency_combined(r.oi, r.mem_bytes, a, b, c) for r in sub]
            actuals = [r.latency_ms for r in sub]
            rmse    = float(np.sqrt(np.mean([(p - ac) ** 2 for p, ac in zip(preds, actuals)])))
            mape    = float(np.mean([abs(p - ac) / max(ac, 1e-9) * 100 for p, ac in zip(preds, actuals)]))
            print(f"  {bench:<14} {prec:<8} {a:>9.4f} {b:>10.4f} "
                  f"{c:>10.4f} {r2:>7.4f} {rmse:>10.4f} {mape:>8.2f}%")
    print(f"  {'='*76}")

if __name__ == "__main__":
    main()