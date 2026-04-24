"""
PyTorch CPU Benchmark
3D Framework: Operation x Precision x Steps
Measures OI (Operational Intensity) and Throughput
"""

import torch
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
MEASURE_RUNS = 20
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
    (128, 128, 128),
    (256, 256, 256),
    (512, 512, 512),
    (768, 768, 768),
]

def run_gemm(dtype: torch.dtype):
    results = []
    prec = "Float32" if dtype == torch.float32 else "Int8"
    print(f"  [GEMM] {prec}")

    for step, (M, K, N) in enumerate(GEMM_SIZES, 1):
        if dtype == torch.int8:
            A = torch.randint(-127, 127, (M, K), dtype=torch.int8).float()
            B = torch.randint(-127, 127, (K, N), dtype=torch.int8).float()
            fn = lambda A=A, B=B: torch.mm(A, B)
        else:
            A = torch.randn(M, K, dtype=dtype)
            B = torch.randn(K, N, dtype=dtype)
            fn = lambda A=A, B=B: torch.mm(A, B)

        lat = measure_latency(fn)
        flops = 2.0 * M * K * N
        elem_bytes = 1 if dtype == torch.int8 else 4
        mem = (M*K + K*N + M*N) * elem_bytes
        tput = (flops / 1e9) / (lat / 1000)
        oi   = flops / mem

        results.append(BenchResult(
            benchmark="GEMM", precision=prec, step=step,
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
    (1, 16,  32,  32, 3),
    (1, 32,  64,  64, 3),
    (1, 64,  64,  64, 3),
    (1, 128, 64,  64, 3),
    (1, 128, 128, 128, 3),
]

def run_conv(dtype: torch.dtype):
    results = []
    prec = "Float32" if dtype == torch.float32 else "Int8"
    print(f"  [Conv] {prec}")

    for step, (B, Cin, H, W, K) in enumerate(CONV_CONFIGS, 1):
        Cout = Cin * 2
        x = torch.randn(B, Cin, H, W, dtype=torch.float32)
        w = torch.randn(Cout, Cin, K, K, dtype=torch.float32)

        if dtype == torch.int8:
            x = x.clamp(-1,1)
            w = w.clamp(-1,1)

        fn = lambda x=x, w=w: F.conv2d(x, w, padding=K//2)

        lat = measure_latency(fn)
        Hout, Wout = H, W
        flops = 2.0 * B * Cout * Hout * Wout * Cin * K * K
        elem_bytes = 4
        mem = (B*Cin*H*W + Cout*Cin*K*K + B*Cout*Hout*Wout) * elem_bytes
        tput = (flops / 1e9) / (lat / 1000)
        oi   = flops / mem

        results.append(BenchResult(
            benchmark="Convolution", precision=prec, step=step,
            param_desc=f"Cin={Cin},H={H},W={W},K={K}",
            flops=flops, mem_bytes=mem,
            latency_ms=lat, throughput_gflops=tput, oi=oi
        ))
        print(f"    Step {step}: Cin={Cin} H={H} W={W} k={K}  lat={lat:.2f}ms  OI={oi:.2f}  Tput={tput:.2f} GFLOPS")
    return results

# ─────────────────────────────────────────────
# 3. Depthwise Convolution Benchmark
# ─────────────────────────────────────────────
DW_CONFIGS = [
    (1, 16,  32,  32, 3),
    (1, 32,  64,  64, 3),
    (1, 64,  64,  64, 3),
    (1, 128, 64,  64, 3),
    (1, 128, 128, 128, 3),
]

def run_depthwise(dtype: torch.dtype):
    results = []
    prec = "Float32" if dtype == torch.float32 else "Int8"
    print(f"  [Depthwise] {prec}")

    for step, (B, C, H, W, K) in enumerate(DW_CONFIGS, 1):
        x = torch.randn(B, C, H, W, dtype=torch.float32)
        w = torch.randn(C, 1, K, K, dtype=torch.float32)

        fn = lambda x=x, w=w: F.conv2d(x, w, padding=K//2, groups=C)

        lat = measure_latency(fn)
        flops = 2.0 * B * C * H * W * K * K
        elem_bytes = 4
        mem = (B*C*H*W + C*K*K + B*C*H*W) * elem_bytes
        tput = (flops / 1e9) / (lat / 1000)
        oi   = flops / mem

        results.append(BenchResult(
            benchmark="Depthwise", precision=prec, step=step,
            param_desc=f"C={C},H={H},W={W},K={K}",
            flops=flops, mem_bytes=mem,
            latency_ms=lat, throughput_gflops=tput, oi=oi
        ))
        print(f"    Step {step}: C={C} H={H} W={W} k={K}  lat={lat:.2f}ms  OI={oi:.2f}  Tput={tput:.2f} GFLOPS")
    return results

# ─────────────────────────────────────────────
# 4. Elementwise Benchmark
# ─────────────────────────────────────────────
EW_SIZES = [
    1024,
    1024*16,
    1024*64,
    1024*128,
    1024*256,
]

def run_elementwise(dtype: torch.dtype):
    results = []
    prec = "Float32" if dtype == torch.float32 else "Int8"
    print(f"  [Elementwise] {prec}")

    for step, N in enumerate(EW_SIZES, 1):
        x = torch.randn(N, dtype=torch.float32)
        y = torch.randn(N, dtype=torch.float32)

        fn = lambda x=x, y=y: x + y

        lat = measure_latency(fn)
        flops = float(N)
        elem_bytes = 4
        mem = 3 * N * elem_bytes   # read x, y; write out
        tput = (flops / 1e9) / (lat / 1000)
        oi   = flops / mem

        results.append(BenchResult(
            benchmark="Elementwise", precision=prec, step=step,
            param_desc=f"N={N}",
            flops=flops, mem_bytes=mem,
            latency_ms=lat, throughput_gflops=tput, oi=oi
        ))
        print(f"    Step {step}: N={N}  lat={lat:.2f}ms  OI={oi:.2f}  Tput={tput:.2f} GFLOPS")
    return results

# ─────────────────────────────────────────────
# 5. Attention Benchmark
# ─────────────────────────────────────────────
ATTN_CONFIGS = [
    (1, 1,  64,  32),
    (1, 2,  128, 64),
    (1, 4,  256, 64),
    (1, 8,  512, 64),
    (1, 16, 512, 64),
]

def run_attention(dtype: torch.dtype):
    results = []
    prec = "Float32" if dtype == torch.float32 else "Int8"
    print(f"  [Attention] {prec}")

    for step, (B, H, S, D) in enumerate(ATTN_CONFIGS, 1):
        Q = torch.randn(B, H, S, D, dtype=torch.float32)
        K_ = torch.randn(B, H, S, D, dtype=torch.float32)
        V  = torch.randn(B, H, S, D, dtype=torch.float32)
        scale = 1.0 / math.sqrt(D)

        def fn(Q=Q, K=K_, V=V, scale=scale):
            attn = torch.softmax(torch.matmul(Q, K.transpose(-2,-1)) * scale, dim=-1)
            return torch.matmul(attn, V)

        lat = measure_latency(fn)
        # QK^T: 2*B*H*S*S*D, softmax: B*H*S*S, AV: 2*B*H*S*S*D
        flops = B * H * (2*S*S*D + S*S + 2*S*S*D)
        elem_bytes = 4
        mem = (3*B*H*S*D + 2*B*H*S*S + B*H*S*D) * elem_bytes
        tput = (flops / 1e9) / (lat / 1000)
        oi   = flops / mem

        results.append(BenchResult(
            benchmark="Attention", precision=prec, step=step,
            param_desc=f"H={H},S={S},D={D}",
            flops=flops, mem_bytes=mem,
            latency_ms=lat, throughput_gflops=tput, oi=oi
        ))
        print(f"    Step {step}: H={H} S={S} D={D}  lat={lat:.2f}ms  OI={oi:.2f}  Tput={tput:.2f} GFLOPS")
    return results

# ─────────────────────────────────────────────
# Regression fit: linear fit on log-log for OI vs Throughput
# ─────────────────────────────────────────────
def regression(results: List[BenchResult]):
    ois   = np.array([r.oi  for r in results])
    tputs = np.array([r.throughput_gflops for r in results])
    # log-log linear regression
    with np.errstate(divide='ignore'):
        log_oi   = np.log10(np.maximum(ois, 1e-9))
        log_tput = np.log10(np.maximum(tputs, 1e-9))
    slope, intercept = np.polyfit(log_oi, log_tput, 1)
    return slope, intercept

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
# Plot: Roofline-style OI vs Throughput per benchmark
# ─────────────────────────────────────────────
BENCH_COLORS = {
    "GEMM":        "#e74c3c",
    "Convolution": "#f39c12",
    "Depthwise":   "#27ae60",
    "Elementwise": "#2980b9",
    "Attention":   "#8e44ad",
}

def plot_results(all_results: List[BenchResult], save_path: str):
    benchmarks = ["GEMM", "Convolution", "Depthwise", "Elementwise", "Attention"]
    precisions = ["Float32", "Int8"]

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#0f0f1a")
    fig.suptitle("PyTorch CPU Benchmark — OI vs Throughput (Roofline)",
                 color="white", fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(len(benchmarks), len(precisions),
                           figure=fig, hspace=0.5, wspace=0.35)

    for bi, bench in enumerate(benchmarks):
        for pi, prec in enumerate(precisions):
            ax = fig.add_subplot(gs[bi, pi])
            ax.set_facecolor("#1a1a2e")

            subset = [r for r in all_results if r.benchmark == bench and r.precision == prec]
            if not subset:
                ax.text(0.5, 0.5, "N/A", color="gray", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            ois   = [r.oi for r in subset]
            tputs = [r.throughput_gflops for r in subset]
            steps = [r.step for r in subset]
            color = BENCH_COLORS[bench]

            # scatter
            sc = ax.scatter(ois, tputs, c=color, s=80, zorder=5,
                            edgecolors="white", linewidths=0.5)

            # step labels
            for x, y, s in zip(ois, tputs, steps):
                ax.annotate(f"S{s}", (x, y), textcoords="offset points",
                            xytext=(5, 4), color="white", fontsize=7)

            # regression line
            if len(subset) >= 2:
                slope, intercept = regression(subset)
                xi = np.linspace(min(ois)*0.8, max(ois)*1.2, 100)
                yi = 10**(slope * np.log10(np.maximum(xi, 1e-9)) + intercept)
                ax.plot(xi, yi, color=color, alpha=0.5, linewidth=1.5,
                        linestyle="--", label=f"slope={slope:.2f}")
                ax.legend(fontsize=7, facecolor="#0f0f1a", edgecolor="gray",
                          labelcolor="white")

            # styling
            title_color = "#ff6b6b" if prec == "Float32" and bench == "GEMM" else color
            label_suffix = " (Baseline)" if prec == "Float32" and bench == "GEMM" else ""
            ax.set_title(f"{bench} · {prec}{label_suffix}",
                         color=title_color, fontsize=9, fontweight="bold")
            ax.set_xlabel("OI (FLOP/byte)", color="#aaaaaa", fontsize=8)
            ax.set_ylabel("Throughput (GFLOPS)", color="#aaaaaa", fontsize=8)
            ax.tick_params(colors="#888888", labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")
            ax.grid(True, color="#2a2a4a", linewidth=0.5)

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
        "GEMM":        run_gemm,
        "Convolution": run_conv,
        "Depthwise":   run_depthwise,
        "Elementwise": run_elementwise,
        "Attention":   run_attention,
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

    # Plot
    plot_path = os.path.join(RESULTS_DIR, "benchmark_plot.png")
    plot_results(all_results, plot_path)

    # Summary table
    print("\n" + "="*70)
    print(f"{'Benchmark':<14} {'Precision':<10} {'Steps':>5} "
          f"{'Avg OI':>8} {'Avg Tput(GFLOPS)':>16} {'Avg Lat(ms)':>12}")
    print("-"*70)
    benchmarks = ["GEMM","Convolution","Depthwise","Elementwise","Attention"]
    for bench in benchmarks:
        for prec in ["Float32","Int8"]:
            sub = [r for r in all_results if r.benchmark==bench and r.precision==prec]
            if not sub: continue
            avg_oi   = np.mean([r.oi for r in sub])
            avg_tput = np.mean([r.throughput_gflops for r in sub])
            avg_lat  = np.mean([r.latency_ms for r in sub])
            suffix = " ← Baseline" if bench=="GEMM" and prec=="Float32" else ""
            print(f"{bench:<14} {prec:<10} {len(sub):>5} "
                  f"{avg_oi:>8.2f} {avg_tput:>16.4f} {avg_lat:>12.3f}{suffix}")
    print("="*70)

if __name__ == "__main__":
    main()
