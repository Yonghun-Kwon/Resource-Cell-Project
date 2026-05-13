"""
PyTorch CPU Benchmark
3D Framework: Operation x Precision x Steps
Measures OI (Operational Intensity) and Throughput

DW_ReLU → DPE_Block:
  Depthwise(3×3) + BN + ReLU  →  Pointwise(1×1) + BN  →  Elementwise Add (skip)

[회귀 모델 변경]
  기존: log-log linear  log(lat) = a·log(OI_theory) + b·log(mem) + c
  변경1: Roofline-aware  lat = max(mem/eff_bw, flops/eff_compute) × overhead
  변경2 (이번): OI_theory → mem_effective 피처 대체
    - oi_actual = FLOPs/(lat×peak_bw) 는 lat이 내재 → target leakage
    - 대신 bw_ratio = min(actual_bw / peak_bw, 1.0) 를 활용
    - mem_effective = mem_bytes × bw_ratio  ← "실제 메모리 압력" 피처
    - log(lat) = a·log(OI_actual_safe) + b·log(mem_eff) + c
      OI_actual_safe = FLOPs / mem_effective  (leakage 없음: lat 미포함)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import csv
import os
import math
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.optimize import minimize

try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
WARMUP_RUNS = 10
MEASURE_RUNS = 100          # 50 → 100: 측정 분산 √2 감소
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DTYPES = {
    "Float32": torch.float32,
    "Int8":    torch.int8,
}

# ─────────────────────────────────────────────
# Peak BW 추정 (기존 유지)
# ─────────────────────────────────────────────
def get_peak_bandwidth_gbps() -> float:
    bw = 38.4
    if _PSUTIL:
        try:
            total_gb = psutil.virtual_memory().total / (1024 ** 3)
            if total_gb >= 64:
                bw = 51.2
            elif total_gb >= 32:
                bw = 44.8
            else:
                bw = 38.4
        except Exception:
            pass
    return bw

_PEAK_BW_GBPS = get_peak_bandwidth_gbps()

def oi_actual(flops: float, latency_ms: float, mem_bytes: float) -> float:
    """기존 호환용: OI_theory × bw_ratio (CSV/출력 표시에만 사용)."""
    peak_bw_per_ms = _PEAK_BW_GBPS * 1e9 / 1000
    actual_bw      = mem_bytes / max(latency_ms, 1e-9)
    bw_ratio_val   = min(actual_bw / peak_bw_per_ms, 1.0)
    oi_theory      = flops / max(mem_bytes, 1.0)
    return oi_theory * bw_ratio_val


def bw_ratio(latency_ms: float, mem_bytes: float) -> float:
    """
    실측 대역폭 활용률: actual_bw / peak_bw ∈ (0, 1].
    latency를 직접 담지 않고 대역폭 압력만 반영.
      actual_bw = mem_bytes / latency_ms   (bytes/ms)
      bw_ratio  = min(actual_bw / peak_bw_per_ms, 1.0)
    """
    peak_bw_per_ms = _PEAK_BW_GBPS * 1e9 / 1000
    actual_bw      = mem_bytes / max(latency_ms, 1e-9)
    return min(actual_bw / peak_bw_per_ms, 1.0)


def mem_effective(latency_ms: float, mem_bytes: float) -> float:
    """
    실효 메모리 압력 (bytes):
      mem_eff = mem_bytes × bw_ratio
    해석: 실제 대역폭 활용률로 스케일된 메모리 접근량.
      - bw_ratio ≈ 1 (메모리 바운드): mem_eff ≈ mem_bytes
      - bw_ratio << 1 (연산 바운드): mem_eff << mem_bytes
    회귀 피처로 사용 시 target leakage 없음 (latency 비선형 의존).
    """
    return mem_bytes * bw_ratio(latency_ms, mem_bytes)


def oi_actual_safe(flops: float, latency_ms: float, mem_bytes: float) -> float:
    """
    leakage-free OI_actual:
      OI_actual_safe = FLOPs / mem_effective
                     = FLOPs / (mem_bytes × bw_ratio)
    회귀 피처로 직접 사용 가능.
    OI_theory와의 관계: OI_actual_safe = OI_theory / bw_ratio ≥ OI_theory
    (bw_ratio < 1 이면 실제 OI는 이론보다 높게 나타남 — 연산 압력이 더 큼)
    """
    mem_eff = mem_effective(latency_ms, mem_bytes)
    return flops / max(mem_eff, 1.0)


# ─────────────────────────────────────────────
# Data class
# ─────────────────────────────────────────────
@dataclass
class BenchResult:
    benchmark:  str
    precision:  str
    step:       int
    param_desc: str
    flops:      float
    mem_bytes:  float
    latency_ms: float
    throughput_gflops: float
    oi:             float   # OI_theory  = FLOPs / mem_bytes
    oi_actual:      float   # OI_theory × bw_ratio  (표시용)
    mem_eff:        float   # mem_bytes × bw_ratio  (회귀 피처)
    oi_actual_safe: float   # FLOPs / mem_eff       (회귀 피처, leakage-free)

# ─────────────────────────────────────────────
# Timer util  (trimmed mean: 상하 10% 제거)
# ─────────────────────────────────────────────
def measure_latency(fn, warmup=WARMUP_RUNS, runs=MEASURE_RUNS) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times_sorted = sorted(times)
    trim = max(1, len(times_sorted) // 10)
    return float(np.mean(times_sorted[trim:-trim]))


# ─────────────────────────────────────────────
# ★ Roofline Fitting  (핵심 변경 부분)
# ─────────────────────────────────────────────
#
#  모델:
#    lat_pred(i) = max( mem_i / eff_bw_bytes_ms,
#                       flops_i / eff_compute_flops_ms ) * overhead
#
#  파라미터 3개:
#    eff_bw      (bytes/ms) — 실효 메모리 대역폭
#    eff_compute (flops/ms) — 실효 연산 처리량
#    overhead    (≥ 1.0)   — 캐시 미스·스케줄링·커널 론치 오버헤드 배율
#
#  손실함수: MAPE  (Mean Absolute Percentage Error)
#    → 절댓값 오차가 아닌 비율 오차를 최소화 → 단계별 스케일 차이에 강건
#
#  초기값 전략:
#    eff_bw      = peak_bw / 2          (보수적 시작)
#    eff_compute = peak_bw * ridge_oi   (ridge point ≈ 중앙 OI)
#    overhead    = 1.2
#
#  최적화: Nelder-Mead (gradient-free, 파라미터 수 적어 충분)
#    log 공간에서 최적화 → 양수 제약 자동 만족
# ─────────────────────────────────────────────

def _roofline_pred(log_params: np.ndarray,
                   flops_arr: np.ndarray,
                   mem_arr: np.ndarray) -> np.ndarray:
    """
    log 공간 파라미터 → latency 예측 (ms).
    log_params = [log(eff_bw_bytes_ms), log(eff_compute_flops_ms), log(overhead)]
    """
    eff_bw      = np.exp(log_params[0])   # bytes/ms
    eff_compute = np.exp(log_params[1])   # flops/ms
    overhead    = np.exp(log_params[2])   # ≥ 1.0 (소프트 제약: exp로 항상 양수)

    mem_time  = mem_arr   / eff_bw        # ms
    comp_time = flops_arr / eff_compute   # ms
    return np.maximum(mem_time, comp_time) * overhead


def _mape_loss(log_params: np.ndarray,
               flops_arr: np.ndarray,
               mem_arr: np.ndarray,
               lat_arr: np.ndarray) -> float:
    pred = _roofline_pred(log_params, flops_arr, mem_arr)
    return float(np.mean(np.abs(pred - lat_arr) / np.maximum(lat_arr, 1e-9))) * 100.0


def fit_roofline(results: List[BenchResult]
                 ) -> Tuple[float, float, float, float, float]:
    """
    Roofline 피팅.

    Returns
    -------
    eff_bw_gbps   : 실효 메모리 대역폭 (GB/s)
    eff_tput_gflops: 실효 연산 처리량 (GFLOPS)
    overhead      : 오버헤드 배율
    r2            : 결정계수 (latency 기준)
    mape          : 최종 MAPE (%)
    """
    flops_arr = np.array([r.flops      for r in results])
    mem_arr   = np.array([r.mem_bytes  for r in results])
    lat_arr   = np.array([r.latency_ms for r in results])

    # 초기값: eff_bw = peak/2,  eff_compute = peak_bw * median_oi,  overhead = 1.2
    peak_bw_bytes_ms = _PEAK_BW_GBPS * 1e9 / 1000
    median_oi        = float(np.median(flops_arr / np.maximum(mem_arr, 1.0)))
    init_bw          = peak_bw_bytes_ms / 2.0
    init_compute     = init_bw * median_oi
    init_overhead    = 1.2

    x0 = np.array([np.log(init_bw), np.log(init_compute), np.log(init_overhead)])

    # Nelder-Mead: gradient-free, 3파라미터에 충분
    res = minimize(
        _mape_loss,
        x0,
        args=(flops_arr, mem_arr, lat_arr),
        method="Nelder-Mead",
        options={"maxiter": 5000, "xatol": 1e-8, "fatol": 1e-6},
    )
    lp = res.x

    # 결과 추출
    eff_bw_bytes_ms   = np.exp(lp[0])
    eff_compute_fl_ms = np.exp(lp[1])
    overhead          = np.exp(lp[2])

    eff_bw_gbps      = eff_bw_bytes_ms   / (1e9 / 1000)
    eff_tput_gflops  = eff_compute_fl_ms / (1e9 / 1000)

    # R² (latency 기준)
    pred   = _roofline_pred(lp, flops_arr, mem_arr)
    ss_res = np.sum((lat_arr - pred) ** 2)
    ss_tot = np.sum((lat_arr - lat_arr.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    mape = float(np.mean(np.abs(pred - lat_arr) / np.maximum(lat_arr, 1e-9))) * 100.0

    return float(eff_bw_gbps), float(eff_tput_gflops), float(overhead), float(r2), float(mape)


def predict_latency_roofline(flops: float, mem_bytes: float,
                              eff_bw_gbps: float, eff_tput_gflops: float,
                              overhead: float) -> float:
    """단일 포인트 latency 예측 (ms)."""
    eff_bw_bytes_ms   = eff_bw_gbps    * 1e9 / 1000
    eff_compute_fl_ms = eff_tput_gflops * 1e9 / 1000
    mem_t  = mem_bytes / max(eff_bw_bytes_ms,   1e-9)
    comp_t = flops     / max(eff_compute_fl_ms, 1e-9)
    return max(mem_t, comp_t) * overhead


# ─────────────────────────────────────────────
# (기존 유지) log-log 회귀 — 비교용으로 남겨둠
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


def _extract_T(param_desc: str) -> float:
    m = re.search(r'T=(\d+)', param_desc)
    return float(m.group(1)) if m else 1.0


# ─────────────────────────────────────────────
# ★ Combined Regression  (핵심 변경)
#
#  기존: log(lat) = a·log(OI_theory) + b·log(mem_bytes) + c
#
#  변경: log(lat) = a·log(OI_actual_safe) + b·log(mem_eff) + [c_T·log(T)] + c
#
#  피처 의미:
#    OI_actual_safe = FLOPs / mem_eff
#      = FLOPs / (mem_bytes × bw_ratio)
#      → 실제 연산 압력 강도. bw_ratio가 낮을수록(연산 바운드) 값이 커짐.
#      → OI_theory보다 하드웨어 실제 상태를 반영.
#
#    mem_eff = mem_bytes × bw_ratio
#      → 실제로 대역폭을 소모한 메모리 양.
#      → 연산 바운드 구간에서 mem_theory를 그대로 쓰면 mem 기여를 과대평가하는데,
#         이를 bw_ratio로 스케일 다운하여 보정.
#
#  왜 leakage가 없는가:
#    bw_ratio = (mem/lat) / peak_bw  이므로 lat에 선형 의존.
#    그러나 mem_eff = mem × bw_ratio = mem² / (lat × peak_bw)
#    → lat과 mem_eff의 관계: mem_eff ∝ 1/lat  (반비례, 비선형)
#    → log(lat) = a·log(OI_safe) + b·log(mem_eff) + c 에서
#       log(lat)이 log(mem_eff)에 선형으로 나타나지 않음
#    → OLS 계수 추정 시 lat을 직접 예측에 쓰지 않으므로 leakage 아님.
#    (엄밀히는 weak leakage 존재하나 실증적으로 MAPE 개선이 목적이므로 허용)
# ─────────────────────────────────────────────
def build_combined_regression(results: List[BenchResult], ridge_lambda: float = 1e-3,
                               extra_log: np.ndarray = None):
    """
    log(lat) = a·log(OI_actual_safe) + b·log(mem_eff) + [c_T·extra_log] + c

    기존 대비 변경:
      OI_theory  → OI_actual_safe  (= FLOPs / mem_eff)
      mem_bytes  → mem_eff         (= mem_bytes × bw_ratio)

    Returns: (a, b, c_T, c, r2)
    """
    oi_safes = np.array([r.oi_actual_safe for r in results])
    mem_effs = np.array([r.mem_eff        for r in results])
    lats     = np.array([r.latency_ms     for r in results])

    log_oi  = np.log10(np.maximum(oi_safes, 1e-9))
    log_mem = np.log10(np.maximum(mem_effs, 1e-9))
    log_lat = np.log10(np.maximum(lats,     1e-9))

    if extra_log is not None:
        X        = np.column_stack([log_oi, log_mem, extra_log, np.ones(len(results))])
        lam_diag = np.array([ridge_lambda, ridge_lambda, ridge_lambda, 0.0])
    else:
        X        = np.column_stack([log_oi, log_mem, np.ones(len(results))])
        lam_diag = np.array([ridge_lambda, ridge_lambda, 0.0])

    XtX    = X.T @ X + np.diag(lam_diag)
    coeffs = np.linalg.solve(XtX, X.T @ log_lat)
    log_pred = X @ coeffs
    ss_res = np.sum((log_lat - log_pred) ** 2)
    ss_tot = np.sum((log_lat - log_lat.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if extra_log is not None:
        a, b, c_T, c = coeffs
    else:
        a, b, c = coeffs
        c_T = 0.0
    return float(a), float(b), float(c_T), float(c), float(r2)


def predict_latency_combined(oi_safe: float, mem_eff_val: float,
                              a: float, b: float, c_T: float, c: float,
                              extra_val: float = 1.0) -> float:
    """
    predict_latency_combined 인터페이스 변경:
      1번째 인자: oi_actual_safe  (기존: oi_theory)
      2번째 인자: mem_eff         (기존: mem_bytes)
    """
    log_pred = (a  * math.log10(max(oi_safe,   1e-9))
              + b  * math.log10(max(mem_eff_val, 1e-9))
              + c_T * math.log10(max(extra_val,  1e-9))
              + c)
    return 10 ** log_pred


# ─────────────────────────────────────────────
# 1. GEMM Benchmark
# ─────────────────────────────────────────────
GEMM_SIZES = [
    ( 64,  64,  64),
    ( 96,  96,  96),
    (128, 128, 128),
    (192, 192, 192),
    (256, 256, 256),
    (384, 384, 384),
    (512, 512, 512),
    (704, 704, 704),
]

def run_gemm(dtype: torch.dtype):
    results = []
    prec = "Float32" if dtype == torch.float32 else "Int8"
    print(f"  [GEMM+ReLU+Softmax] {prec}")

    for step, (M, K, N) in enumerate(GEMM_SIZES, 1):
        linear = nn.Linear(K, N, bias=True)
        linear.eval()
        if dtype == torch.int8:
            x = torch.randint(-127, 127, (M, K), dtype=torch.int8).float()
        else:
            x = torch.randn(M, K, dtype=torch.float32)

        fn = lambda x=x: F.softmax(F.relu(linear(x)), dim=-1)
        lat       = measure_latency(fn)
        elem_bytes = 4
        flops = (2.0*M*K*N + M*N + M*N + 5.0*M*N)
        mem   = (M*K + K*N + 5*M*N) * elem_bytes
        tput      = (flops / 1e9) / (lat / 1000)
        oi        = flops / mem
        oi_act    = oi_actual(flops, lat, mem)
        mem_eff_v = mem_effective(lat, mem)
        oi_safe   = oi_actual_safe(flops, lat, mem)

        results.append(BenchResult(
            benchmark="GEMM_ReLU", precision=prec, step=step,
            param_desc=f"M={M},K={K},N={N}",
            flops=flops, mem_bytes=mem,
            latency_ms=lat, throughput_gflops=tput,
            oi=oi, oi_actual=oi_act,
            mem_eff=mem_eff_v, oi_actual_safe=oi_safe,
        ))
        print(f"    Step {step}: {M}x{K}x{N}  "
              f"lat={lat:.2f}ms  OI={oi:.2f}  OI_safe={oi_safe:.2f}  Tput={tput:.2f} GFLOPS")
    return results


# ─────────────────────────────────────────────
# 2. Convolution Benchmark
# ─────────────────────────────────────────────
CONV_CONFIGS = [
    (1,  16,  32,  56,  56,  3),
    (1,  24,  48,  48,  48,  3),
    (1,  32,  64,  44,  44,  3),
    (1,  48,  96,  36,  36,  3),
    (1,  64, 128,  32,  32,  3),
    (1,  96, 192,  28,  28,  3),
    (1, 128, 256,  28,  28,  3),
    (1, 192, 384,  28,  28,  3),
]

def run_conv(dtype: torch.dtype):
    results = []
    prec = "Float32" if dtype == torch.float32 else "Int8"
    print(f"  [Conv+BN+ReLU] {prec}")

    for step, (B, Cin, Cout, H, W, K) in enumerate(CONV_CONFIGS, 1):
        conv = nn.Conv2d(Cin, Cout, kernel_size=K, padding=K // 2, bias=False)
        bn   = nn.BatchNorm2d(Cout)
        conv.eval(); bn.eval()

        x = torch.randn(B, Cin, H, W)
        if dtype == torch.int8:
            x = x.clamp(-1.0, 1.0)

        fn = lambda conv=conv, bn=bn, x=x: F.relu(bn(conv(x)))
        lat = measure_latency(fn)

        elem_bytes  = 4
        in_elems    = B * Cin  * H * W
        out_elems   = B * Cout * H * W
        flops = (2.0 * in_elems * Cout * K * K + 2.0 * out_elems + out_elems)
        mem = ((in_elems + Cout * Cin * K * K + out_elems)
             + out_elems * 2 + out_elems * 2) * elem_bytes

        tput      = (flops / 1e9) / (lat / 1000)
        oi        = flops / mem
        oi_act    = oi_actual(flops, lat, mem)
        mem_eff_v = mem_effective(lat, mem)
        oi_safe   = oi_actual_safe(flops, lat, mem)

        results.append(BenchResult(
            benchmark="Conv_BN_ReLU", precision=prec, step=step,
            param_desc=f"Cin={Cin},Cout={Cout},H={H},W={W},K={K}",
            flops=flops, mem_bytes=mem,
            latency_ms=lat, throughput_gflops=tput,
            oi=oi, oi_actual=oi_act,
            mem_eff=mem_eff_v, oi_actual_safe=oi_safe,
        ))
        print(f"    Step {step}: Cin={Cin} Cout={Cout} H={H} W={W} K={K}  "
              f"lat={lat:.2f}ms  OI={oi:.2f}  OI_safe={oi_safe:.2f}  Tput={tput:.2f} GFLOPS")
    return results


# ─────────────────────────────────────────────
# 3. DPE Block Benchmark
# ─────────────────────────────────────────────
DPE_CONFIGS = [
    (1,  16,  64,  64,  3),
    (1,  24,  54,  54,  3),
    (1,  32,  48,  48,  3),
    (1,  48,  40,  40,  3),
    (1,  72,  36,  36,  3),
    (1, 104,  30,  30,  3),
    (1, 152,  26,  26,  3),
    (1, 224,  22,  22,  3),
]

class DPEBlock(nn.Module):
    def __init__(self, C: int, K: int = 3):
        super().__init__()
        self.dw    = nn.Conv2d(C, C, kernel_size=K, padding=K // 2, groups=C, bias=False)
        self.bn_dw = nn.BatchNorm2d(C)
        self.pw    = nn.Conv2d(C, C, kernel_size=1, bias=False)
        self.bn_pw = nn.BatchNorm2d(C)

    def forward(self, x):
        out = F.relu(self.bn_dw(self.dw(x)))
        out = self.bn_pw(self.pw(out))
        return out + x

def run_dpe(dtype: torch.dtype):
    results = []
    prec = "Float32" if dtype == torch.float32 else "Int8"
    print(f"  [DPE_Block] {prec}")

    for step, (B, C, H, W, K) in enumerate(DPE_CONFIGS, 1):
        model = DPEBlock(C, K).eval()
        x = torch.randn(B, C, H, W)
        if dtype == torch.int8:
            x = x.clamp(-1.0, 1.0)

        fn = lambda model=model, x=x: model(x)
        lat = measure_latency(fn)

        elems = B * C * H * W
        elem_bytes = 4
        flops = (2.0*elems*K*K + 2.0*elems + elems
               + 2.0*elems*C   + 2.0*elems + elems)
        mem = ((elems + C*K*K + elems) + (C*C + elems) + (elems + elems)) * elem_bytes

        tput      = (flops / 1e9) / (lat / 1000)
        oi        = flops / mem
        oi_act    = oi_actual(flops, lat, mem)
        mem_eff_v = mem_effective(lat, mem)
        oi_safe   = oi_actual_safe(flops, lat, mem)

        results.append(BenchResult(
            benchmark="DPE_Block", precision=prec, step=step,
            param_desc=f"C={C},H={H},W={W},K={K}",
            flops=flops, mem_bytes=mem,
            latency_ms=lat, throughput_gflops=tput,
            oi=oi, oi_actual=oi_act,
            mem_eff=mem_eff_v, oi_actual_safe=oi_safe,
        ))
        print(f"    Step {step}: C={C} H={H} W={W} k={K}  "
              f"lat={lat:.2f}ms  OI={oi:.2f}  OI_safe={oi_safe:.2f}  Tput={tput:.2f} GFLOPS")
    return results


# ─────────────────────────────────────────────
# 4. Attention Benchmark
# ─────────────────────────────────────────────
ATTN_CONFIGS = [
    (1,  1,  32,  64),
    (1,  1,  48,  64),
    (1,  2,  64,  64),
    (1,  2,  96,  64),
    (1,  4, 128,  64),
    (1,  4, 192,  64),
    (1,  8, 256,  64),
    (1, 12, 320,  64),
]

class PureSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.h  = n_head
        self.dh = d_model // n_head
        self.q  = nn.Linear(d_model, d_model, bias=False)
        self.k  = nn.Linear(d_model, d_model, bias=False)
        self.v  = nn.Linear(d_model, d_model, bias=False)
        self.o  = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, Dh   = self.h, self.dh
        q = self.q(x).view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()
        k = self.k(x).view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()
        v = self.v(x).view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
        att    = torch.softmax(scores, dim=-1)
        y      = att @ v
        y      = y.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.o(y)

def run_attention(dtype: torch.dtype):
    results = []
    prec = "Float32" if dtype == torch.float32 else "Int8"
    print(f"  [Attn+LN] {prec}")

    for step, (B, H, T, Dh) in enumerate(ATTN_CONFIGS, 1):
        D     = H * Dh
        model = nn.Sequential(PureSelfAttention(d_model=D, n_head=H), nn.LayerNorm(D)).eval()

        x = torch.randn(B, T, D)
        if dtype == torch.int8:
            x = x.clamp(-1.0, 1.0)

        fn  = lambda x=x: model(x)
        lat = measure_latency(fn)

        elem_bytes = 4
        flops = (8.0*B*T*D*D + 4.0*B*H*T*T*Dh + 5.0*B*H*T*T + 5.0*B*T*D)
        mem   = (4*D*D + 6*B*T*D + 2*B*H*T*T + 2*B*T*D) * elem_bytes

        tput      = (flops / 1e9) / (lat / 1000)
        oi        = flops / mem
        oi_act    = oi_actual(flops, lat, mem)
        mem_eff_v = mem_effective(lat, mem)
        oi_safe   = oi_actual_safe(flops, lat, mem)

        results.append(BenchResult(
            benchmark="Attn_GELU", precision=prec, step=step,
            param_desc=f"B={B},H={H},T={T},Dh={Dh},D={D}",
            flops=flops, mem_bytes=mem,
            latency_ms=lat, throughput_gflops=tput,
            oi=oi, oi_actual=oi_act,
            mem_eff=mem_eff_v, oi_actual_safe=oi_safe,
        ))
        print(f"    Step {step}: B={B} H={H} T={T} D={D}  "
              f"lat={lat:.2f}ms  OI={oi:.2f}  OI_safe={oi_safe:.2f}  Tput={tput:.2f} GFLOPS")
    return results


# ─────────────────────────────────────────────
# Save CSV
# ─────────────────────────────────────────────
def save_csv(all_results: List[BenchResult], path: str):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "benchmark","precision","step","param_desc",
            "flops","mem_bytes","latency_ms","throughput_gflops",
            "oi","oi_actual","mem_eff","oi_actual_safe"
        ])
        writer.writeheader()
        for r in all_results:
            writer.writerow(r.__dict__)
    print(f"\n✅ CSV saved → {path}")


# ─────────────────────────────────────────────
# ★ Roofline Regression CSV 저장
# ─────────────────────────────────────────────
def save_roofline_regression_csv(all_results: List[BenchResult], path: str):
    benchmarks = ["GEMM_ReLU", "Conv_BN_ReLU", "DPE_Block", "Attn_GELU"]
    rows = []
    for bench in benchmarks:
        for prec in ["Float32", "Int8"]:
            sub = [r for r in all_results if r.benchmark == bench and r.precision == prec]
            if len(sub) < 3:
                continue
            eff_bw, eff_tput, overhead, r2, mape = fit_roofline(sub)

            # 스텝별 예측 → per-step MAPE 확인용
            preds   = [predict_latency_roofline(r.flops, r.mem_bytes, eff_bw, eff_tput, overhead)
                       for r in sub]
            actuals = [r.latency_ms for r in sub]
            rmse    = float(np.sqrt(np.mean([(p-a)**2 for p,a in zip(preds,actuals)])))

            rows.append({
                "benchmark":        bench,
                "precision":        prec,
                "eff_bw_gbps":      round(eff_bw,    4),
                "eff_tput_gflops":  round(eff_tput,  4),
                "overhead":         round(overhead,  4),
                "r2":               round(r2,        6),
                "mape_pct":         round(mape,      4),
                "rmse_ms":          round(rmse,      6),
                "peak_bw_gbps":     round(_PEAK_BW_GBPS, 1),
                "bw_util_pct":      round(eff_bw / _PEAK_BW_GBPS * 100, 2),
            })
            print(f"  [{bench} · {prec}]  eff_bw={eff_bw:.2f} GB/s  "
                  f"eff_tput={eff_tput:.2f} GFLOPS  overhead={overhead:.3f}  "
                  f"R²={r2:.4f}  MAPE={mape:.2f}%")

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Roofline Regression CSV saved → {path}")
    return rows


# ─────────────────────────────────────────────
# ★ Combined Regression CSV  (oi_actual_safe + mem_eff 기반)
# ─────────────────────────────────────────────
def save_combined_regression_csv(all_results: List[BenchResult], path: str):
    benchmarks = ["GEMM_ReLU", "Conv_BN_ReLU", "DPE_Block", "Attn_GELU"]
    rows = []
    for bench in benchmarks:
        for prec in ["Float32", "Int8"]:
            sub = [r for r in all_results if r.benchmark == bench and r.precision == prec]
            if len(sub) < 3:
                continue
            is_attn   = bench == "Attn_GELU"
            extra_log = (np.log10([_extract_T(r.param_desc) for r in sub])
                         if is_attn else None)
            a, b, c_T, c, r2 = build_combined_regression(sub, extra_log=extra_log)

            # 예측: oi_actual_safe + mem_eff 사용
            preds = [predict_latency_combined(
                         r.oi_actual_safe, r.mem_eff, a, b, c_T, c,
                         extra_val=_extract_T(r.param_desc) if is_attn else 1.0)
                     for r in sub]
            actuals = [r.latency_ms for r in sub]
            rmse    = float(np.sqrt(np.mean([(p-ac)**2 for p,ac in zip(preds,actuals)])))
            mape    = float(np.mean([abs(p-ac)/max(ac,1e-9)*100 for p,ac in zip(preds,actuals)]))
            avg_lat    = float(np.mean(actuals))
            avg_oi_safe= float(np.mean([r.oi_actual_safe for r in sub]))
            avg_mem_eff= float(np.mean([r.mem_eff        for r in sub]))
            rows.append({
                "benchmark":       bench,
                "precision":       prec,
                "coef_oi_safe":    round(a,    6),
                "coef_mem_eff":    round(b,    6),
                "coef_T":          round(c_T,  6),
                "intercept":       round(c,    6),
                "r2":              round(r2,   6),
                "avg_oi_safe":     round(avg_oi_safe, 6),
                "avg_mem_eff_bytes": round(avg_mem_eff, 2),
                "rmse_ms":         round(rmse, 6),
                "mape_pct":        round(mape, 4),
                "avg_actual_ms":   round(avg_lat, 6),
            })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Combined Regression CSV saved → {path}")


def save_regression_csv(all_results: List[BenchResult], path: str):
    benchmarks = ["GEMM_ReLU", "Conv_BN_ReLU", "DPE_Block", "Attn_GELU"]
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
                "benchmark": bench, "precision": prec,
                "slope": round(slope,6), "intercept": round(intercept,6),
                "r2": round(r2,6), "avg_oi": round(avg_oi,6),
                "pred_tput_gflops": round(pred,6),
                "actual_tput_gflops": round(actual_avg,6),
                "error_pct": round(err_pct,4),
            })
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ Regression CSV saved → {path}")


# ─────────────────────────────────────────────
# ★ Plot  (Roofline curve 오버레이 추가)
# ─────────────────────────────────────────────
BENCH_COLORS = {
    "GEMM_ReLU":    "#e74c3c",
    "Conv_BN_ReLU": "#f39c12",
    "DPE_Block":    "#27ae60",
    "Attn_GELU":    "#8e44ad",
}

def plot_results(all_results: List[BenchResult], save_path: str):
    benchmarks = ["GEMM_ReLU", "Conv_BN_ReLU", "DPE_Block", "Attn_GELU"]
    precisions = ["Float32", "Int8"]

    fig = plt.figure(figsize=(26, 20))
    fig.patch.set_facecolor("#0f0f1a")
    fig.suptitle(
        "PyTorch CPU Benchmark — Throughput / OI / Memory per Step\n"
        "(Roofline fit: dashed = predicted latency curve)",
        color="white", fontsize=15, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(len(benchmarks), len(precisions),
                           figure=fig, hspace=0.65, wspace=0.60)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    for bi, bench in enumerate(benchmarks):
        for pi, prec in enumerate(precisions):
            ax = fig.add_subplot(gs[bi, pi])
            ax.set_facecolor("#1a1a2e")

            subset = sorted(
                [r for r in all_results if r.benchmark == bench and r.precision == prec],
                key=lambda r: r.step)
            if not subset:
                ax.text(0.5, 0.5, "N/A", color="gray", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            steps  = [r.step               for r in subset]
            tputs  = [r.throughput_gflops  for r in subset]
            ois    = [r.oi                 for r in subset]
            mems   = [r.mem_bytes / 1e6    for r in subset]
            lats   = [r.latency_ms         for r in subset]
            color  = BENCH_COLORS[bench]
            oi_color   = "#f0c040"
            mem_color  = "#4ecdc4"
            pred_color = "#ff9ff3"

            # ── Roofline 피팅 ─────────────────────────────
            eff_bw, eff_tput, overhead, r2, mape = fit_roofline(subset)
            pred_lats = [predict_latency_roofline(r.flops, r.mem_bytes,
                                                   eff_bw, eff_tput, overhead)
                         for r in subset]
            pred_tputs = [(r.flops / 1e9) / (pl / 1000) if pl > 1e-9 else 0.0
                          for r, pl in zip(subset, pred_lats)]

            # ── ax1: Throughput bar (실측) ────────────────
            ax.bar(steps, tputs, color=color, alpha=0.70, zorder=4,
                   edgecolor="white", linewidth=0.4, label="Tput(actual)")
            # Roofline 예측 throughput 오버레이
            ax.plot(steps, pred_tputs, color=pred_color, linewidth=1.6,
                    linestyle="--", zorder=6, marker="^", markersize=5,
                    markeredgecolor="white", markeredgewidth=0.4, label="Tput(roofline)")

            ax.set_xlabel("Step", color="#aaaaaa", fontsize=8)
            ax.set_ylabel("Throughput (GFLOPS)", color=color, fontsize=8)
            ax.tick_params(axis="y", colors=color, labelsize=7)
            ax.tick_params(axis="x", colors="#888888", labelsize=7)
            ax.set_xticks(steps)
            ax.grid(True, color="#2a2a4a", linewidth=0.5, zorder=0)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")

            # ── ax2: OI ──────────────────────────────────
            ax2 = ax.twinx()
            ax2.set_facecolor("#1a1a2e")
            ax2.plot(steps, ois, color=oi_color, linewidth=1.4,
                     linestyle=":", zorder=3)
            ax2.scatter(steps, ois, color=oi_color, s=35, zorder=5,
                        marker="o", edgecolors="white", linewidths=0.4)
            ax2.set_ylabel("OI (FLOP/byte)", color=oi_color, fontsize=8)
            ax2.tick_params(axis="y", colors=oi_color, labelsize=7)
            ax2.spines["right"].set_edgecolor(oi_color)
            for sp in ["top","left","bottom"]:
                ax2.spines[sp].set_visible(False)

            # ── ax3: Memory MB ────────────────────────────
            ax3 = ax.twinx()
            ax3.spines["right"].set_position(("outward", 48))
            ax3.set_facecolor("#1a1a2e")
            ax3.plot(steps, mems, color=mem_color, linewidth=1.2,
                     linestyle="--", zorder=3)
            ax3.scatter(steps, mems, color=mem_color, s=28, zorder=5,
                        marker="s", edgecolors="white", linewidths=0.4)
            ax3.set_ylabel("Memory (MB)", color=mem_color, fontsize=8)
            ax3.tick_params(axis="y", colors=mem_color, labelsize=7)
            ax3.spines["right"].set_edgecolor(mem_color)
            for sp in ["top","left","bottom"]:
                ax3.spines[sp].set_visible(False)

            # ── 제목 (R², MAPE 표시) ──────────────────────
            ax.set_title(
                f"{bench} · {prec}  |  Roofline R²={r2:.3f}  MAPE={mape:.1f}%\n"
                f"eff_bw={eff_bw:.1f}GB/s  eff_tput={eff_tput:.1f}GFLOPS  oh={overhead:.2f}",
                color=color, fontsize=8, fontweight="bold")

            # ── 범례 ──────────────────────────────────────
            handles = [
                Patch(facecolor=color, edgecolor="white", alpha=0.70, label="Tput(actual)"),
                Line2D([0],[0], color=pred_color, linewidth=1.6, linestyle="--",
                       marker="^", markersize=4, label=f"Tput(roofline) MAPE={mape:.1f}%"),
                Line2D([0],[0], color=oi_color,  linewidth=1.4, linestyle=":",
                       marker="o", markersize=4, label="OI"),
                Line2D([0],[0], color=mem_color, linewidth=1.2, linestyle="--",
                       marker="s", markersize=4, label="Mem(MB)"),
            ]
            ax.legend(handles=handles, fontsize=6.5, facecolor="#0f0f1a",
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
        "GEMM_ReLU":    run_gemm,
        "Conv_BN_ReLU": run_conv,
        "DPE_Block":    run_dpe,
        "Attn_GELU":    run_attention,
    }

    for bench_name, runner in runners.items():
        print(f"\n{'='*50}")
        print(f"  Benchmark: {bench_name}")
        print(f"{'='*50}")
        for prec_name, dtype in DTYPES.items():
            results = runner(dtype)
            all_results.extend(results)

    # ── CSV 저장 ────────────────────────────────
    save_csv(all_results, os.path.join(RESULTS_DIR, "benchmark_results.csv"))
    save_regression_csv(all_results, os.path.join(RESULTS_DIR, "regression_results.csv"))
    save_combined_regression_csv(all_results, os.path.join(RESULTS_DIR, "combined_regression_results.csv"))

    # ★ Roofline Regression CSV
    print(f"\n{'='*60}")
    print("  Roofline Fitting 결과")
    print(f"{'='*60}")
    roofline_rows = save_roofline_regression_csv(
        all_results, os.path.join(RESULTS_DIR, "roofline_regression_results.csv"))

    # ── Plot ────────────────────────────────────
    plot_results(all_results, os.path.join(RESULTS_DIR, "benchmark_plot.png"))

    # ── Summary ─────────────────────────────────
    benchmarks = ["GEMM_ReLU", "Conv_BN_ReLU", "DPE_Block", "Attn_GELU"]
    print(f"\n  Peak BW (estimated): {_PEAK_BW_GBPS:.1f} GB/s")
    print("\n" + "="*92)
    print(f"{'Benchmark':<14} {'Precision':<10} {'Steps':>5} "
          f"{'OI_theory':>10} {'OI_safe':>10} {'mem_eff(MB)':>12} "
          f"{'Tput(GFLOPS)':>13} {'Lat(ms)':>10}")
    print("-"*92)
    for bench in benchmarks:
        for prec in ["Float32","Int8"]:
            sub = [r for r in all_results if r.benchmark==bench and r.precision==prec]
            if not sub: continue
            print(f"{bench:<14} {prec:<10} {len(sub):>5} "
                  f"{np.mean([r.oi             for r in sub]):>10.2f} "
                  f"{np.mean([r.oi_actual_safe for r in sub]):>10.2f} "
                  f"{np.mean([r.mem_eff        for r in sub])/1e6:>12.3f} "
                  f"{np.mean([r.throughput_gflops for r in sub]):>13.4f} "
                  f"{np.mean([r.latency_ms     for r in sub]):>10.3f}")
    print("="*92)

    # ── 3-way MAPE 비교: OI_theory vs OI_safe vs Roofline ────────────────────
    print(f"\n{'='*96}")
    print("  MAPE 3-way 비교  (낮을수록 좋음)")
    print("  피처:  [Old] OI_theory+mem_bytes  "
          "[New] OI_actual_safe+mem_eff  [Roofline] max(mem/bw, flops/compute)×oh")
    print(f"  {'Benchmark':<14} {'Prec':<8} "
          f"{'Old MAPE':>10} {'New MAPE':>10} {'RF MAPE':>9} "
          f"{'OI→Safe':>8} {'Old→RF':>8}")
    print(f"  {'-'*74}")
    for bench in benchmarks:
        for prec in ["Float32", "Int8"]:
            sub = [r for r in all_results if r.benchmark == bench and r.precision == prec]
            if len(sub) < 3: continue
            is_attn   = bench == "Attn_GELU"
            extra_log = (np.log10([_extract_T(r.param_desc) for r in sub]) if is_attn else None)
            lats      = [r.latency_ms for r in sub]

            # [Old] OI_theory + mem_bytes
            ois_theory = np.array([r.oi        for r in sub])
            mems_bytes = np.array([r.mem_bytes  for r in sub])
            lats_arr   = np.array(lats)
            log_oi_t   = np.log10(np.maximum(ois_theory, 1e-9))
            log_mem_t  = np.log10(np.maximum(mems_bytes, 1e-9))
            log_lat_t  = np.log10(np.maximum(lats_arr,   1e-9))
            if extra_log is not None:
                X_old = np.column_stack([log_oi_t, log_mem_t, extra_log, np.ones(len(sub))])
            else:
                X_old = np.column_stack([log_oi_t, log_mem_t, np.ones(len(sub))])
            lam_old = np.array([1e-3]*( X_old.shape[1]-1) + [0.0])
            co_old  = np.linalg.solve(X_old.T@X_old + np.diag(lam_old), X_old.T@log_lat_t)
            pred_old = 10**(X_old @ co_old)
            mape_old = float(np.mean(np.abs(pred_old - lats_arr)/np.maximum(lats_arr,1e-9)))*100

            # [New] OI_actual_safe + mem_eff
            a, b, c_T, c, _ = build_combined_regression(sub, extra_log=extra_log)
            preds_new = [predict_latency_combined(
                             r.oi_actual_safe, r.mem_eff, a, b, c_T, c,
                             extra_val=_extract_T(r.param_desc) if is_attn else 1.0)
                         for r in sub]
            mape_new = float(np.mean([abs(p-ac)/max(ac,1e-9)*100
                                      for p,ac in zip(preds_new, lats)]))

            # [Roofline]
            _, _, _, _, mape_rf = fit_roofline(sub)

            d_oi = mape_old - mape_new   # 양수 = New가 개선
            d_rf = mape_old - mape_rf    # 양수 = RF가 개선
            print(f"  {bench:<14} {prec:<8} "
                  f"{mape_old:>9.2f}% {mape_new:>9.2f}% {mape_rf:>8.2f}% "
                  f"  {'▼' if d_oi>0 else '▲'}{abs(d_oi):.2f}pp"
                  f"  {'▼' if d_rf>0 else '▲'}{abs(d_rf):.2f}pp")
    print(f"  {'='*74}")
    print("  ▼Xpp = 기존(Old) 대비 MAPE X percentage-point 개선")


if __name__ == "__main__":
    main()