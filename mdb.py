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
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
try:
    import psutil
    _PSUTIL = True
except ImportError:
    _PSUTIL = False

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
WARMUP_RUNS = 10
MEASURE_RUNS = 50
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

DTYPES = {
    "Float32": torch.float32,
    "Int8":    torch.int8,
}

# ─────────────────────────────────────────────
# 방법 2: 실측 대역폭 역산 OI (수정판)
#
#  핵심 문제:
#    기존 공식 OI = FLOPs / (latency × peak_bw) 는
#    "latency 전체를 메모리 접근에 쓴다"고 가정 → 분모 과대 → OI 과소
#
#  수정 공식:
#    mem_time  = theoretical_mem / peak_bw   (이론 메모리 전송 시간)
#    comp_time = latency - mem_time           (나머지 = 연산 시간)
#    actual_bw = theoretical_mem / latency    (실제 평균 대역폭)
#    OI_actual = FLOPs / theoretical_mem
#                × (actual_bw / peak_bw)      (대역폭 활용률로 스케일)
#              = FLOPs × latency / (theoretical_mem² / peak_bw)  [정리]
#
#  결론: OI_actual = OI_theory × (actual_bw / peak_bw)
#    → 실제로 peak 대비 얼마나 대역폭을 쓰고 있는지를 OI에 반영
#    → OI_theory ≥ OI_actual 항상 성립
#    → 두 값 차이 = 대역폭 낭비 or 연산 병목 정도
#
#  peak_bandwidth: psutil RAM 용량으로 DDR 세대 추정
# ─────────────────────────────────────────────

def get_peak_bandwidth_gbps() -> float:
    """
    psutil로 총 RAM 크기를 읽어 DDR 세대를 추정하고
    이론 peak bandwidth(GB/s)를 반환.
    실패 시 DDR4-2400 듀얼채널(38.4 GB/s) 가정.
    """
    bw = 38.4  # 기본값: DDR4-2400 dual-channel
    if _PSUTIL:
        try:
            total_gb = psutil.virtual_memory().total / (1024 ** 3)
            if total_gb >= 64:
                bw = 51.2   # DDR4-3200 dual-channel
            elif total_gb >= 32:
                bw = 44.8   # DDR4-2800 dual-channel
            else:
                bw = 38.4   # DDR4-2400 dual-channel
        except Exception:
            pass
    return bw

# 모듈 로드 시 1회 측정
_PEAK_BW_GBPS = get_peak_bandwidth_gbps()

def oi_actual(flops: float, latency_ms: float, mem_bytes: float) -> float:
    """
    방법 2 (수정):
      actual_bw  = mem_bytes / latency_ms          (실제 평균 대역폭, bytes/ms)
      bw_ratio   = actual_bw / peak_bw             (대역폭 활용률, 0~1)
      OI_actual  = OI_theory × bw_ratio
                 = (FLOPs / mem_bytes) × (mem_bytes / latency_ms) / peak_bw_per_ms
                 = FLOPs / (latency_ms × peak_bw_per_ms)  … 이전 공식과 동일하나
      ※ 단, OI_theory 기준으로 표현하면 의미가 명확:
         OI_actual = OI_theory × min(bw_ratio, 1.0)
      → peak를 초과하는 경우 클램프(1.0)로 OI_theory 이하 보장
    """
    peak_bw_per_ms = _PEAK_BW_GBPS * 1e9 / 1000   # bytes/ms
    actual_bw      = mem_bytes / max(latency_ms, 1e-9)
    bw_ratio       = min(actual_bw / peak_bw_per_ms, 1.0)
    oi_theory      = flops / max(mem_bytes, 1.0)
    return oi_theory * bw_ratio


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
    oi:        float   # OI theory = FLOPs / theoretical_mem
    oi_actual: float   # OI method2 = FLOPs / (latency × peak_bw)

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
# ─────────────────────────────────────────────
# GEMM_SIZES 설계 원칙 (8 스텝)
#
#  OI(정방행렬 M=K=N=n) ≈ 2n³ / (5n²·4) = n/10
#    → log(OI) ∝ log(n)  : n을 등비수열로 잡으면 log(OI) 선형
#    → mem     ∝ n²       : n 등비 → mem 지수, 불가피한 트레이드오프
#
#  트레이드오프 해결:
#    n을 2배 등비(64→704)로 설계 → Δlog₁₀(OI) ≈ 0.13~0.18 (균등)
#    mem 증가는 n² 특성상 후반부에서 커지지만, log 회귀에선 문제없음
# ─────────────────────────────────────────────
GEMM_SIZES = [
    ( 64,  64,  64),   # step1  OI≈6.4   mem≈0.08 MB
    ( 96,  96,  96),   # step2  OI≈9.6   mem≈0.18 MB
    (128, 128, 128),   # step3  OI≈12.8  mem≈0.33 MB
    (192, 192, 192),   # step4  OI≈19.2  mem≈0.74 MB
    (256, 256, 256),   # step5  OI≈25.6  mem≈1.31 MB
    (384, 384, 384),   # step6  OI≈38.4  mem≈2.95 MB
    (512, 512, 512),   # step7  OI≈51.2  mem≈5.24 MB
    (704, 704, 704),   # step8  OI≈70.4  mem≈9.91 MB
    #  Δlog₁₀(OI): 0.18, 0.12, 0.18, 0.12, 0.18, 0.12, 0.14  (균등)
]

def run_gemm(dtype: torch.dtype):
    """
    Linear(M,K→N) + ReLU + Softmax 벤치마크.

    multi_dimensio_benchmark.py 구조 반영:
      - nn.Linear(bias=True) + ReLU + Softmax 로 실행
      - Int8은 float32 clamp 모사 (grouped conv 제한과 동일 정책)

    FLOPs:
      - Linear : 2·M·K·N + M·N (bias add)
      - ReLU   : M·N
      - Softmax: 5·M·N (max, exp, sum, div — 근사)

    Memory (float32 기준):
      - Linear : (M·K + K·N + M·N) · 4   ← 입력 + weight + 출력
                 bias(N)는 weight에 비해 작으므로 근사 생략
      - ReLU   : (M·N + M·N) · 4          ← read + write
      - Softmax: (M·N + M·N) · 4          ← read + write
      합계      : (M·K + K·N + 5·M·N) · 4
    """
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
        flops = (2.0*M*K*N + M*N   # Linear (matmul + bias)
               + M*N               # ReLU
               + 5.0*M*N)          # Softmax (max/exp/sum/div — 근사)
        mem   = (M*K + K*N + 5*M*N) * elem_bytes
        tput  = (flops / 1e9) / (lat / 1000)
        oi    = flops / mem
        oi_act = oi_actual(flops, lat, mem)

        results.append(BenchResult(
            benchmark="GEMM_ReLU", precision=prec, step=step,
            param_desc=f"M={M},K={K},N={N}",
            flops=flops, mem_bytes=mem,
            latency_ms=lat, throughput_gflops=tput,
            oi=oi, oi_actual=oi_act,
        ))
        print(f"    Step {step}: {M}x{K}x{N}  "
              f"lat={lat:.2f}ms  OI={oi:.2f}  OI_act={oi_act:.2f}  Tput={tput:.2f} GFLOPS")
    return results

# ─────────────────────────────────────────────
# 2. Convolution Benchmark  (Conv + BN + ReLU)
#
#  현실적 단일 레이어 기준:
#    - Cout = Cin 유지  (VGG/ResNet/DenseNet 블록 내부 설정)
#    - Conv(3×3, padding=1) → BN → ReLU
#    - 채널·공간 크기는 실제 모델 스테이지에서 발췌
#      step 1~3  : VGG-16 초반  (64ch, 112~56px)
#      step 4~6  : ResNet-50 중반 (128~256ch, 56~28px)
#      step 7~8  : ResNet-50 후반 (512ch, 14px)
#      step 9~10 : DenseNet-121 후반 (512~1024ch, 7px)
#
#  FLOPs 산정:
#    - Conv  : 2 * B*C*H*W * C * K²   (Cin=Cout=C)
#    - BN    : 2 * B*C*H*W            (정규화 근사)
#    - ReLU  : 1 * B*C*H*W
#
#  Memory 산정:
#    - Conv  : (B*C*H*W + C*C*K*K + B*C*H*W) * 4   (in + weight + out)
#    - BN    : (B*C*H*W + B*C*H*W)            * 4   (read + write)
#    - ReLU  : (B*C*H*W + B*C*H*W)            * 4   (read + write)
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# CONV_CONFIGS 설계 원칙 (8 스텝, B=1, K=3)
#
#  multi_dimensio_benchmark.py 구조 반영:
#    Cin ≠ Cout (실제 모델처럼 채널 확장)
#    Conv(3×3) + BN + ReLU
#
#  OI = FLOPs / mem
#    FLOPs ≈ 2·B·Cin·Cout·H·W·K²  (Conv 지배)
#    mem   ≈ (B·Cin·H·W + Cout·Cin·K² + B·Cout·H·W) · 4  (Conv)
#           + (B·Cout·H·W · 2) · 4                        (BN read+write)
#           + (B·Cout·H·W · 2) · 4                        (ReLU read+write)
#
#  C를 등비수열, H를 역비례 조정 → OI 단조 증가 + mem 단조 증가
#  mem 단조 증가 조건: (Cin+5·Cout)·H² + 9·Cout·Cin 이 스텝마다 증가해야 함
#  → step2~5의 H를 기존보다 크게 조정 (step6~8은 H=28 고정으로 자연히 증가)
# ─────────────────────────────────────────────
CONV_CONFIGS = [
    # (B, Cin, Cout,  H,   W,  K)
    (1,  16,  32,  56,  56,  3),   # step1  OI≈13.1  mem≈2.25 MB
    (1,  24,  48,  48,  48,  3),   # step2  OI≈19.4  mem≈2.50 MB  H: 44→48
    (1,  32,  64,  44,  44,  3),   # step3  OI≈25.7  mem≈2.80 MB  H: 40→44
    (1,  48,  96,  36,  36,  3),   # step4  OI≈37.2  mem≈2.90 MB  H: 32→36
    (1,  64, 128,  32,  32,  3),   # step5  OI≈47.6  mem≈3.21 MB  H: 28→32
    (1,  96, 192,  28,  28,  3),   # step6  OI≈65.5  mem≈4.02 MB
    (1, 128, 256,  28,  28,  3),   # step7  OI≈82.8  mem≈5.66 MB
    (1, 192, 384,  28,  28,  3),   # step8  OI≈112.2 mem≈9.38 MB
    #  Δlog₁₀(OI): 0.17, 0.12, 0.16, 0.11, 0.14, 0.10, 0.13  (균등)
    #  mem: 2.25→2.50→2.80→2.90→3.21→4.02→5.66→9.38 MB (단조 증가)
]

def run_conv(dtype: torch.dtype):
    """
    Conv(3×3, Cin→Cout) + BN + ReLU 벤치마크.

    multi_dimensio_benchmark.py 구조 반영:
      - Cin ≠ Cout (채널 확장, 실제 모델 구조)
      - Conv → BN → ReLU 순서
      - Int8은 float32 clamp 모사

    FLOPs:
      - Conv : 2·B·Cin·Cout·H·W·K²
      - BN   : 2·B·Cout·H·W
      - ReLU : 1·B·Cout·H·W

    Memory:
      - Conv : (B·Cin·H·W + Cout·Cin·K² + B·Cout·H·W) · 4
      - BN   : (B·Cout·H·W · 2) · 4   ← read + write
      - ReLU : (B·Cout·H·W · 2) · 4   ← read + write
    """
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

        flops = (2.0 * in_elems * Cout * K * K   # Conv
               + 2.0 * out_elems                  # BN
               + out_elems)                        # ReLU

        mem = ((in_elems + Cout * Cin * K * K + out_elems)   # Conv: in+weight+out
             + out_elems * 2                                   # BN:  read+write
             + out_elems * 2                                   # ReLU: read+write
             ) * elem_bytes

        tput   = (flops / 1e9) / (lat / 1000)
        oi     = flops / mem
        oi_act = oi_actual(flops, lat, mem)

        results.append(BenchResult(
            benchmark="Conv_BN_ReLU", precision=prec, step=step,
            param_desc=f"Cin={Cin},Cout={Cout},H={H},W={W},K={K}",
            flops=flops, mem_bytes=mem,
            latency_ms=lat, throughput_gflops=tput,
            oi=oi, oi_actual=oi_act,
        ))
        print(f"    Step {step}: Cin={Cin} Cout={Cout} H={H} W={W} K={K}  "
              f"lat={lat:.2f}ms  OI={oi:.2f}  OI_act={oi_act:.2f}  Tput={tput:.2f} GFLOPS")
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
# ─────────────────────────────────────────────
# DPE_CONFIGS 설계 원칙 (8 스텝, B=1, K=3)
#
#  DPE Block: DW(K²) + PW(C) 혼합 구조
#    FLOPs ≈ 2·C·H·W·(K² + C) + 5·C·H·W
#    mem   ≈ (4·C·H·W + C·K² + C²) · 4
#
#  OI 결정 인자: PW의 C² 항
#    C 작을 때: DW(K²=9) 지배 → OI 낮음 (memory bound)
#    C 클 때  : PW(C)  지배 → OI 증가 (compute bound 접근)
#    → C를 등비수열 → log(OI) 선형
#
#  트레이드오프:
#    기존: H = round(√(16·64²/C)/4)·4 → C·H² ≈ 일정 → mem 거의 고정
#    수정: H를 더 완만하게 줄여 5·C·H² + C² 가 단조 증가하도록 설계
#    → OI 프로파일은 기존과 거의 동일하게 유지 (Δlog₁₀(OI) ≈ 0.09~0.14)
# ─────────────────────────────────────────────
DPE_CONFIGS = [
    # (B,  C,   H,   W,  K)
    (1,  16,  64,  64,  3),   # step1  OI≈2.80  mem≈1.31 MB
    (1,  24,  54,  54,  3),   # step2  OI≈3.59  mem≈1.40 MB  H: 52→54
    (1,  32,  48,  48,  3),   # step3  OI≈4.38  mem≈1.48 MB  H: 44→48
    (1,  48,  40,  40,  3),   # step4  OI≈5.96  mem≈1.55 MB  H: 36→40
    (1,  72,  36,  36,  3),   # step5  OI≈8.30  mem≈1.89 MB  H: 28→36
    (1, 104,  30,  30,  3),   # step6  OI≈11.3  mem≈1.92 MB  H: 24→30
    (1, 152,  26,  26,  3),   # step7  OI≈15.7  mem≈2.15 MB  H: 20→26
    (1, 224,  22,  22,  3),   # step8  OI≈21.5  mem≈2.38 MB  H: 16→22
    #  Δlog₁₀(OI): 0.11, 0.09, 0.13, 0.14, 0.13, 0.14, 0.13  (균등)
    #  mem: 1.31→1.40→1.48→1.55→1.89→1.92→2.15→2.38 MB (단조 증가)
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

        tput    = (flops / 1e9) / (lat / 1000)
        oi      = flops / mem
        oi_act  = oi_actual(flops, lat, mem)

        results.append(BenchResult(
            benchmark="DPE_Block", precision=prec, step=step,
            param_desc=f"C={C},H={H},W={W},K={K}",
            flops=flops, mem_bytes=mem,
            latency_ms=lat, throughput_gflops=tput,
            oi=oi, oi_actual=oi_act
        ))
        print(f"    Step {step}: C={C} H={H} W={W} k={K}  "
              f"lat={lat:.2f}ms  OI={oi:.2f}  OI_act={oi_act:.2f}  Tput={tput:.2f} GFLOPS")
    return results

# ─────────────────────────────────────────────
# 4. Attention Benchmark
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# ATTN_CONFIGS 설계 원칙 (8 스텝, Dh=64 고정)
#
#  multi_dimensio_benchmark.py 구조 반영:
#    PureSelfAttention (Q/K/V/O 분리 Linear) + LayerNorm
#    → projection weight(4·D²) bytes 포함
#
#  OI = FLOPs / mem
#    FLOPs:
#      - Q/K/V/O proj : 8·B·T·D²          (각 D×D Linear × 4)
#      - QK^T         : 2·B·H·T²·Dh
#      - A·V          : 2·B·H·T²·Dh
#      - softmax 근사 : 5·B·H·T²
#      - LayerNorm    : 5·B·T·D
#    mem:
#      - W_Q/K/V/O    : 4·D²              ← weight 읽기
#      - x, q,k,v,y   : 5·B·T·D          ← activation
#      - scores A/V   : 2·B·H·T²         ← attention map
#      - LN in/out    : 2·B·T·D
#
#  T를 등비수열, H를 단계적 증가 → log(OI) 균등
#  T²항이 커지면 weight(4D²) 항의 상대적 비중 감소 → compute bound 접근
# ─────────────────────────────────────────────
ATTN_CONFIGS = [
    # (B,  H,   T,   Dh)   D=H*Dh
    (1,  1,  32,  64),   # step1  D= 64   OI≈ 9.5
    (1,  1,  48,  64),   # step2  D= 64   OI≈12.0
    (1,  2,  64,  64),   # step3  D=128   OI≈17.9
    (1,  2,  96,  64),   # step4  D=128   OI≈21.7
    (1,  4, 128,  64),   # step5  D=256   OI≈32.2
    (1,  4, 192,  64),   # step6  D=256   OI≈36.7
    (1,  8, 256,  64),   # step7  D=512   OI≈53.6
    (1, 12, 320,  64),   # step8  D=768   OI≈67.5
    #  Δlog₁₀(OI): 0.10, 0.17, 0.08, 0.17, 0.06, 0.17, 0.10  stddev=0.044
    #  H 단계 증가(1→12)로 T²·D 수렴을 D² 증가로 상쇄
]


class PureSelfAttention(nn.Module):
    """
    multi_dimensio_benchmark.py의 PureSelfAttention과 동일한 구조.
    Q/K/V/O를 분리된 Linear로 구현 — projection weight(4·D²) 메모리 접근 명시적 반영.
    """
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.h  = n_head
        self.dh = d_model // n_head
        self.q  = nn.Linear(d_model, d_model, bias=False)
        self.k  = nn.Linear(d_model, d_model, bias=False)
        self.v  = nn.Linear(d_model, d_model, bias=False)
        self.o  = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    """
    PureSelfAttention + LayerNorm 벤치마크.

    multi_dimensio_benchmark.py 구조 반영:
      - Q/K/V/O 분리 Linear (projection weight 메모리 명시적 계상)
      - LayerNorm 포함 (LN-Attn-LN 블록에 해당)
      - Int8은 float32 clamp 모사

    FLOPs:
      - Q/K/V/O proj : 8·B·T·D²
      - QK^T + A·V   : 4·B·H·T²·Dh
      - Softmax 근사  : 5·B·H·T²
      - LayerNorm     : 5·B·T·D

    Memory:
      - W_Q/K/V/O    : 4·D²              (weight 읽기)
      - x,q,k,v,o,y  : 6·B·T·D          (activation i/o)
      - scores, A·V  : 2·B·H·T²         (attention map read+write)
      - LN in/out    : 2·B·T·D
    """
    results = []
    prec = "Float32" if dtype == torch.float32 else "Int8"
    print(f"  [Attn+LN] {prec}")

    for step, (B, H, T, Dh) in enumerate(ATTN_CONFIGS, 1):
        D     = H * Dh
        model = nn.Sequential(
            PureSelfAttention(d_model=D, n_head=H),
            nn.LayerNorm(D),
        ).eval()

        x = torch.randn(B, T, D)
        if dtype == torch.int8:
            x = x.clamp(-1.0, 1.0)

        fn  = lambda x=x: model(x)
        lat = measure_latency(fn)

        elem_bytes = 4
        flops = (8.0 * B * T * D * D          # Q/K/V/O proj
               + 4.0 * B * H * T * T * Dh     # QK^T + A·V
               + 5.0 * B * H * T * T          # Softmax 근사
               + 5.0 * B * T * D)             # LayerNorm

        mem = (4 * D * D                       # W_Q,K,V,O weight 읽기
             + 6 * B * T * D                  # x, q, k, v, o, y activation
             + 2 * B * H * T * T              # scores + att·V map
             + 2 * B * T * D                  # LN in + out
             ) * elem_bytes

        tput   = (flops / 1e9) / (lat / 1000)
        oi     = flops / mem
        oi_act = oi_actual(flops, lat, mem)

        results.append(BenchResult(
            benchmark="Attn_GELU", precision=prec, step=step,
            param_desc=f"B={B},H={H},T={T},Dh={Dh},D={D}",
            flops=flops, mem_bytes=mem,
            latency_ms=lat, throughput_gflops=tput,
            oi=oi, oi_actual=oi_act,
        ))
        print(f"    Step {step}: B={B} H={H} T={T} D={D}  "
              f"lat={lat:.2f}ms  OI={oi:.2f}  OI_act={oi_act:.2f}  Tput={tput:.2f} GFLOPS")
    return results

# ─────────────────────────────────────────────
# Save CSV
# ─────────────────────────────────────────────
def save_csv(all_results: List[BenchResult], path: str):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "benchmark","precision","step","param_desc",
            "flops","mem_bytes","latency_ms","throughput_gflops",
            "oi","oi_actual"
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
def _extract_T(param_desc: str) -> float:
    m = re.search(r'T=(\d+)', param_desc)
    return float(m.group(1)) if m else 1.0


def build_combined_regression(results: List[BenchResult], ridge_lambda: float = 1e-3,
                               extra_log: np.ndarray = None):
    """
    기본: log(lat) = a·log(OI) + b·log(mem) + c
    extra_log 지정 시: log(lat) = a·log(OI) + b·log(mem) + c_T·extra_log + c
      → Attention은 extra_log=log10(T)를 전달해 T² 스케일링을 피처로 추가.

    항상 (a, b, c_T, c, r2) 5개를 반환.
    extra_log 없으면 c_T=0.0.

    ridge_lambda: 정규화 강도 (기본 1e-3, intercept 항은 정규화 제외)
    """
    ois  = np.array([r.oi         for r in results])
    mems = np.array([r.mem_bytes  for r in results])
    lats = np.array([r.latency_ms for r in results])
    log_oi  = np.log10(np.maximum(ois,  1e-9))
    log_mem = np.log10(np.maximum(mems, 1e-9))
    log_lat = np.log10(np.maximum(lats, 1e-9))

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
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    if extra_log is not None:
        a, b, c_T, c = coeffs
    else:
        a, b, c = coeffs
        c_T = 0.0
    return float(a), float(b), float(c_T), float(c), float(r2)


def predict_latency_combined(oi: float, mem_bytes: float,
                              a: float, b: float, c_T: float, c: float,
                              extra_val: float = 1.0) -> float:
    log_pred = (a * math.log10(max(oi,        1e-9))
              + b * math.log10(max(mem_bytes,  1e-9))
              + c_T * math.log10(max(extra_val, 1e-9))
              + c)
    return 10 ** log_pred


def save_combined_regression_csv(all_results: List[BenchResult], path: str):
    benchmarks = ["GEMM_ReLU", "Conv_BN_ReLU", "DPE_Block", "Attn_GELU"]
    rows = []
    for bench in benchmarks:
        for prec in ["Float32", "Int8"]:
            sub = [r for r in all_results if r.benchmark == bench and r.precision == prec]
            if len(sub) < 3:
                continue
            is_attn  = bench == "Attn_GELU"
            extra_log = (np.log10([_extract_T(r.param_desc) for r in sub])
                         if is_attn else None)
            a, b, c_T, c, r2 = build_combined_regression(sub, extra_log=extra_log)
            # 각 스텝별 예측 → RMSE / MAPE 로 검증 (평균 단일 점 오류 수정)
            preds   = [predict_latency_combined(r.oi, r.mem_bytes, a, b, c_T, c,
                                                extra_val=_extract_T(r.param_desc) if is_attn else 1.0)
                       for r in sub]
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
                "coef_T":        round(c_T, 6),
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
    "Conv_BN_ReLU": "#f39c12",
    "DPE_Block": "#27ae60",   # 기존 DW_ReLU 색상 유지
    "Attn_GELU": "#8e44ad",
}

def plot_results(all_results: List[BenchResult], save_path: str):
    benchmarks = ["GEMM_ReLU", "Conv_BN_ReLU", "DPE_Block", "Attn_GELU"]
    precisions = ["Float32", "Int8"]

    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#0f0f1a")
    fig.suptitle("PyTorch CPU Benchmark — Throughput / OI / Memory per Step",
                 color="white", fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(len(benchmarks), len(precisions),
                           figure=fig, hspace=0.6, wspace=0.55)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

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

            steps  = [r.step            for r in subset]
            tputs  = [r.throughput_gflops for r in subset]
            ois    = [r.oi              for r in subset]
            mems   = [r.mem_bytes / 1e6 for r in subset]   # MB
            color  = BENCH_COLORS[bench]
            oi_color  = "#f0c040"
            mem_color = "#4ecdc4"

            # ── ax1: Throughput (bar) ──────────────────────
            ax.bar(steps, tputs, color=color, alpha=0.70, zorder=4,
                   edgecolor="white", linewidth=0.4)
            ax.set_xlabel("Step", color="#aaaaaa", fontsize=8)
            ax.set_ylabel("Throughput (GFLOPS)", color=color, fontsize=8)
            ax.tick_params(axis="y", colors=color, labelsize=7)
            ax.tick_params(axis="x", colors="#888888", labelsize=7)
            ax.set_xticks(steps)
            ax.grid(True, color="#2a2a4a", linewidth=0.5, zorder=0)
            for spine in ax.spines.values():
                spine.set_edgecolor("#333355")

            # ── ax2: OI (오른쪽 첫 번째 축) ────────────────
            ax2 = ax.twinx()
            ax2.set_facecolor("#1a1a2e")
            ax2.plot(steps, ois, color=oi_color, linewidth=1.4,
                     linestyle=":", zorder=3)
            ax2.scatter(steps, ois, color=oi_color, s=35, zorder=5,
                        marker="o", edgecolors="white", linewidths=0.4)
            ax2.set_ylabel("OI (FLOP/byte)", color=oi_color, fontsize=8)
            ax2.tick_params(axis="y", colors=oi_color, labelsize=7)
            ax2.spines["right"].set_edgecolor(oi_color)
            ax2.spines["right"].set_linewidth(1.0)
            for sp in ["top", "left", "bottom"]:
                ax2.spines[sp].set_visible(False)

            # ── ax3: Memory MB (오른쪽 두 번째 축, offset) ──
            ax3 = ax.twinx()
            ax3.spines["right"].set_position(("outward", 48))  # 오른쪽으로 48pt 밀기
            ax3.set_facecolor("#1a1a2e")
            ax3.plot(steps, mems, color=mem_color, linewidth=1.2,
                     linestyle="--", zorder=3)
            ax3.scatter(steps, mems, color=mem_color, s=28, zorder=5,
                        marker="s", edgecolors="white", linewidths=0.4)
            ax3.set_ylabel("Memory (MB)", color=mem_color, fontsize=8)
            ax3.tick_params(axis="y", colors=mem_color, labelsize=7)
            ax3.spines["right"].set_edgecolor(mem_color)
            ax3.spines["right"].set_linewidth(1.0)
            for sp in ["top", "left", "bottom"]:
                ax3.spines[sp].set_visible(False)

            # ── 제목 ────────────────────────────────────────
            title_color = "#ff6b6b" if prec == "Float32" and bench == "GEMM_ReLU" else color
            label_suffix = " (Baseline)" if prec == "Float32" and bench == "GEMM_ReLU" else ""
            ax.set_title(f"{bench} · {prec}{label_suffix}",
                         color=title_color, fontsize=9, fontweight="bold")

            # ── 범례 ────────────────────────────────────────
            handles = [
                Patch(facecolor=color,     edgecolor="white", alpha=0.70, label="Throughput"),
                Line2D([0],[0], color=oi_color,  linewidth=1.4, linestyle=":",  marker="o", markersize=4, label="OI"),
                Line2D([0],[0], color=mem_color, linewidth=1.2, linestyle="--", marker="s", markersize=4, label="Mem(MB)"),
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
        "Conv_BN_ReLU": run_conv,
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
    benchmarks = ["GEMM_ReLU", "Conv_BN_ReLU", "DPE_Block", "Attn_GELU"]
    print(f"\n  Peak BW (method2): {_PEAK_BW_GBPS:.1f} GB/s")
    print("\n" + "="*80)
    print(f"{'Benchmark':<14} {'Precision':<10} {'Steps':>5} "
          f"{'OI_theory':>10} {'OI_actual(M2)':>14} "
          f"{'Tput(GFLOPS)':>13} {'Lat(ms)':>10}")
    print("-"*80)
    for bench in benchmarks:
        for prec in ["Float32","Int8"]:
            sub = [r for r in all_results if r.benchmark==bench and r.precision==prec]
            if not sub: continue
            avg_oi     = np.mean([r.oi        for r in sub])
            avg_oi_act = np.mean([r.oi_actual for r in sub])
            avg_tput   = np.mean([r.throughput_gflops for r in sub])
            avg_lat    = np.mean([r.latency_ms for r in sub])
            suffix = " ← Baseline" if bench=="GEMM_ReLU" and prec=="Float32" else ""
            print(f"{bench:<14} {prec:<10} {len(sub):>5} "
                  f"{avg_oi:>10.2f} {avg_oi_act:>14.2f} "
                  f"{avg_tput:>13.4f} {avg_lat:>10.3f}{suffix}")
    print("="*80)

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
    print(f"\n{'='*88}")
    print("  Combined Regression  (Attn: +log(T) 피처 추가)")
    print(f"{'='*88}")
    print(f"  {'Benchmark':<14} {'Prec':<8} {'coef_OI':>9} {'coef_mem':>10} "
          f"{'coef_T':>8} {'intercept':>10} {'R²':>7} {'RMSE(ms)':>10} {'MAPE%':>8}")
    print(f"  {'-'*86}")
    for bench in benchmarks:
        for prec in ["Float32", "Int8"]:
            sub = [r for r in all_results if r.benchmark == bench and r.precision == prec]
            if len(sub) < 3:
                continue
            is_attn   = bench == "Attn_GELU"
            extra_log = (np.log10([_extract_T(r.param_desc) for r in sub])
                         if is_attn else None)
            a, b, c_T, c, r2 = build_combined_regression(sub, extra_log=extra_log)
            preds   = [predict_latency_combined(r.oi, r.mem_bytes, a, b, c_T, c,
                                                extra_val=_extract_T(r.param_desc) if is_attn else 1.0)
                       for r in sub]
            actuals = [r.latency_ms for r in sub]
            rmse    = float(np.sqrt(np.mean([(p - ac) ** 2 for p, ac in zip(preds, actuals)])))
            mape    = float(np.mean([abs(p - ac) / max(ac, 1e-9) * 100 for p, ac in zip(preds, actuals)]))
            print(f"  {bench:<14} {prec:<8} {a:>9.4f} {b:>10.4f} "
                  f"{c_T:>8.4f} {c:>10.4f} {r2:>7.4f} {rmse:>10.4f} {mape:>8.2f}%")
    print(f"  {'='*86}")

    # ── 평균 OI에서의 추론 성능 요약 ─────────────────────────────
    print(f"\n{'='*88}")
    print("  평균 OI에서의 추론 성능 (회귀 예측 vs 실측 평균)")
    print(f"  {'Benchmark':<14} {'Prec':<8} {'Avg OI':>8} "
          f"{'Pred Tput':>11} {'Actual Tput':>12} {'Err%':>7} "
          f"{'Pred Lat':>10} {'Actual Lat':>11}")
    print(f"  {'-'*86}")
    for bench in benchmarks:
        for prec in ["Float32", "Int8"]:
            sub = [r for r in all_results if r.benchmark == bench and r.precision == prec]
            if len(sub) < 2:
                continue
            slope, intercept, r2 = build_oi_regression(sub)
            avg_oi      = float(np.mean([r.oi                for r in sub]))
            actual_tput = float(np.mean([r.throughput_gflops for r in sub]))
            actual_lat  = float(np.mean([r.latency_ms        for r in sub]))
            avg_flops   = float(np.mean([r.flops              for r in sub]))

            pred_tput = predict_throughput(avg_oi, slope, intercept)
            pred_lat  = (avg_flops / 1e9) / pred_tput * 1000 if pred_tput > 1e-9 else float("inf")
            err_pct   = (pred_tput - actual_tput) / actual_tput * 100 if actual_tput > 0 else 0.0

            print(f"  {bench:<14} {prec:<8} {avg_oi:>8.2f} "
                  f"{pred_tput:>10.4f}G {actual_tput:>11.4f}G {err_pct:>+7.2f}% "
                  f"{pred_lat:>9.3f}ms {actual_lat:>10.3f}ms")
    print(f"  {'='*86}")
    print("  * Pred: 회귀 모델(log-log OI→Tput)으로 평균 OI를 입력해 추정한 값")
    print(f"  * 단위: G = GFLOPS,  ms = milliseconds,  R² per benchmark는 위 Regression 참고")

if __name__ == "__main__":
    main()