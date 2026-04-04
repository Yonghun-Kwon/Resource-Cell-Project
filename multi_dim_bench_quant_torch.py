#!/usr/bin/env python3
"""
GEMM Quantized Sweep Benchmark
- FP32 vs INT8(Dynamic) vs INT8(Static) 성능 비교
- M, K, N을 선형 스케일로 스윕
- CPU 전용 (PyTorch quantization은 CPU 지원)
"""

import torch
import torch.nn as nn
import torch.quantization as tq
import time

# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
DEVICE    = "cpu"
RUNS      = 128
N_STEPS   = 10
BYTES_FP32 = 4
BYTES_INT8 = 1

SIZE_FIT  = dict(M=256,  K=256,  N=256)
SIZE_MISS = dict(M=1024, K=1024, N=1024)


# ─────────────────────────────────────────
# 선형 보간
# ─────────────────────────────────────────
def scale_linear(v_fit: int, v_miss: int, step: int, steps: int) -> int:
    if v_fit == v_miss or steps == 1:
        return v_fit
    return int(round(v_fit + (v_miss - v_fit) * step / (steps - 1)))


def interpolate_cfg(cfg_fit, cfg_miss, step, steps):
    return {k: scale_linear(cfg_fit[k], cfg_miss[k], step, steps) for k in cfg_fit}


# ─────────────────────────────────────────
# FLOPs / Bytes 추정
# ─────────────────────────────────────────
def flops_gemm(M, K, N):
    return 2 * M * K * N

def bytes_fp32(M, K, N):
    # A(MxK) + B(KxN) + C(MxN) read + C(MxN) write
    return (M * K + K * N + 2 * M * N) * BYTES_FP32

def bytes_int8(M, K, N):
    # weight(KxN) INT8, activation(MxK) INT8, output(MxN) FP32
    return (M * K + K * N) * BYTES_INT8 + (2 * M * N) * BYTES_FP32

def arith_intensity(flops, nbytes):
    return flops / nbytes


# ─────────────────────────────────────────
# 모델 준비
# ─────────────────────────────────────────
def make_fp32_model(K: int, N: int) -> nn.Linear:
    return nn.Linear(K, N, bias=False).eval()


def make_dynamic_quant_model(K: int, N: int) -> nn.Module:
    """Dynamic quantization: 가중치 INT8, 활성화는 런타임에 양자화"""
    model = nn.Linear(K, N, bias=False).eval()
    return tq.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)


def make_static_quant_model(K: int, N: int, x_calib: torch.Tensor) -> nn.Module:
    """Static quantization: 가중치+활성화 모두 INT8, calibration 필요"""
    model = nn.Linear(K, N, bias=False).eval()

    model.qconfig = tq.get_default_qconfig("x86")
    tq.prepare(model, inplace=True)

    # calibration: 대표 입력으로 scale/zero_point 결정
    with torch.no_grad():
        model(x_calib)

    tq.convert(model, inplace=True)
    return model


# ─────────────────────────────────────────
# 단일 모델 벤치마크
# ─────────────────────────────────────────
@torch.no_grad()
def bench_model(model: nn.Module, x: torch.Tensor, runs: int = RUNS) -> float:
    """평균 실행 시간(ms) 반환"""
    # 워밍업
    for _ in range(min(8, runs)):
        _ = model(x)

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _  = model(x)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return sum(times) / len(times) * 1e3  # ms


# ─────────────────────────────────────────
# 스텝별 FP32 / Dynamic / Static 비교
# ─────────────────────────────────────────
def bench_all(M: int, K: int, N: int, runs: int = RUNS) -> dict:
    x_fp32 = torch.randn(M, K)

    # ── FP32 ──
    m_fp32  = make_fp32_model(K, N)
    ms_fp32 = bench_model(m_fp32, x_fp32, runs)

    # ── Dynamic INT8 ──
    m_dyn   = make_dynamic_quant_model(K, N)
    ms_dyn  = bench_model(m_dyn, x_fp32, runs)

    # ── Static INT8 ──
    m_sta   = make_static_quant_model(K, N, x_fp32[:1])  # 1샘플로 calib
    ms_sta  = bench_model(m_sta, x_fp32, runs)

    f        = flops_gemm(M, K, N)
    b_fp32   = bytes_fp32(M, K, N)
    b_int8   = bytes_int8(M, K, N)

    return dict(
        M        = M, K = K, N = N,
        ms_fp32  = ms_fp32,
        ms_dyn   = ms_dyn,
        ms_sta   = ms_sta,
        sp_dyn   = ms_fp32 / ms_dyn,   # 스피드업
        sp_sta   = ms_fp32 / ms_sta,
        gflops_fp32 = f / (ms_fp32 * 1e-3) / 1e9,
        gflops_dyn  = f / (ms_dyn  * 1e-3) / 1e9,
        gflops_sta  = f / (ms_sta  * 1e-3) / 1e9,
        ai_fp32  = arith_intensity(f, b_fp32),
        ai_int8  = arith_intensity(f, b_int8),
        flops    = f,
    )


# ─────────────────────────────────────────
# 선형 스윕
# ─────────────────────────────────────────
def gemm_quant_sweep(size_fit  : dict = SIZE_FIT,
                     size_miss : dict = SIZE_MISS,
                     n_steps   : int  = N_STEPS,
                     runs      : int  = RUNS) -> list[dict]:

    W = 115
    print(f"\n{'─'*W}")
    print(f"{'GEMM Quantization Sweep  (FP32 vs Dynamic-INT8 vs Static-INT8)':^{W}}")
    print(f"FIT  : M={size_fit['M']}, K={size_fit['K']}, N={size_fit['N']}")
    print(f"MISS : M={size_miss['M']}, K={size_miss['K']}, N={size_miss['N']}")
    print(f"Steps: {n_steps}  |  Runs/step: {runs}  |  Device: cpu")
    print(f"{'─'*W}")
    print(f"{'Step':>4} {'M':>5} {'K':>5} {'N':>5} "
          f"{'FLOPs(M)':>10} "
          f"{'FP32 ms':>9} {'FP32 GF':>9} "
          f"{'Dyn ms':>9} {'Dyn GF':>8} {'Dyn↑':>6} "
          f"{'Sta ms':>9} {'Sta GF':>8} {'Sta↑':>6} "
          f"{'AI_fp32':>8} {'AI_int8':>8}")
    print(f"{'─'*W}")

    results = []
    for step in range(n_steps):
        cfg = interpolate_cfg(size_fit, size_miss, step, n_steps)
        r   = bench_all(**cfg, runs=runs)

        print(f"{step+1:>4} {r['M']:>5} {r['K']:>5} {r['N']:>5} "
              f"{r['flops']/1e6:>10.1f} "
              f"{r['ms_fp32']:>9.3f} {r['gflops_fp32']:>9.2f} "
              f"{r['ms_dyn']:>9.3f} {r['gflops_dyn']:>8.2f} {r['sp_dyn']:>5.2f}× "
              f"{r['ms_sta']:>9.3f} {r['gflops_sta']:>8.2f} {r['sp_sta']:>5.2f}× "
              f"{r['ai_fp32']:>8.2f} {r['ai_int8']:>8.2f}")

        results.append(r)

    print(f"{'─'*W}")
    print("  Dynamic-INT8 : 가중치 INT8 고정, 활성화 런타임 양자화 (추가 메모리 오버헤드 없음)")
    print("  Static-INT8  : 가중치+활성화 모두 INT8, calibration 필요 (가장 빠른 추론)")
    print("  AI_fp32/int8 : Arithmetic Intensity — int8는 weight 절반 크기로 AI 상승")
    return results


# ─────────────────────────────────────────
# 양자화 오차 확인 (참고용)
# ─────────────────────────────────────────
@torch.no_grad()
def check_quant_error(M=64, K=128, N=64):
    x = torch.randn(M, K)

    m_fp32 = make_fp32_model(K, N)
    m_dyn  = make_dynamic_quant_model(K, N)

    # static: fp32 모델과 동일 가중치로 calibrate
    import copy
    m_sta = copy.deepcopy(m_fp32)
    m_sta.qconfig = tq.get_default_qconfig("x86")
    tq.prepare(m_sta, inplace=True)
    m_sta(x[:1])
    tq.convert(m_sta, inplace=True)

    out_fp32 = m_fp32(x)
    out_dyn  = m_dyn(x)
    out_sta  = m_sta(x)

    err_dyn = (out_fp32 - out_dyn).abs().mean().item()
    err_sta = (out_fp32 - out_sta).abs().mean().item()

    print(f"\n{'─'*45}")
    print(f"{'양자화 오차 확인 (M={M}, K={K}, N={N})':^45}")
    print(f"{'─'*45}")
    print(f"  Dynamic INT8 MAE : {err_dyn:.6f}")
    print(f"  Static  INT8 MAE : {err_sta:.6f}")
    print(f"  출력 범위(FP32)  : [{out_fp32.min():.3f}, {out_fp32.max():.3f}]")
    print(f"{'─'*45}")


# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────
if __name__ == "__main__":
    check_quant_error()
    results = gemm_quant_sweep()
