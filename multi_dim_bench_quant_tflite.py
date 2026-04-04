#!/usr/bin/env python3
"""
GEMM Quantized Sweep Benchmark (TensorFlow Lite 버전)
- FP32 vs INT8(Dynamic) vs INT8(Full Integer) 성능 비교
- M, K, N을 선형 스케일로 스윕
- CPU 전용 (TFLite는 라즈베리파이에서도 동작)
"""

import numpy as np
import tensorflow as tf
import time
import tempfile
import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')


# ─────────────────────────────────────────
# 설정
# ─────────────────────────────────────────
RUNS      = 128
N_STEPS   = 10
BYTES_FP32 = 4
BYTES_INT8 = 1

SIZE_FIT  = dict(M=256,  K=256,  N=256)
SIZE_MISS = dict(M=1024, K=1024, N=1024)


# ─────────────────────────────────────────
# 선형 보간
# ─────────────────────────────────────────
def scale_linear(v_fit, v_miss, step, steps):
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
    return (M * K + K * N + 2 * M * N) * BYTES_FP32

def bytes_int8(M, K, N):
    return (M * K + K * N) * BYTES_INT8 + (2 * M * N) * BYTES_FP32

def arith_intensity(flops, nbytes):
    return flops / nbytes


# ─────────────────────────────────────────
# TFLite 모델 생성
# ─────────────────────────────────────────
def make_keras_model(K, N):
    """Dense 레이어 하나짜리 Keras 모델"""
    inp = tf.keras.Input(shape=(K,), batch_size=1)
    out = tf.keras.layers.Dense(N, use_bias=False)(inp)
    model = tf.keras.Model(inp, out)
    return model


def convert_fp32(model, K):
    """FP32 TFLite 변환"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    return tflite_model


def convert_dynamic_int8(model, K):
    """Dynamic Range Quantization (가중치 INT8)"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    return tflite_model


def convert_full_int8(model, K):
    def representative_dataset():
        for _ in range(200):  # 샘플 수 늘리기
            yield [np.random.randn(1, K).astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS  # fallback 추가
    ]
    converter.inference_input_type  = tf.float32
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()
    return tflite_model

# ─────────────────────────────────────────
# TFLite 인터프리터 생성
# ─────────────────────────────────────────
def make_interpreter(tflite_model):
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()
    return interpreter


# ─────────────────────────────────────────
# 단일 인터프리터 벤치마크
# ─────────────────────────────────────────
def bench_interpreter(interpreter, x_input, runs=RUNS):
    """평균 실행 시간(ms) 반환"""
    input_details  = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # 워밍업
    for _ in range(min(8, runs)):
        interpreter.set_tensor(input_details[0]['index'], x_input)
        interpreter.invoke()

    times = []
    for _ in range(runs):
        interpreter.set_tensor(input_details[0]['index'], x_input)
        t0 = time.perf_counter()
        interpreter.invoke()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return sum(times) / len(times) * 1e3  # ms


# ─────────────────────────────────────────
# 스텝별 FP32 / Dynamic / Full INT8 비교
# ─────────────────────────────────────────
def bench_all(M, K, N, runs=RUNS):
    x_input = np.random.randn(1, K).astype(np.float32)  # batch=1 고정

    model = make_keras_model(K, N)

    # ── FP32 ──
    interp_fp32 = make_interpreter(convert_fp32(model, K))
    ms_fp32     = bench_interpreter(interp_fp32, x_input, runs)

    # ── Dynamic INT8 ──
    interp_dyn  = make_interpreter(convert_dynamic_int8(model, K))
    ms_dyn      = bench_interpreter(interp_dyn, x_input, runs)

    # ── Full INT8 ──
    interp_sta  = make_interpreter(convert_full_int8(model, K))
    ms_sta      = bench_interpreter(interp_sta, x_input, runs)

    f       = flops_gemm(M, K, N)
    b_fp32  = bytes_fp32(M, K, N)
    b_int8  = bytes_int8(M, K, N)

    return dict(
        M=M, K=K, N=N,
        ms_fp32=ms_fp32,
        ms_dyn=ms_dyn,
        ms_sta=ms_sta,
        sp_dyn=ms_fp32 / ms_dyn,
        sp_sta=ms_fp32 / ms_sta,
        gflops_fp32=f / (ms_fp32 * 1e-3) / 1e9,
        gflops_dyn =f / (ms_dyn  * 1e-3) / 1e9,
        gflops_sta =f / (ms_sta  * 1e-3) / 1e9,
        ai_fp32=arith_intensity(f, b_fp32),
        ai_int8=arith_intensity(f, b_int8),
        flops=f,
    )


# ─────────────────────────────────────────
# 선형 스윕
# ─────────────────────────────────────────
def gemm_quant_sweep(size_fit=SIZE_FIT, size_miss=SIZE_MISS,
                     n_steps=N_STEPS, runs=RUNS):
    W = 115
    print(f"\n{'─'*W}")
    print(f"{'GEMM Quantization Sweep  (FP32 vs Dynamic-INT8 vs Full-INT8) [TFLite]':^{W}}")
    print(f"FIT  : M={size_fit['M']}, K={size_fit['K']}, N={size_fit['N']}")
    print(f"MISS : M={size_miss['M']}, K={size_miss['K']}, N={size_miss['N']}")
    print(f"Steps: {n_steps}  |  Runs/step: {runs}  |  Device: cpu")
    print(f"{'─'*W}")
    print(f"{'Step':>4} {'M':>5} {'K':>5} {'N':>5} "
          f"{'FLOPs(M)':>10} "
          f"{'FP32 ms':>9} {'FP32 GF':>9} "
          f"{'Dyn ms':>9} {'Dyn GF':>8} {'Dyn↑':>6} "
          f"{'Full ms':>9} {'Full GF':>8} {'Full↑':>6} "
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
    print("  Dynamic-INT8 : 가중치만 INT8, 활성화는 FP32 (추가 캘리브레이션 불필요)")
    print("  Full-INT8    : 가중치+활성화 모두 INT8, 캘리브레이션 필요 (라즈베리파이에서 가장 빠름)")
    print("  AI_fp32/int8 : Arithmetic Intensity — int8는 weight 절반 크기로 AI 상승")
    return results


# ─────────────────────────────────────────
# 양자화 오차 확인
# ─────────────────────────────────────────
def check_quant_error(M=1, K=128, N=64):
    x_input = np.random.randn(M, K).astype(np.float32)
    model   = make_keras_model(K, N)

    interp_fp32 = make_interpreter(convert_fp32(model, K))
    interp_dyn  = make_interpreter(convert_dynamic_int8(model, K))
    interp_sta  = make_interpreter(convert_full_int8(model, K))

    def run(interp):
        inp = interp.get_input_details()
        out = interp.get_output_details()
        interp.set_tensor(inp[0]['index'], x_input)
        interp.invoke()
        return interp.get_tensor(out[0]['index'])

    out_fp32 = run(interp_fp32)
    out_dyn  = run(interp_dyn)
    out_sta  = run(interp_sta)

    err_dyn = np.abs(out_fp32 - out_dyn).mean()
    err_sta = np.abs(out_fp32 - out_sta).mean()

    print(f"\n{'─'*45}")
    print(f"{'양자화 오차 확인 (M={M}, K={K}, N={N})':^45}")
    print(f"{'─'*45}")
    print(f"  Dynamic INT8 MAE : {err_dyn:.6f}")
    print(f"  Full    INT8 MAE : {err_sta:.6f}")
    print(f"  출력 범위(FP32)  : [{out_fp32.min():.3f}, {out_fp32.max():.3f}]")
    print(f"{'─'*45}")


# ─────────────────────────────────────────
# 실행
# ─────────────────────────────────────────
if __name__ == "__main__":
    # TF 로그 억제
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')

    check_quant_error()
    results = gemm_quant_sweep()
