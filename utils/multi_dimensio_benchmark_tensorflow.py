#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_dimensio_benchmark_tensorflow.py
TensorFlow (+ TFLite) 버전의 멀티 차원 벤치마크

PyTorch 원본 대비 주요 차이:
  - nn.Module       → tf.keras.layers.Layer / tf.keras.Sequential
  - nn.Linear       → tf.keras.layers.Dense
  - nn.Conv2d       → tf.keras.layers.Conv2D  (channels_last: B,H,W,C)
  - nn.BatchNorm2d  → tf.keras.layers.BatchNormalization
  - nn.LayerNorm    → tf.keras.layers.LayerNormalization
  - torch.jit.trace → @tf.function (JIT 컴파일, USE_TF_FUNCTION)
  - optimize_for_mobile → TFLite 변환   (USE_TFLITE)
  - CUDA sync 없음  (CPU 타겟)

데이터 포맷:
  Conv/DW : (B, H, W, C)  ← TF channels_last 기본값
  GEMM    : (M, K)
  Attention: (B, T, D)
"""
import time, math, os
import numpy as np
import tensorflow as tf

# ============================================================
# 전역 설정
# ============================================================
TF_DEVICE = "/CPU:0"          # "/GPU:0" 으로 변경 가능
RUNS   = 128
N_STEPS = 10
BYTES_PER = 4
REBUILD_MODEL_EVERY_RUN = False  # TFLite 변환 오버헤드로 False 권장

USE_TF_FUNCTION = True   # @tf.function JIT 컴파일 (torch.jit.trace 대응)
USE_TFLITE      = True   # TFLite 변환      (optimize_for_mobile 대응)
SAVE_TFLITE     = False  # True이면 .tflite 저장 (실제 모바일 배포용)
TFLITE_SAVE_DIR = "."    # .tflite 저장 디렉터리

# ============================================================
# 계단형 비율(층 수 제어)
# ============================================================
RATIOS = {
    "conv":       [1],
    "dw":         [1],
    "gemm":       [1],
    "attn":       [1],
    "ln_attn_ln": [1],
}

# ============================================================
# 크기 설정 (Fit ~ Miss)
# ============================================================
SIZES_FIT = {
    "gemm":       dict(M=256,  K=256,  N=256),
    "conv":       dict(B=1, Cin=6,  Cout=12, H=224, W=224, K=3),
    "dw":         dict(B=1, C=8,   H=64,  W=64,  K=3),
    "vit":        dict(B=1, T=57,  H=4,   Dh=64),
    "ln_attn_ln": dict(B=1, T=160, H=4,   Dh=64),
}

SIZES_MISS = {
    "gemm":       dict(M=1024, K=1024, N=1024),
    "conv":       dict(B=1, Cin=24, Cout=48, H=224, W=224, K=3),
    "dw":         dict(B=1, C=96,  H=256, W=256, K=3),
    "vit":        dict(B=1, T=632, H=8,   Dh=64),
    "ln_attn_ln": dict(B=1, T=632, H=8,   Dh=64),
}

# ============================================================
# FLOPs / Bytes 유틸 (수식은 레이아웃 무관으로 동일)
# ============================================================
def flops_conv3x3(B, Cin, Cout, H, W, K=3): return 2*B*Cout*H*W*Cin*K*K
def bytes_conv3x3(B, Cin, Cout, H, W, K=3): return (B*Cin*H*W + B*Cout*H*W + Cout*Cin*K*K)*BYTES_PER
def flops_dw(B, C, H, W, K=3):  return 2*B*C*H*W*K*K
def bytes_dw(B, C, H, W, K=3):  return (B*C*H*W + B*C*H*W + C*K*K)*BYTES_PER
def flops_gemm(M, K, N):         return 2*M*K*N
def bytes_gemm(M, K, N):         return (M*K + K*N + 2*M*N)*BYTES_PER

def flops_attn_total(B, T, H, Dh):
    D = H * Dh
    return 8.0*B*T*D*D + 4.0*B*H*T*T*Dh

def flops_bytes_layernorm(B, T, D):
    f = 5.0 * B * T * D
    b = ((B*T*D) + (B*T*D) + 2*D) * BYTES_PER
    return f, b

def bytes_attn_total(B, T, H, Dh):
    D = H * Dh
    x_io   = B*T*D + B*T*D
    qkv_o  = 3*B*T*D + B*T*D
    scores = B*H*T*T * 2
    return (x_io + qkv_o + scores) * BYTES_PER

# ============================================================
# TFLite / tf.function 변환 헬퍼
# ============================================================
def to_tflite_runner(model: tf.keras.Model,
                     example_input: np.ndarray,
                     save_name: str = None):
    """
    1) @tf.function + ConcreteFunction 생성
    2) TFLiteConverter 변환  (optimize_for_mobile 대응)
    3) SAVE_TFLITE=True 시 .tflite 저장
    4) pre-allocated Interpreter runner 반환
    """
    @tf.function(input_signature=[
        tf.TensorSpec(shape=example_input.shape, dtype=tf.float32)
    ])
    def predict(x):
        return model(x, training=False)

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [predict.get_concrete_function()]
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    if SAVE_TFLITE and save_name:
        path = os.path.join(TFLITE_SAVE_DIR, f"{save_name}.tflite")
        with open(path, "wb") as f:
            f.write(tflite_model)
        print(f"  [저장] {path}")

    # Interpreter 사전 할당 → 측정 중 오버헤드 최소화
    interp = tf.lite.Interpreter(model_content=tflite_model)
    interp.allocate_tensors()
    inp_idx = interp.get_input_details()[0]["index"]
    out_idx = interp.get_output_details()[0]["index"]

    def runner(x_np: np.ndarray):
        interp.set_tensor(inp_idx, x_np)
        interp.invoke()
        return interp.get_tensor(out_idx)

    return runner


def to_tf_function_runner(model: tf.keras.Model,
                           example_input: np.ndarray):
    """@tf.function 컴파일 (TFLite 미사용 시, torch.jit.trace 대응)"""
    @tf.function(input_signature=[
        tf.TensorSpec(shape=example_input.shape, dtype=tf.float32)
    ])
    def predict(x):
        return model(x, training=False)

    predict(tf.constant(example_input))   # graph tracing warm-up
    return lambda x: predict(tf.constant(x))


def make_runner(model: tf.keras.Model,
                example_input: np.ndarray,
                save_name: str = None):
    """
    플래그에 따라 최적 runner 반환.
    USE_TFLITE > USE_TF_FUNCTION > 일반 eager 순으로 적용.
    """
    if USE_TFLITE:
        return to_tflite_runner(model, example_input, save_name)
    elif USE_TF_FUNCTION:
        return to_tf_function_runner(model, example_input)
    else:
        return lambda x: model(tf.constant(x), training=False)


def run_model(runner, x_np: np.ndarray, runs: int = RUNS) -> float:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = runner(x_np)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times)

# ============================================================
# 모듈 정의
# ============================================================

class PureSelfAttention(tf.keras.layers.Layer):
    """Q/K/V/O 분리형 Self-Attention (PyTorch PureSelfAttention 대응)"""

    def __init__(self, d_model: int, n_head: int, **kwargs):
        super().__init__(**kwargs)
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.h  = n_head
        self.dh = d_model // n_head
        self.D  = d_model
        self.scale = math.sqrt(self.dh)
        self.q = tf.keras.layers.Dense(d_model, use_bias=False)
        self.k = tf.keras.layers.Dense(d_model, use_bias=False)
        self.v = tf.keras.layers.Dense(d_model, use_bias=False)
        self.o = tf.keras.layers.Dense(d_model, use_bias=False)

    def call(self, x, training=False):
        # x: [B, T, D]
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]
        H, Dh, D = self.h, self.dh, self.D

        # Q/K/V 투사 → [B, H, T, Dh]
        q = tf.transpose(tf.reshape(self.q(x), [B, T, H, Dh]), [0, 2, 1, 3])
        k = tf.transpose(tf.reshape(self.k(x), [B, T, H, Dh]), [0, 2, 1, 3])
        v = tf.transpose(tf.reshape(self.v(x), [B, T, H, Dh]), [0, 2, 1, 3])

        # Attention score → [B, H, T, T]
        scores = tf.matmul(q, k, transpose_b=True) / self.scale
        att    = tf.nn.softmax(scores, axis=-1)

        # A·V → [B, H, T, Dh] → [B, T, D]
        y = tf.matmul(att, v)
        y = tf.reshape(tf.transpose(y, [0, 2, 1, 3]), [B, T, D])
        return self.o(y)


class LNAttnLNBlock(tf.keras.layers.Layer):
    """LayerNorm → Attention → LayerNorm (PyTorch LNAttnLNBlock 대응)"""

    def __init__(self, d_model: int, n_head: int, eps: float = 1e-5, **kwargs):
        super().__init__(**kwargs)
        self.ln1  = tf.keras.layers.LayerNormalization(epsilon=eps)
        self.attn = PureSelfAttention(d_model, n_head)
        self.ln2  = tf.keras.layers.LayerNormalization(epsilon=eps)

    def call(self, x, training=False):
        x = self.ln1(x)
        x = self.attn(x, training=training)
        return self.ln2(x)

# ============================================================
# 벤치 함수 — TensorFlow 버전
# ============================================================

def bench_conv(B=1, Cin=3, Cout=8, H=64, W=64, K=3,
               runs=RUNS, device=TF_DEVICE):
    """단일 Conv2D 레이어 벤치마크"""
    with tf.device(device):
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(Cout, K, padding="same",
                                   use_bias=False, data_format="channels_last")
        ])
        # TF는 channels_last: (B, H, W, C)
        x_np  = np.random.randn(B, H, W, Cin).astype(np.float32)
        runner = make_runner(model, x_np,
                             save_name=f"conv_single_{B}_{Cin}_{Cout}")
        avg = run_model(runner, x_np, runs)

    f = 2 * B * Cin * Cout * K * K * H * W
    b = (B * Cin * H * W + B * Cout * H * W) * 4
    g = f / avg / 1e9
    return avg, f, b, g


def bench_conv_block(B, Cin, Cout, H, W, K=3, runs=RUNS, device=TF_DEVICE):
    ratios = RATIOS["conv"]

    def build_net():
        layers = []
        in_ch = Cin
        for _ in ratios:
            out_ch = max(1, int(Cout))
            layers += [
                tf.keras.layers.Conv2D(out_ch, K, padding="same",
                                       use_bias=False, data_format="channels_last"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.ReLU(),
            ]
            in_ch = out_ch
        return tf.keras.Sequential(layers)

    x_np = np.random.randn(B, H, W, Cin).astype(np.float32)

    with tf.device(device):
        if REBUILD_MODEL_EVERY_RUN:
            times = []
            for _ in range(runs):
                runner = make_runner(build_net(), x_np)
                times.append(run_model(runner, x_np, 1))
            avg = sum(times) / len(times)
        else:
            runner = make_runner(build_net(), x_np,
                                 save_name=f"conv_block_{B}_{Cin}_{Cout}")
            avg = run_model(runner, x_np, runs)

    f = flops_conv3x3(B, Cin, Cout, H, W, K) * len(ratios)
    b = bytes_conv3x3(B, Cin, Cout, H, W, K) * len(ratios)
    g = f / avg / 1e9
    return avg, f, b, g


def bench_depthwise_block(B, C, H, W, K=3, runs=RUNS, device=TF_DEVICE):
    ratios = RATIOS["dw"]

    def build_net():
        layers = []
        for _ in ratios:
            layers += [
                # TF DepthwiseConv2D = PyTorch Conv2d(groups=in_channels)
                tf.keras.layers.DepthwiseConv2D(K, padding="same", use_bias=False),
                tf.keras.layers.ReLU(),
            ]
        return tf.keras.Sequential(layers)

    x_np = np.random.randn(B, H, W, C).astype(np.float32)

    with tf.device(device):
        if REBUILD_MODEL_EVERY_RUN:
            times = []
            for _ in range(runs):
                runner = make_runner(build_net(), x_np)
                times.append(run_model(runner, x_np, 1))
            avg = sum(times) / len(times)
        else:
            runner = make_runner(build_net(), x_np,
                                 save_name=f"dw_block_{B}_{C}")
            avg = run_model(runner, x_np, runs)

    f = flops_dw(B, C, H, W, K) * len(ratios)
    b = bytes_dw(B, C, H, W, K) * len(ratios)
    g = f / avg / 1e9
    return avg, f, b, g


def bench_gemm_block(M, K, N, runs=RUNS, device=TF_DEVICE):
    ratios = RATIOS["gemm"]

    def build_net():
        layers = []
        in_dim = K
        for r in ratios:
            out_dim = max(1, int(N * (0.8 + 0.1*r)))
            layers += [
                tf.keras.layers.Dense(out_dim, use_bias=True),
                tf.keras.layers.ReLU(),
                tf.keras.layers.Softmax(axis=-1),
            ]
            in_dim = out_dim
        return tf.keras.Sequential(layers)

    x_np = np.random.randn(M, K).astype(np.float32)

    with tf.device(device):
        if REBUILD_MODEL_EVERY_RUN:
            times = []
            for _ in range(runs):
                runner = make_runner(build_net(), x_np)
                times.append(run_model(runner, x_np, 1))
            avg = sum(times) / len(times)
        else:
            runner = make_runner(build_net(), x_np,
                                 save_name=f"gemm_block_{M}_{K}_{N}")
            avg = run_model(runner, x_np, runs)

    f = flops_gemm(M, K, N) * len(ratios)
    b = bytes_gemm(M, K, N) * len(ratios)
    g = f / avg / 1e9
    return avg, f, b, g


def bench_attention_block(B, T, H, Dh, runs=RUNS, device=TF_DEVICE):
    ratios = RATIOS["attn"]
    D = H * Dh

    def build_net():
        layers = []
        for _ in ratios:
            layers += [
                PureSelfAttention(d_model=D, n_head=H),
                tf.keras.layers.LayerNormalization(),
                tf.keras.layers.ReLU(),
            ]
        return tf.keras.Sequential(layers)

    x_np = np.random.randn(B, T, D).astype(np.float32)

    with tf.device(device):
        if REBUILD_MODEL_EVERY_RUN:
            times = []
            for _ in range(runs):
                runner = make_runner(build_net(), x_np)
                times.append(run_model(runner, x_np, 1))
            avg = sum(times) / len(times)
        else:
            runner = make_runner(build_net(), x_np,
                                 save_name=f"attn_block_{B}_{T}_{H}_{Dh}")
            avg = run_model(runner, x_np, runs)

    f = flops_attn_total(B, T, H, Dh) * len(ratios)
    b = bytes_attn_total(B, T, H, Dh) * len(ratios)
    g = f / avg / 1e9
    return avg, f, b, g


def bench_ln_attn_ln_block(B, T, H, Dh, runs=RUNS, device=TF_DEVICE):
    ratios = RATIOS["ln_attn_ln"]
    D = H * Dh

    def build_net():
        layers = []
        for _ in ratios:
            layers += [LNAttnLNBlock(d_model=D, n_head=H)]
        return tf.keras.Sequential(layers)

    x_np = np.random.randn(B, T, D).astype(np.float32)

    with tf.device(device):
        if REBUILD_MODEL_EVERY_RUN:
            times = []
            for _ in range(runs):
                runner = make_runner(build_net(), x_np)
                times.append(run_model(runner, x_np, 1))
            avg = sum(times) / len(times)
        else:
            runner = make_runner(build_net(), x_np,
                                 save_name=f"ln_attn_ln_{B}_{T}_{H}")
            avg = run_model(runner, x_np, runs)

    f_ln1, b_ln1 = flops_bytes_layernorm(B, T, D)
    f_attn        = flops_attn_total(B, T, H, Dh)
    b_attn        = bytes_attn_total(B, T, H, Dh)
    f_ln2, b_ln2  = flops_bytes_layernorm(B, T, D)
    f = (f_ln1 + f_attn + f_ln2) * len(ratios)
    b = (b_ln1 + b_attn + b_ln2) * len(ratios)
    g = f / avg / 1e9
    return avg, f, b, g

# ============================================================
# 스케일 보간 & 실행
# ============================================================
def scale_linear(v_fit, v_miss, step, steps):
    return int(round(v_fit + (v_miss - v_fit) * (step / (steps - 1))))

def interpolate_cfg(cfg_fit, cfg_miss, step, steps):
    return {k: scale_linear(cfg_fit[k], cfg_miss[k], step, steps) for k in cfg_fit}


def run_scaled_suite():
    ops = [
        #("DEPTH",       bench_depthwise_block,   "dw"),
        ("CONV",        bench_conv_block,         "conv"),
        #("GEMM",        bench_gemm_block,          "gemm"),
        #("ATTN",        bench_attention_block,    "attn"),
        #("LN-ATTN-LN",  bench_ln_attn_ln_block,  "ln_attn_ln"),
    ]
    print(f"[TensorFlow]  REBUILD={REBUILD_MODEL_EVERY_RUN}  "
          f"USE_TF_FUNCTION={USE_TF_FUNCTION}  "
          f"USE_TFLITE={USE_TFLITE}  SAVE_TFLITE={SAVE_TFLITE}")
    print(f"{'OP':<12}{'Step':<5}{'GFLOP/s':>10}{'GFLOPs':>10}"
          f"{'GBytes':>10}{'ms':>12}")
    for name, fn, key in ops:
        for step in range(N_STEPS):
            cfg = interpolate_cfg(SIZES_FIT[key], SIZES_MISS[key], step, N_STEPS)
            avg, f, b, g = fn(**cfg)
            print(f"{name:<12}{step+1:<5}{g:10.2f}{f/1e9:10.3f}"
                  f"{b/1e9:10.3f}{avg*1000:12.2f}")


if __name__ == "__main__":
    run_scaled_suite()
