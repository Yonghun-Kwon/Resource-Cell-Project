#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
multi_dimensio_benchmark_tflite.py
TensorFlow Lite 전용 벤치마크

TF 버전 대비 차이:
  - 항상 TFLite Interpreter 로만 추론 (eager / tf.function 없음)
  - 양자화 모드 선택 가능: FP32 | FP16 | INT8(dynamic-range)
  - 모델 크기(KB) 출력 → 엣지/모바일 배포 시 참고
  - TFLiteModel 클래스로 변환·실행을 캡슐화

데이터 포맷:
  Conv/DW  : (B, H, W, C)  ← channels_last
  GEMM     : (M, K)
  Attention: (B, T, D)
"""
import time, math, os
import numpy as np
import tensorflow as tf

# ============================================================
# 전역 설정
# ============================================================
RUNS    = 128
N_STEPS = 10
BYTES_PER = 4
REBUILD_MODEL_EVERY_RUN = False   # True이면 매 실행마다 재변환 (느림)

# 양자화 모드: "fp32" | "fp16" | "int8"
QUANTIZATION = "fp32"

SAVE_TFLITE     = False   # True이면 .tflite 파일 저장
TFLITE_SAVE_DIR = "."     # 저장 경로

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
# FLOPs / Bytes 유틸
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
    b = ((B*T*D)*2 + 2*D) * BYTES_PER
    return f, b

def bytes_attn_total(B, T, H, Dh):
    D = H * Dh
    return ((B*T*D*2) + (B*T*D*4) + (B*H*T*T*2)) * BYTES_PER

# ============================================================
# TFLite 핵심: 변환 + 실행 래퍼
# ============================================================

def _apply_quantization(converter: tf.lite.TFLiteConverter,
                         mode: str,
                         rep_data_fn=None) -> tf.lite.TFLiteConverter:
    """양자화 옵션 적용"""
    if mode == "fp32":
        pass  # 변환 없음 (FP32 그대로)

    elif mode == "fp16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    elif mode == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # representative_dataset 없이 dynamic-range quantization 적용
        # (calibration 불필요, 가중치만 INT8 → 활성값은 FP32)
        if rep_data_fn is not None:
            converter.representative_dataset = rep_data_fn
            converter.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8
            ]
            converter.inference_input_type  = tf.int8
            converter.inference_output_type = tf.int8

    return converter


class TFLiteModel:
    """
    tf.keras.Model → TFLite 변환 & 추론 래퍼.

    속성:
        model_bytes  : 변환된 .tflite 바이너리 크기 (bytes)
        model_kb     : 위를 KB 단위로
    """
    def __init__(self, keras_model: tf.keras.Model,
                 example_input: np.ndarray,
                 quantization: str = QUANTIZATION,
                 save_name: str = None):

        # 1) ConcreteFunction 생성
        @tf.function(input_signature=[
            tf.TensorSpec(shape=example_input.shape, dtype=tf.float32)
        ])
        def predict(x):
            return keras_model(x, training=False)

        # 2) TFLite 변환
        converter = tf.lite.TFLiteConverter.from_concrete_functions(
            [predict.get_concrete_function()]
        )
        converter = _apply_quantization(converter, quantization)
        tflite_bytes = converter.convert()

        self.model_bytes = len(tflite_bytes)
        self.model_kb    = self.model_bytes / 1024.0
        self._quant      = quantization

        # 3) 파일 저장 (옵션)
        if SAVE_TFLITE and save_name:
            path = os.path.join(TFLITE_SAVE_DIR, f"{save_name}_{quantization}.tflite")
            with open(path, "wb") as f:
                f.write(tflite_bytes)
            print(f"  [저장] {path}  ({self.model_kb:.1f} KB)")

        # 4) Interpreter 사전 할당 → 측정 중 할당 오버헤드 제거
        interp = tf.lite.Interpreter(model_content=tflite_bytes)
        interp.allocate_tensors()
        self._inp_idx = interp.get_input_details()[0]["index"]
        self._out_idx = interp.get_output_details()[0]["index"]
        self._inp_dtype = interp.get_input_details()[0]["dtype"]
        self._interp  = interp

    def __call__(self, x_np: np.ndarray) -> np.ndarray:
        """TFLite Interpreter로 단일 추론"""
        inp = x_np.astype(self._inp_dtype)
        self._interp.set_tensor(self._inp_idx, inp)
        self._interp.invoke()
        return self._interp.get_tensor(self._out_idx)


def run_tflite(model: TFLiteModel, x_np: np.ndarray, runs: int = RUNS) -> float:
    """TFLite 반복 추론 → 평균 소요 시간(초) 반환"""
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model(x_np)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times)

# ============================================================
# Keras 모듈 정의
# ============================================================

class PureSelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model: int, n_head: int, **kw):
        super().__init__(**kw)
        assert d_model % n_head == 0
        self.h    = n_head
        self.dh   = d_model // n_head
        self.D    = d_model
        self.scale = math.sqrt(self.dh)
        self.q = tf.keras.layers.Dense(d_model, use_bias=False)
        self.k = tf.keras.layers.Dense(d_model, use_bias=False)
        self.v = tf.keras.layers.Dense(d_model, use_bias=False)
        self.o = tf.keras.layers.Dense(d_model, use_bias=False)

    def call(self, x, training=False):
        B = tf.shape(x)[0]; T = tf.shape(x)[1]
        H, Dh, D = self.h, self.dh, self.D
        q = tf.transpose(tf.reshape(self.q(x), [B, T, H, Dh]), [0, 2, 1, 3])
        k = tf.transpose(tf.reshape(self.k(x), [B, T, H, Dh]), [0, 2, 1, 3])
        v = tf.transpose(tf.reshape(self.v(x), [B, T, H, Dh]), [0, 2, 1, 3])
        att = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / self.scale, axis=-1)
        y = tf.reshape(tf.transpose(tf.matmul(att, v), [0, 2, 1, 3]), [B, T, D])
        return self.o(y)


class LNAttnLNBlock(tf.keras.layers.Layer):
    def __init__(self, d_model: int, n_head: int, eps: float = 1e-5, **kw):
        super().__init__(**kw)
        self.ln1  = tf.keras.layers.LayerNormalization(epsilon=eps)
        self.attn = PureSelfAttention(d_model, n_head)
        self.ln2  = tf.keras.layers.LayerNormalization(epsilon=eps)

    def call(self, x, training=False):
        return self.ln2(self.attn(self.ln1(x), training=training))

# ============================================================
# 벤치 함수 — TFLite 전용
#   반환: (avg_sec, flops, bytes_mem, gflops_per_sec, model_kb)
# ============================================================

def bench_conv(B=1, Cin=3, Cout=8, H=64, W=64, K=3,
               runs=RUNS, quant=QUANTIZATION):
    keras_model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(Cout, K, padding="same",
                               use_bias=False, data_format="channels_last")
    ])
    x_np  = np.random.randn(B, H, W, Cin).astype(np.float32)
    tfl   = TFLiteModel(keras_model, x_np, quant,
                        save_name=f"conv_single_{B}_{Cin}_{Cout}")
    avg   = run_tflite(tfl, x_np, runs)
    f = 2 * B * Cin * Cout * K * K * H * W
    b = (B*Cin*H*W + B*Cout*H*W) * 4
    return avg, f, b, f/avg/1e9, tfl.model_kb


def bench_conv_block(B, Cin, Cout, H, W, K=3, runs=RUNS, quant=QUANTIZATION):
    ratios = RATIOS["conv"]

    def build_net():
        layers = []; in_ch = Cin
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

    if REBUILD_MODEL_EVERY_RUN:
        times = []
        for _ in range(runs):
            tfl = TFLiteModel(build_net(), x_np, quant)
            times.append(run_tflite(tfl, x_np, 1))
        avg = sum(times) / len(times)
        model_kb = tfl.model_kb
    else:
        tfl = TFLiteModel(build_net(), x_np, quant,
                          save_name=f"conv_block_{B}_{Cin}_{Cout}")
        avg      = run_tflite(tfl, x_np, runs)
        model_kb = tfl.model_kb

    f = flops_conv3x3(B, Cin, Cout, H, W, K) * len(ratios)
    b = bytes_conv3x3(B, Cin, Cout, H, W, K) * len(ratios)
    return avg, f, b, f/avg/1e9, model_kb


def bench_depthwise_block(B, C, H, W, K=3, runs=RUNS, quant=QUANTIZATION):
    ratios = RATIOS["dw"]

    def build_net():
        layers = []
        for _ in ratios:
            layers += [
                tf.keras.layers.DepthwiseConv2D(K, padding="same", use_bias=False),
                tf.keras.layers.ReLU(),
            ]
        return tf.keras.Sequential(layers)

    x_np = np.random.randn(B, H, W, C).astype(np.float32)

    if REBUILD_MODEL_EVERY_RUN:
        times = []; model_kb = 0.0
        for _ in range(runs):
            tfl = TFLiteModel(build_net(), x_np, quant)
            times.append(run_tflite(tfl, x_np, 1)); model_kb = tfl.model_kb
        avg = sum(times) / len(times)
    else:
        tfl = TFLiteModel(build_net(), x_np, quant,
                          save_name=f"dw_block_{B}_{C}")
        avg = run_tflite(tfl, x_np, runs); model_kb = tfl.model_kb

    f = flops_dw(B, C, H, W, K) * len(ratios)
    b = bytes_dw(B, C, H, W, K) * len(ratios)
    return avg, f, b, f/avg/1e9, model_kb


def bench_gemm_block(M, K, N, runs=RUNS, quant=QUANTIZATION):
    ratios = RATIOS["gemm"]

    def build_net():
        layers = []; in_dim = K
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

    if REBUILD_MODEL_EVERY_RUN:
        times = []; model_kb = 0.0
        for _ in range(runs):
            tfl = TFLiteModel(build_net(), x_np, quant)
            times.append(run_tflite(tfl, x_np, 1)); model_kb = tfl.model_kb
        avg = sum(times) / len(times)
    else:
        tfl = TFLiteModel(build_net(), x_np, quant,
                          save_name=f"gemm_block_{M}_{K}_{N}")
        avg = run_tflite(tfl, x_np, runs); model_kb = tfl.model_kb

    f = flops_gemm(M, K, N) * len(ratios)
    b = bytes_gemm(M, K, N) * len(ratios)
    return avg, f, b, f/avg/1e9, model_kb


def bench_attention_block(B, T, H, Dh, runs=RUNS, quant=QUANTIZATION):
    ratios = RATIOS["attn"]; D = H * Dh

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

    if REBUILD_MODEL_EVERY_RUN:
        times = []; model_kb = 0.0
        for _ in range(runs):
            tfl = TFLiteModel(build_net(), x_np, quant)
            times.append(run_tflite(tfl, x_np, 1)); model_kb = tfl.model_kb
        avg = sum(times) / len(times)
    else:
        tfl = TFLiteModel(build_net(), x_np, quant,
                          save_name=f"attn_block_{B}_{T}_{H}_{Dh}")
        avg = run_tflite(tfl, x_np, runs); model_kb = tfl.model_kb

    f = flops_attn_total(B, T, H, Dh) * len(ratios)
    b = bytes_attn_total(B, T, H, Dh) * len(ratios)
    return avg, f, b, f/avg/1e9, model_kb


def bench_ln_attn_ln_block(B, T, H, Dh, runs=RUNS, quant=QUANTIZATION):
    ratios = RATIOS["ln_attn_ln"]; D = H * Dh

    def build_net():
        layers = []
        for _ in ratios:
            layers += [LNAttnLNBlock(d_model=D, n_head=H)]
        return tf.keras.Sequential(layers)

    x_np = np.random.randn(B, T, D).astype(np.float32)

    if REBUILD_MODEL_EVERY_RUN:
        times = []; model_kb = 0.0
        for _ in range(runs):
            tfl = TFLiteModel(build_net(), x_np, quant)
            times.append(run_tflite(tfl, x_np, 1)); model_kb = tfl.model_kb
        avg = sum(times) / len(times)
    else:
        tfl = TFLiteModel(build_net(), x_np, quant,
                          save_name=f"ln_attn_ln_{B}_{T}_{H}")
        avg = run_tflite(tfl, x_np, runs); model_kb = tfl.model_kb

    f_ln1, b_ln1 = flops_bytes_layernorm(B, T, D)
    f_attn        = flops_attn_total(B, T, H, Dh)
    b_attn        = bytes_attn_total(B, T, H, Dh)
    f_ln2, b_ln2  = flops_bytes_layernorm(B, T, D)
    f = (f_ln1 + f_attn + f_ln2) * len(ratios)
    b = (b_ln1 + b_attn + b_ln2) * len(ratios)
    return avg, f, b, f/avg/1e9, model_kb

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
    print(f"[TFLite]  QUANTIZATION={QUANTIZATION}  "
          f"REBUILD={REBUILD_MODEL_EVERY_RUN}  SAVE_TFLITE={SAVE_TFLITE}")
    print(f"{'OP':<12}{'Step':<5}{'GFLOP/s':>10}{'GFLOPs':>10}"
          f"{'GBytes':>10}{'ms':>12}{'KB':>10}")
    for name, fn, key in ops:
        for step in range(N_STEPS):
            cfg = interpolate_cfg(SIZES_FIT[key], SIZES_MISS[key], step, N_STEPS)
            avg, f, b, g, kb = fn(**cfg)
            print(f"{name:<12}{step+1:<5}{g:10.2f}{f/1e9:10.3f}"
                  f"{b/1e9:10.3f}{avg*1000:12.2f}{kb:10.1f}")


if __name__ == "__main__":
    run_scaled_suite()
