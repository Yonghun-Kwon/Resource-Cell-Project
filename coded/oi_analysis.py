# -*- coding: utf-8 -*-
import math, time, re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Literal
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import Bottleneck, BasicBlock
from torchvision.models.mobilenetv2 import InvertedResidual

# =========================
# 설정
# =========================
Device = Literal["rpi3","rpi4","rpi5"]
TARGET_OPS = ["conv","gemm","depth","pointwise"]
MAX_OI = 64.0
DTYPE = torch.float32
DEVICE_TORCH = "cpu"   # 측정 디바이스 (CPU 기준). GPU면 sync 필요.

# 선형 회귀 계수 (throughput = a*ln(OI)+b)  ← 예측용
REG_COEFFS: Dict[Device, Dict[str, Tuple[float, float]]] = {
    "rpi3": {"conv":4.24551e+08, "gemm":3.32980e+09, "depth":3.54778e+06, "pointwise":1.35939e+08},
    "rpi4": {"conv":1.88223e+09, "gemm":4.71932e+09, "depth":2.34243e+07, "pointwise":2.98930e+08},
    "rpi5": {"conv":5.77382e+09, "gemm":1.41914e+10,"depth":1.10606e+08,"pointwise":3.04987e+09},
}
REG_INTERC: Dict[Device, Dict[str, float]] = {
    "rpi3": {"conv":1.40716e+09,"gemm":-1.57298e+09,"depth":2.36980e+08,"pointwise":1.02132e+09},;
    "rpi4": {"conv":9.21596e+08,"gemm":-2.99982e+09,"depth":5.91791e+08,"pointwise":1.56170e+09},
    "rpi5": {"conv":1.54402e+09,"gemm":-1.08294e+10,"depth":1.92669e+09,"pointwise":3.90586e+09},
}
# 미세 스케일 (선택)
OP_THR_SCALE = {"pointwise": 1.10, "conv": 0.95, "depth": 1.0, "gemm": 1.0}

# Elementwise 처리량(GOPS) (예측용)
ELEM_GOPS: Dict[Device, Dict[str, float]] = {
    "rpi5": {"add":0.602333,"bn":2.042167,"hardswish":4.183833,"relu":1.089,"relu6":2.120167,"sigmoid":1.677667,"silu":1.836333,"tanh":0.5755,},
    "rpi4": {"add":0.276333,"bn":0.6995,"hardswish":1.646833,"relu":0.423333,"relu6":0.859167,"sigmoid":0.812,"silu":1.019,"tanh":0.218833,},
    "rpi3": {"add":0.16,"bn":0.47,"hardswish":0.9615,"relu":0.207833,"relu6":0.491833,"sigmoid":0.338167,"silu":0.403833,"tanh":0.1265,},
}
OPS_PER_ELEM = {"add":1.0,"bn":2.0,"relu":1.0,"relu6":2.0,"silu":6.0,"sigmoid":5.0,"tanh":5.0,"hardswish":4.0}

# =========================
# 유틸: 분류/계산
# =========================
def clamp_oi(oi: float) -> float:
    return min(float(oi), MAX_OI)

def predict_throughput_linear(device: Device, op: str, oi: float) -> float:
    a = REG_COEFFS[device][op]; b = REG_INTERC[device][op]
    thr = max(a * math.log(max(oi, 1e-12)) + b, 1e-6)
    return thr * OP_THR_SCALE.get(op, 1.0)

def classify_conv2d(m: nn.Conv2d) -> str:
    if m.groups == m.in_channels and m.out_channels == m.in_channels: return "depth"
    if m.kernel_size == (1, 1) and m.groups == 1: return "pointwise"
    return "conv"

def flops_conv2d(m: nn.Conv2d, in_shape, out_shape) -> float:
    B, Cin, H, W = in_shape; _, Cout, Hout, Wout = out_shape; Kh, Kw = m.kernel_size
    return 2.0 * Cout * Hout * Wout * (Cin / m.groups) * Kh * Kw * B

def flops_linear(m: nn.Linear, in_shape, out_shape) -> float:
    B, in_f = in_shape; _, out_f = out_shape
    return 2.0 * B * in_f * out_f

def bytes_of_tensor_shape(shape, dtype=torch.float32) -> int:
    n = 1
    for d in shape: n *= int(d)
    return n * (torch.finfo(dtype).bits // 8)

# =========================
# 리프 모듈 타이머 (측정)
# =========================
class TimingLeafTapper:
    def __init__(self, model: nn.Module):
        self.in_shapes: Dict[int, Tuple[int, ...]] = {}
        self.out_shapes: Dict[int, Tuple[int, ...]] = {}
        self.idx_of: Dict[nn.Module, int] = {}
        self.t_sum: Dict[int, float] = {}
        self._t0: Dict[int, float] = {}
        self.handles = []
        idx = 0
        for m in model.modules():
            if m is model or any(True for _ in m.children()):  # 리프만
                continue
            self.idx_of[m] = idx
            self.t_sum[idx] = 0.0
            self.handles.append(m.register_forward_pre_hook(self._pre(idx)))
            self.handles.append(m.register_forward_hook(self._post(idx)))
            idx += 1
    def _pre(self, i: int):
        def fn(module, inp):
            self._t0[i] = time.perf_counter()
            # 입력 shape 기록
            def to_shape(x):
                if isinstance(x, (list, tuple)):
                    for t in x:
                        if isinstance(t, torch.Tensor): return tuple(t.shape)
                    return None
                if isinstance(x, torch.Tensor): return tuple(x.shape)
                return None
            self.in_shapes[i] = to_shape(inp)
        return fn
    def _post(self, i: int):
        def fn(module, inp, out):
            t1 = time.perf_counter()
            self.t_sum[i] += (t1 - self._t0.get(i, t1))
            if isinstance(out, torch.Tensor): self.out_shapes[i] = tuple(out.shape)
            elif isinstance(out, (list, tuple)):
                for t in out:
                    if isinstance(t, torch.Tensor):
                        self.out_shapes[i] = tuple(t.shape); break
        return fn
    def remove(self):
        for h in self.handles: h.remove()

# 잔차 Add 카운터 (모양 수집)
class ResidualAddCollector:
    def __init__(self, model: nn.Module):
        self.out_shapes: List[Tuple[int, ...]] = []
        self.handles = []
        def hook_fn(module, inp, out):
            if isinstance(out, torch.Tensor):
                self.out_shapes.append(tuple(out.shape))
        for m in model.modules():
            if isinstance(m, (Bottleneck, BasicBlock, InvertedResidual)):
                # InvertedResidual은 use_res_connect일 때만 add 발생
                if isinstance(m, InvertedResidual) and not getattr(m, "use_res_connect", False):
                    continue
                self.handles.append(m.register_forward_hook(hook_fn))
    def remove(self):
        for h in self.handles: h.remove()

# 특정 shape에 대한 add 마이크로벤치마크 (ms/1회)
@torch.no_grad()
def bench_add_ms(shape: Tuple[int, ...], repeat: int = 30, device="cpu", dtype=torch.float32) -> float:
    a = torch.randn(*shape, device=device, dtype=float if dtype is float else dtype)
    b = torch.randn(*shape, device=device, dtype=a.dtype)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = a + b
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)
    times.sort()
    return float(np.median(times))

# =========================
# 예측에 필요한 per-op FLOPs & 대표 OI 집계
# =========================
@dataclass
class OpAgg:
    flops: float
    oi: float

def analyze_ops_for_prediction(model: nn.Module, input_shape=(1,3,224,224),
                               device="cpu", dtype=torch.float32):
    model = model.to(device=device).eval()
    x = torch.randn(*input_shape, device=device, dtype=dtype)
    tap = TimingLeafTapper(model)  # shape 수집 겸용
    with torch.no_grad(): _ = model(x)
    tap.remove()

    flops_sum = {k: 0.0 for k in TARGET_OPS}
    ln_oi_times_flops = {k: 0.0 for k in TARGET_OPS}
    elem_ops = {k: 0.0 for k in OPS_PER_ELEM.keys()}  # relu/bn/...

    for m, idx in tap.idx_of.items():
        in_shape, out_shape = tap.in_shapes.get(idx), tap.out_shapes.get(idx)
        if in_shape is None or out_shape is None: continue

        if isinstance(m, nn.Conv2d):
            op = classify_conv2d(m)
            f  = flops_conv2d(m, in_shape, out_shape)
            in_b = bytes_of_tensor_shape(in_shape, dtype)
            out_b = bytes_of_tensor_shape(out_shape, dtype)
            oi = max(f / float(in_b + out_b), 1e-12)
            flops_sum[op] += f
            ln_oi_times_flops[op] += math.log(oi) * f
            continue

        if isinstance(m, nn.Linear):
            f  = flops_linear(m, in_shape, out_shape)
            in_b = bytes_of_tensor_shape(in_shape, dtype)
            out_b = bytes_of_tensor_shape(out_shape, dtype)
            oi = max(f / float(in_b + out_b), 1e-12)
            flops_sum["gemm"] += f
            ln_oi_times_flops["gemm"] += math.log(oi) * f
            continue

        # elementwise ops (측정은 별도, 예측을 위해 ops count)
        if isinstance(m, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.Sigmoid, nn.Tanh, nn.Hardswish)):
            numel = math.prod(out_shape)
            key = m.__class__.__name__.lower()
            elem_ops[key] += OPS_PER_ELEM[key] * numel
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            numel = math.prod(out_shape)
            elem_ops["bn"] += OPS_PER_ELEM["bn"] * numel

    # 대표 OI
    opagg = {}
    for op in TARGET_OPS:
        F = flops_sum[op]
        if F > 0:
            opagg[op] = OpAgg(flops=F, oi=math.exp(ln_oi_times_flops[op] / F))
    return opagg, elem_ops

# =========================
# 예측 시간 (카테고리별)
# =========================
def predict_per_category_times(device_key: Device, model: nn.Module,
                               input_shape=(1,3,224,224), dtype=torch.float32) -> Dict[str,float]:
    opagg, elem_ops = analyze_ops_for_prediction(model, input_shape, device=DEVICE_TORCH, dtype=dtype)

    # conv/gemm/depth/pointwise
    res = {k:0.0 for k in ["conv","gemm","depth","pointwise","elementwise","total"]}
    for op, agg in opagg.items():
        thr = predict_throughput_linear(device_key, op, clamp_oi(agg.oi))
        res[op] = agg.flops / max(thr, 1e-6) * 1000.0  # ms

    # elementwise (relu/bn/..., add는 아래에서 추가)
    gops_tbl = ELEM_GOPS[device_key]
    for k, ops in elem_ops.items():
        gops = gops_tbl.get(k, 0.1)
        res["elementwise"] += (ops / (gops * 1e9)) * 1000.0

    res["total"] = sum(res[k] for k in ["conv","gemm","depth","pointwise","elementwise"])
    return res

# =========================
# 실제 측정 시간 (카테고리별)
#  - 리프 모듈 시간 누적 + Add는 마이크로벤치로 추정
# =========================
@torch.no_grad()
def measure_per_category_times(model: nn.Module, input_shape=(1,3,224,224),
                               warmup=5, repeat=10, device="cpu", dtype=torch.float32) -> Dict[str,float]:
    model = model.to(device=device, dtype=dtype).eval()
    x = torch.randn(*input_shape, device=device, dtype=dtype)

    timer = TimingLeafTapper(model)
    addc  = ResidualAddCollector(model)

    # 워밍업
    for _ in range(warmup):
        _ = model(x)

    # 반복 측정
    for _ in range(repeat):
        t0 = time.perf_counter()
        _ = model(x)
        if device != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
        _ = time.perf_counter() - t0  # 총시간 필요하면 활용 가능

    timer.remove(); addc.remove()

    # 카테고리 합산
    ms = {"conv":0.0,"gemm":0.0,"depth":0.0,"pointwise":0.0,"elementwise":0.0,"total":0.0}
    for m, idx in timer.idx_of.items():
        dt_ms = (timer.t_sum[idx] / max(repeat,1)) * 1000.0
        in_shape, out_shape = timer.in_shapes.get(idx), timer.out_shapes.get(idx)
        if isinstance(m, nn.Conv2d):
            kind = classify_conv2d(m)
            ms[kind] += dt_ms
        elif isinstance(m, nn.Linear):
            ms["gemm"] += dt_ms
        elif isinstance(m, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.Sigmoid, nn.Tanh, nn.Hardswish,
                            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            ms["elementwise"] += dt_ms
        else:
            # 기타 리프 모듈은 elementwise에 포함 (필요시 분리)
            ms["elementwise"] += dt_ms

    # 잔차 Add 시간(추정): 블록 출력 shape별 add 마이크로벤치 × 개수
    # (중복 shape는 캐시하여 벤치 최소화)
    cache: Dict[Tuple[int,...], float] = {}
    add_ms_total = 0.0
    for shp in addc.out_shapes:
        if shp not in cache:
            cache[shp] = bench_add_ms(shp, repeat=30, device=device, dtype=dtype)
        add_ms_total += cache[shp]
    ms["elementwise"] += add_ms_total

    ms["total"] = ms["conv"] + ms["gemm"] + ms["depth"] + ms["pointwise"] + ms["elementwise"]
    return {k: float(ms[k]) for k in ms}

# =========================
# 비교 실행
# =========================
def compare_model(model_name: str, model_obj: nn.Module, device_key: Device,
                  input_shape=(1,3,224,224), warmup=5, repeat=10, dtype=torch.float32):
    pred = predict_per_category_times(device_key, model_obj, input_shape, dtype)
    meas = measure_per_category_times(model_obj, input_shape, warmup, repeat, device=DEVICE_TORCH, dtype=dtype)

    rows = []
    for k in ["conv","gemm","depth","pointwise","elementwise","total"]:
        rows.append({
            "model": model_name,
            "device": device_key.upper(),
            "category": k,
            "pred_ms": round(pred[k], 3),
            "meas_ms": round(meas[k], 3),
            "abs_err_ms": round(abs(pred[k]-meas[k]), 3),
            "rel_err_%": round((abs(pred[k]-meas[k]) / max(meas[k], 1e-6)) * 100.0, 2)
        })
    df = pd.DataFrame(rows)
    print(f"\n=== {model_name} | {device_key.upper()} | per-category: predicted vs measured (ms) ===")
    print(df.to_string(index=False))
    return df

# =========================
# main 예시
# =========================
if __name__ == "__main__":
    torch.set_grad_enabled(False)

    models_to_test = {
        "MobileNetV2":      models.mobilenet_v2(weights=None),
        "ResNet18":         models.resnet18(weights=None),
        "ResNet50":         models.resnet50(weights=None),
        "SqueezeNet1_0":    models.squeezenet1_0(weights=None),
        "EfficientNet_B0":  models.efficientnet_b0(weights=None),
    }

    DEVICE_KEY: Device = "rpi5"  # "rpi3" | "rpi4" | "rpi5"
    INPUT_SHAPE = (1,3,224,224)

    all_rows = []
    for name, mdl in models_to_test.items():
        df = compare_model(name, mdl, DEVICE_KEY, input_shape=INPUT_SHAPE, warmup=5, repeat=10, dtype=DTYPE)
        all_rows.append(df)
    all_df = pd.concat(all_rows, ignore_index=True)
    # CSV로 저장하려면:
    # all_df.to_csv("per_category_pred_vs_meas.csv", index=False)

