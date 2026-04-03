# -*- coding: utf-8 -*-
"""
MLP(사전 학습된 PyTorch .pt) 기반 레이어별 추론시간 예측 버전
- conv / depth / pointwise / gemm: MLP(OI→time_ms@1e9ops)로 예측 후 FLOPs 비율로 스케일
- batchnorm, relu, silu, gelu, hardswish 등: 'relu'로 통합하고 ELEM_GOPS로 처리
- 출력: 레이어별 예측시간(ms)과 총합(ms)

필수: ./pt_mlp_models/ 폴더에 다음 규칙의 파일이 존재해야 함
  f"{slug('<edgeA> + <edgeB>')}__{op}.pt"
  예) "roofline_data_multi_edge8 + roofline_data_multi_edge9" → slug →
      "roofline_data_multi_edge8_roofline_data_multi_edge9__conv.pt"
디바이스 매핑:
  rpi3 → edge2+edge3, rpi4 → edge6+edge5, rpi5 → edge8+edge9
"""

import math
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


# -----------------------------
# (0) MLP 모델 로딩/예측 유틸
# -----------------------------
MODEL_DIR = torch.path = __import__("pathlib").Path("pt_mlp_models")
OI_CONST = 1_000_000_000.0  # MLP가 학습된 기준 연산량(ops)

PAIR_LABEL_BY_DEVICE: Dict[str, str] = {
    "rpi3": "roofline_data_multi_edge2 + roofline_data_multi_edge3",
    "rpi4": "roofline_data_multi_edge6 + roofline_data_multi_edge5",
    "rpi5": "roofline_data_multi_edge8 + roofline_data_multi_edge9",
}

def _slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "model"

_loaded_bundles: Dict[Tuple[str, str], Tuple[nn.Module, dict, dict]] = {}

# 저장 당시 구조(레거시): Linear, ReLU, (Dropout/Identity), Linear, ReLU, (Dropout/Identity), Linear
# → Linear 키: net.0 / net.3 / net.6
class TinyMLP_Legacy(nn.Module):
    def __init__(self, hidden=(64, 32), dropout=False):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(1, h1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0) if dropout else nn.Identity(),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0) if dropout else nn.Identity(),
            nn.Linear(h2, 1),
        )
        self.hidden = tuple(hidden)
    def forward(self, x): return self.net(x)

# 현대 구조: Linear, ReLU, Linear, ReLU, Linear
# → Linear 키: net.0 / net.2 / net.4
class TinyMLP_Modern(nn.Module):
    def __init__(self, hidden=(64, 32)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(1, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, h2),
            nn.ReLU(inplace=True),
            nn.Linear(h2, 1),
        )
        self.hidden = tuple(hidden)
    def forward(self, x): return self.net(x)

def _resolve_model_path(fname: str):
    p = MODEL_DIR / fname
    if p.exists():
        return p
    alt = __import__("pathlib").Path("/mnt/data/pt_mlp_models") / fname  # 보조 경로
    if alt.exists():
        return alt
    raise FileNotFoundError(f"MLP model file not found: {p} (also tried {alt})")

def _load_mlp_bundle(path):
    b = torch.load(path, map_location="cpu")
    hidden = tuple(b["arch"]["hidden"])
    sd = b["state_dict"]
    keys = list(sd.keys())

    has_legacy = any(k.startswith("net.6.") or k.startswith("net.3.") for k in keys)
    has_modern = any(k.startswith("net.4.") or k.startswith("net.2.") for k in keys)

    if has_legacy and not has_modern:
        mdl = TinyMLP_Legacy(hidden=hidden, dropout=False)
    elif has_modern and not has_legacy:
        mdl = TinyMLP_Modern(hidden=hidden)
    else:
        # 애매하면 레거시 먼저 시도
        try:
            mdl = TinyMLP_Legacy(hidden=hidden, dropout=False)
            mdl.load_state_dict(sd)
        except Exception:
            mdl = TinyMLP_Modern(hidden=hidden)

    mdl.load_state_dict(sd)  # 최종 로드
    mdl.eval()
    return mdl, b["stats"], b.get("meta", {})

def predict_time_const_ms(pair_label: str, op: str, oi_value: float) -> float:
    """
    사전학습된 MLP로 'OI_CONST(=1e9 ops)' 처리 시간(ms)을 예측.
    layer_time_ms ≈ time_ms_at_OI_CONST * (layer_FLOPs / OI_CONST)
    """
    key = (pair_label, op)
    if key not in _loaded_bundles:
        fname = f"{_slug(pair_label)}__{op}.pt"
        fpath = _resolve_model_path(fname)
        mdl, stats, meta = _load_mlp_bundle(fpath)
        _loaded_bundles[key] = (mdl, stats, meta)
    mdl, stats, _ = _loaded_bundles[key]

    X = np.array([[float(max(oi_value, 1e-12))]], dtype=np.float32)
    Xn = (X - stats["x_mean"]) / stats["x_std"]
    with torch.no_grad():
        pred_n = mdl(torch.from_numpy(Xn)).numpy()
    pred_ms = float(pred_n * stats["y_std"] + stats["y_mean"])  # ms@1e9ops
    return max(pred_ms, 0.0)


# -----------------------------
# (1) 디바이스별 스케일 / 엘리먼트 GOPS
# -----------------------------
Device = str  # "rpi3" | "rpi4" | "rpi5"

# (선택적) pointwise 약간 ↑, conv 살짝 ↓ 스케일이 필요하면 아래 쓰세요 (현재는 사용 안 함)
OP_THR_SCALE = {"pointwise": 1.10, "conv": 0.95, "depth": 1.0, "gemm": 1.0}

ELEM_GOPS: Dict[Device, Dict[str, float]] = {
    "rpi5": {"add":0.602333,"bn":2.042167,"hardswish":4.183833,"relu":1.089,"relu6":2.120167,"sigmoid":1.677667,"silu":1.836333,"tanh":0.5755,},
    "rpi4": {"add":0.276333,"bn":0.6995,"hardswish":1.646833,"relu":0.423333,"relu6":0.859167,"sigmoid":0.812,"silu":1.019,"tanh":0.218833,},
    "rpi3": {"add":0.16,"bn":0.47,"hardswish":0.9615,"relu":0.207833,"relu6":0.491833,"sigmoid":0.338167,"silu":0.403833,"tanh":0.1265,},
}
OPS_PER_ELEM = {"add":1.0,"bn":2.0,"relu":1.0,"relu6":2.0,"silu":6.0,"sigmoid":5.0,"tanh":5.0,"hardswish":4.0}


# -----------------------------
# (2) 유틸
# -----------------------------
def dtype_bytes(dtype: torch.dtype) -> int:
    return {
        torch.float32: 4, torch.float: 4, torch.float16: 2, torch.bfloat16: 2,
        torch.int8: 1, torch.uint8: 1, torch.int16: 2, torch.int32: 4, torch.int64: 8, torch.bool: 1,
    }.get(dtype, 4)

def classify_conv2d(m: nn.Conv2d) -> str:
    k_h, k_w = m.kernel_size
    if m.groups == m.in_channels and m.out_channels == m.in_channels:
        return f"depthwise_conv_{k_h}x{k_w}"
    if k_h == 1 and k_w == 1:
        return "pointwise_1x1"
    if m.groups > 1:
        return f"group_conv_{k_h}x{k_w}"
    return f"conv_{k_h}x{k_w}"

def canonical_class(op_class: str, mtype: str) -> str:
    if op_class.startswith("depthwise_conv"): return "depth"
    if op_class == "pointwise_1x1": return "pointwise"
    if op_class.startswith("group_conv") or op_class.startswith("conv_"): return "conv"
    if op_class == "gemm_linear": return "gemm"
    if op_class in {"batchnorm2d", "relu", "relu6", "silu", "hardswish", "gelu", "elu", "leakyrelu"}: return "relu"
    return "other"


# -----------------------------
# (3) FLOPs/Bytes 계산자
# -----------------------------
def flops_conv2d(m: nn.Conv2d, x: torch.Tensor, y: torch.Tensor) -> Tuple[int, int]:
    N = x.shape[0]; C_out = y.shape[1]; H_out, W_out = y.shape[2], y.shape[3]
    k_h, k_w = m.kernel_size; groups = m.groups; c_in_per_group = m.in_channels // groups
    macs = N * H_out * W_out * C_out * c_in_per_group * k_h * k_w
    flops = 2 * macs
    return macs, flops

def bytes_conv2d(m: nn.Conv2d, x: torch.Tensor, y: torch.Tensor) -> int:
    dt = dtype_bytes(x.dtype)
    in_b  = dt * x.numel(); out_b = dt * y.numel()
    w_elems = (m.out_channels * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1])
    w_b  = dt * w_elems
    b_b  = dt * m.out_channels if m.bias is not None else 0
    return in_b + out_b + w_b + b_b

def flops_linear(m: nn.Linear, x: torch.Tensor, y: torch.Tensor) -> Tuple[int, int]:
    in_features = m.in_features; out_features = m.out_features; Nstar = x.numel() // in_features
    macs = Nstar * in_features * out_features; flops = 2 * macs
    return macs, flops

def bytes_linear(m: nn.Linear, x: torch.Tensor, y: torch.Tensor) -> int:
    dt = dtype_bytes(x.dtype)
    in_b  = dt * x.numel(); out_b = dt * y.numel()
    w_b   = dt * (m.in_features * m.out_features)
    b_b   = dt * m.out_features if m.bias is not None else 0
    return in_b + out_b + w_b + b_b

def flops_batchnorm2d(m: nn.BatchNorm2d, x: torch.Tensor, y: torch.Tensor) -> Tuple[int, int]:
    elems = y.numel(); flops = 2 * elems
    return 0, flops

def bytes_batchnorm2d(m: nn.BatchNorm2d, x: torch.Tensor, y: torch.Tensor) -> int:
    dt = dtype_bytes(x.dtype)
    return dt * (x.numel() + y.numel())

def flops_activation(m: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Tuple[int, int]:
    elems = y.numel()
    return 0, elems

def bytes_activation(m: nn.Module, x: torch.Tensor, y: torch.Tensor) -> int:
    dt = dtype_bytes(x.dtype)
    return dt * (x.numel() + y.numel())

def flops_pool2d(m: nn.Module, x: torch.Tensor, y: torch.Tensor) -> Tuple[int, int]:
    N, C, H_in, W_in = x.shape; _, _, H_o, W_o = y.shape
    if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d)):
        kh, kw = (m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size, m.kernel_size))
    elif isinstance(m, nn.AdaptiveAvgPool2d):
        kh = math.floor(H_in / H_o) if H_o > 0 else H_in
        kw = math.floor(W_in / W_o) if W_o > 0 else W_in
    else:
        kh = kw = 1
    flops = N * C * H_o * W_o * kh * kw
    return 0, flops

def bytes_pool2d(m: nn.Module, x: torch.Tensor, y: torch.Tensor) -> int:
    dt = dtype_bytes(x.dtype)
    return dt * (x.numel() + y.numel())


# -----------------------------
# (4) 레코드 수집 (hook)
# -----------------------------
@dataclass
class Record:
    layer_idx: int
    name: str
    type: str
    op_class_raw: str
    op_class: str     # canonical: conv/pointwise/depth/gemm/relu/other
    flops: int
    bytes: int
    oi: float

def collect_layerwise_stats(model: nn.Module, input_shape=(1, 3, 224, 224)) -> List[Record]:
    model.eval()
    example = torch.zeros(input_shape)

    records: List[Record] = []
    hooks = []
    counter = {"i": 0}

    def make_hook(name: str, module: nn.Module):
        def hook(m, inp, out):
            counter["i"] += 1
            idx = counter["i"]
            try:
                x = inp[0] if isinstance(inp, (list, tuple)) else inp
                y = out[0] if isinstance(out, (list, tuple)) else out

                mtype = m.__class__.__name__
                flops = 0
                bytes_ = 0
                raw = "other"

                if isinstance(m, nn.Conv2d):
                    _, flops = flops_conv2d(m, x, y)
                    bytes_ = bytes_conv2d(m, x, y)
                    raw = classify_conv2d(m)

                elif isinstance(m, nn.Linear):
                    _, flops = flops_linear(m, x, y)
                    bytes_ = bytes_linear(m, x, y)
                    raw = "gemm_linear"

                elif isinstance(m, nn.BatchNorm2d):
                    _, flops = flops_batchnorm2d(m, x, y)
                    bytes_ = bytes_batchnorm2d(m, x, y)
                    raw = "batchnorm2d"

                elif isinstance(m, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.Hardswish, nn.LeakyReLU, nn.GELU, nn.ELU)):
                    _, flops = flops_activation(m, x, y)
                    bytes_ = bytes_activation(m, x, y)
                    raw = mtype.lower()

                elif isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                    _, flops = flops_pool2d(m, x, y)
                    bytes_ = bytes_pool2d(m, x, y)
                    raw = mtype.lower()

                oi = (flops / bytes_) if bytes_ > 0 else 0.0
                records.append(Record(
                    layer_idx=idx,
                    name=name,
                    type=mtype,
                    op_class_raw=raw,
                    op_class=canonical_class(raw, mtype),
                    flops=flops,
                    bytes=bytes_,
                    oi=oi
                ))
            except Exception as e:
                records.append(Record(
                    layer_idx=idx,
                    name=name,
                    type=m.__class__.__name__,
                    op_class_raw=f"error:{e}",
                    op_class="other",
                    flops=0, bytes=0, oi=0.0
                ))
        return hook

    TARGETS = (
        nn.Conv2d, nn.Linear,
        nn.BatchNorm2d,
        nn.ReLU, nn.ReLU6, nn.SiLU, nn.Hardswish, nn.LeakyReLU, nn.GELU, nn.ELU,
        nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
    )
    for name, module in model.named_modules():
        if not name:
            continue
        if isinstance(module, TARGETS):
            hooks.append(module.register_forward_hook(make_hook(name, module)))

    with torch.no_grad():
        _ = model(example)

    for h in hooks:
        h.remove()

    return records


# -----------------------------
# (5) MLP 기반 시간 예측
# -----------------------------
def clamp_oi(oi: float, max_oi: float = 64.0) -> float:
    return min(max(oi, 1e-12), max_oi)

def predict_time_ms_for_record(device: Device, r: Record) -> Tuple[float, float]:
    """
    반환: (pred_time_ms, effective_throughput)
      - conv/pointwise/depth/gemm: MLP로 ms@1e9ops 예측 → FLOPs 비율로 스케일
      - relu: ELEM_GOPS 기반
      - other/pool: 0
    """
    op = r.op_class
    if op in ("conv", "pointwise", "depth", "gemm"):
        pair_label = PAIR_LABEL_BY_DEVICE[device]
        base_ms = predict_time_const_ms(pair_label, op, clamp_oi(r.oi))  # ms for 1e9 ops at given OI
        t_ms = base_ms * (r.flops / OI_CONST)
        thr = (r.flops / (t_ms / 1e3)) if t_ms > 0 else 0.0  # FLOPs/s (effective)
        return t_ms, thr

    if op == "relu":
        elems = r.flops  # 1 FLOP/elem로 가정 → elems == flops
        ops = OPS_PER_ELEM["relu"] * elems
        gops = ELEM_GOPS[device]["relu"]
        thr_ops_per_s = gops * 1e9
        thr_ops_per_s = max(thr_ops_per_s, 1e3)
        t_ms = (ops / thr_ops_per_s) * 1e3
        return t_ms, thr_ops_per_s

    return 0.0, 0.0


# -----------------------------
# (6) 출력
# -----------------------------
def human_read(n: float, unit="FLOPs/s") -> str:
    a = abs(n)
    if a >= 1e12: return f"{n/1e12:.3f} T{unit}"
    if a >= 1e9:  return f"{n/1e9:.3f} G{unit}"
    if a >= 1e6:  return f"{n/1e6:.3f} M{unit}"
    if a >= 1e3:  return f"{n/1e3:.3f} K{unit}"
    return f"{n:.3f} {unit}"

def print_predicted_times(device: Device, records: List[Record]):
    recs = sorted(records, key=lambda r: r.layer_idx)
    header = f"{'idx':>4} {'layer':38} {'op':10} {'FLOPs':>12} {'Bytes':>12} {'OI':>8} {'thr_eff':>14} {'time_ms':>10}"
    print(header)
    print("-" * len(header))

    total_ms = 0.0
    for r in recs:
        t_ms, thr = predict_time_ms_for_record(device, r)
        total_ms += t_ms
        thr_unit = "FLOPs/s" if r.op_class in ("conv","pointwise","depth","gemm") else "ops/s"
        thr_str = human_read(thr, thr_unit) if thr > 0 else "-"
        print(f"{r.layer_idx:4d} {r.name:38} {r.op_class:10} "
              f"{r.flops:12,d} {r.bytes:12,d} {r.oi:8.3f} {thr_str:>14} {t_ms:10.3f}")
    print("\n[Total predicted inference time]")
    print(f"{total_ms:.3f} ms")


# -----------------------------
# (7) 모델 로더 & main
# -----------------------------
def load_model(name: str = "resnet50") -> nn.Module:
    name = name.lower()
    if name == "resnet50":        return models.resnet50(weights=None)
    if name == "squeezenet1_0":   return models.squeezenet1_0(weights=None)
    if name == "mobilenet_v2":    return models.mobilenet_v2(weights=None)
    if name == "efficientnet_b0": return models.efficientnet_b0(weights=None)
    if name == "resnet18":        return models.resnet18(weights=None)
    raise ValueError(f"Unsupported model name: {name}")

def main():
    device: Device = "rpi5"            # ← "rpi3" | "rpi4" | "rpi5"
    model_name = "resnet18"        # ← 변경 가능
    input_shape = (1, 3, 224, 224)

    model = load_model(model_name)
    records = collect_layerwise_stats(model, input_shape=input_shape)

    print(f"=== Predicted per-layer times (MLP) | model={model_name} | device={device} | input={input_shape} ===")
    print_predicted_times(device, records)

if __name__ == "__main__":
    main()
