#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision.models import swin_v2_t, Swin_V2_T_Weights
import math
import time
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import torchvision.models as models

# ========================== 사용자 스위치 ==========================
USE_MINI_LM = False   # ← True: 초경량 트랜스포머 LM 사용, False: ResNet-50
SEQ_LEN = 128
DEVICE = "cpu"
NUM_RUNS = 10
# ================================================================

ATTN_FLOP_MODE = "full"          # "full" | "projections" | "core"
ATTN_INCLUDE_SOFTMAX = True
ATTN_INCLUDE_LAYERNORM = False
ATTN_FLOP_SCALE = 1.0

# =========================================================
# (Mini) 초경량 트랜스포머 언어모델
# =========================================================
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model=128, n_head=2, attn_dropout=0.0, resid_dropout=0.0):
        super().__init__()
        assert d_model % n_head == 0
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.resid_drop = nn.Dropout(resid_dropout)

    def forward(self, x):
        B, T, C = x.shape
        H, Dh = self.n_head, self.head_dim
        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2).contiguous()
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2).contiguous()
        att = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.o_proj(y))

class TransformerBlock(nn.Module):
    def __init__(self, d_model=128, n_head=2, mlp_ratio=4, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model=d_model, n_head=n_head,
                                           attn_dropout=dropout, resid_dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        hidden = d_model * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class MiniTransformerLM(nn.Module):
    def __init__(self, vocab_size=320, seq_len=128, d_model=128,
                 n_head=2, n_layer=2, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, seq_len, d_model))
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model=d_model, n_head=n_head, mlp_ratio=4, dropout=dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, input_ids):
        B, T = input_ids.shape
        assert T <= self.seq_len, "seq_len 초과"
        x = self.tok_emb(input_ids) + self.pos_emb[:, :T, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

# =========================================================
# (공통) Attention 감지 / 파라미터 추출 / FLOPs·Bytes
# =========================================================
def _looks_like_attention(m: nn.Module) -> bool:
    n = m.__class__.__name__.lower()
    if "attention" in n or "attn" in n:
        return True
    if isinstance(m, nn.MultiheadAttention):
        return True
    has_heads = any(hasattr(m, attr) for attr in (
        "num_heads", "n_heads", "num_attention_heads", "heads", "head_dim"
    ))
    has_qkv = hasattr(m, "qkv") or hasattr(m, "q_proj") or hasattr(m, "k_proj") or hasattr(m, "v_proj")
    has_proj_out = hasattr(m, "proj") or hasattr(m, "o_proj") or hasattr(m, "out_proj")
    if has_heads and (has_qkv or has_proj_out or "window" in n):
        return True
    if hasattr(m, "window_size"):
        return True
    return False


def _extract_attn_params(m: nn.Module, xin: Optional[torch.Tensor], yout: Optional[torch.Tensor]):
    if xin is None:
        return None
    a, b, c = xin.shape[-3], xin.shape[-2], xin.shape[-1]
    batch_first = getattr(m, "batch_first", False)
    if isinstance(m, nn.MultiheadAttention) and not batch_first:
        T, B, D = a, b, c
    else:
        if a <= b:
            B, T, D = a, b, c
        else:
            T, B, D = a, b, c

    H = (getattr(m, "num_heads", None) or
         getattr(m, "n_heads", None) or
         getattr(m, "num_attention_heads", None) or
         getattr(m, "heads", None) or 1)
    H = int(H)
    Dh = max(1, D // H)

    span = T
    win = getattr(m, "window_size", None)
    if isinstance(win, (tuple, list)) and len(win) >= 2:
        span = int(win[0]) * int(win[1])
    elif isinstance(win, int) and win > 0:
        span = min(T, int(win) * int(win))

    return int(B), int(T), int(D), int(H), int(Dh), int(span)


def _attn_flops_bytes(B: int, T: int, H: int, Dh: int, span: int):
    D = H * Dh
    proj = 8.0 * B * T * D * D
    core = 4.0 * B * H * T * span * Dh

    if ATTN_FLOP_MODE == "full":
        f = proj + core
    elif ATTN_FLOP_MODE == "projections":
        f = proj
    elif ATTN_FLOP_MODE == "core":
        f = core
    else:
        f = proj + core

    if ATTN_INCLUDE_SOFTMAX:
        f += 3.0 * B * H * T * span
    if ATTN_INCLUDE_LAYERNORM:
        f += 5.0 * B * T * D
    f *= ATTN_FLOP_SCALE

    xio   = (B*T*D + B*T*D)
    qkvo  = 4 * B * T * D
    scores = 2 * B * H * T * span
    ybytes = B * T * D
    b = (xio + qkvo + scores + ybytes) * 4
    return f, b

# =========================================================
# (A) pointwise/pool → conv, embedding → gemm 병합
# =========================================================
def merge_pointwise_into_conv_stats(cat_flops: dict, cat_bytes: dict):
    f = dict(cat_flops); b = dict(cat_bytes)
    for extra in ("pointwise", "pool"):
        f_extra = f.pop(extra, 0.0)
        b_extra = b.pop(extra, 0)
        f["conv"] = f.get("conv", 0.0) + f_extra
        b["conv"] = b.get("conv", 0)   + b_extra
    emb_f = f.pop("embedding", 0.0); emb_b = b.pop("embedding", 0)
    f["gemm"] = f.get("gemm", 0.0) + emb_f
    b["gemm"] = b.get("gemm", 0)   + emb_b
    return f, b

# =========================================================
# (1) 1회 프로파일 + N회 타이밍 측정
# =========================================================
def analyze_model_operations(model, input_tensor, num_runs=NUM_RUNS):
    def classify(m: nn.Module) -> str:
        if isinstance(m, nn.Conv2d):
            g = m.groups; kh, kw = m.kernel_size
            if g == m.in_channels == m.out_channels: return "depthwise"
            if (kh, kw) == (1, 1) and g == 1:        return "pointwise"
            return "conv"
        if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            return "pool"
        if isinstance(m, nn.Embedding):
            return "embedding"
        if _looks_like_attention(m):
            return "attention"
        if isinstance(m, nn.Linear):
            return "gemm"
        return "nonlinear"

    class LayerAwareTimingHook:
        def __init__(self):
            self.start = {}
            self.cat_time_s = defaultdict(float)
            self.handles = []
            self.current_major = None

        def register(self, model: nn.Module):
            for m in model.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear,
                                  nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU,
                                  nn.BatchNorm2d, nn.LayerNorm,
                                  nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
                                  nn.MultiheadAttention)) or _looks_like_attention(m):
                    self.handles.append(m.register_forward_pre_hook(self.pre))
                    self.handles.append(m.register_forward_hook(self.post))

        def remove(self):
            for h in self.handles: h.remove()
            self.handles.clear()
            self.start.clear()

        def classify_major(self, m):
            if isinstance(m, nn.Conv2d):
                return "depthwise" if m.groups == m.in_channels == m.out_channels else "conv"
            if isinstance(m, nn.Linear):
                return "gemm"
            if _looks_like_attention(m):
                return "attention"
            return None

        def classify_minor(self, m):
            if isinstance(m, (nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU,
                               nn.BatchNorm2d, nn.LayerNorm)):
                return "nonlinear"
            if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                return "pool"
            return None

        def pre(self, m, x):
            self.start[id(m)] = time.perf_counter()
            if self.classify_major(m) is not None:
                self.current_major = self.classify_major(m)

        def post(self, m, x, y):
            t0 = self.start.pop(id(m), None)
            if t0 is None:
                return
            dt = time.perf_counter() - t0
            major = self.classify_major(m)
            minor = self.classify_minor(m)
            if major is not None:
                self.cat_time_s[major] += dt
                self.current_major = major
                return
            if minor == "nonlinear":
                self.cat_time_s[self.current_major if self.current_major in ("conv", "depthwise") else "nonlinear"] += dt
                return
            if minor == "pool":
                self.cat_time_s[self.current_major if self.current_major in ("conv", "depthwise") else "conv"] += dt
                return
            self.cat_time_s["nonlinear"] += dt

        def results(self):
            return {op: {"time_ms": t * 1000.0} for op, t in self.cat_time_s.items()}

    model.eval()
    hook = LayerAwareTimingHook()
    hook.register(model)
    with profile(activities=[ProfilerActivity.CPU],
                 with_flops=True, record_shapes=False, profile_memory=False) as prof:
        with torch.no_grad():
            with record_function("profile_run"): _ = model(input_tensor)
    per_op = hook.results()
    hook.remove()

    total_flops = sum(evt.flops for evt in prof.key_averages() if getattr(evt, "flops", 0))

    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.time(); _ = model(input_tensor); t1 = time.time()
            times.append((t1 - t0) * 1000)
    avg_ms = sum(times) / len(times)

    print(f"\n== Inference Report (Profile 1회, Timing {num_runs}회) ==")
    print(f"Avg Time per Inference : {avg_ms:.2f} ms")
    print(f"Total FLOPs (profile)  : {total_flops / 1e9:.3f} GFLOPs")
    return total_flops, avg_ms, per_op

# =========================================================
# (2) FLOPs / Bytes 집계 훅
# =========================================================
class OpCategorizerHook:
    def __init__(self):
        self.cat_flops = defaultdict(float)
        self.cat_bytes = defaultdict(int)
        self.handles = []

    def register(self, model: nn.Module):
        for m in model.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear, nn.Embedding,
                               nn.ReLU, nn.ReLU6, nn.GELU, nn.SiLU,
                               nn.BatchNorm2d, nn.LeakyReLU,
                               nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
                               nn.MultiheadAttention)) or _looks_like_attention(m):
                self.handles.append(m.register_forward_hook(self.hook))

    def remove(self):
        for h in self.handles: h.remove()
        self.handles.clear()

    @staticmethod
    def bytes_of(t): return t.element_size() * t.numel()

    def classify(self, m):
        if isinstance(m, nn.Conv2d):
            g = m.groups; kh, kw = m.kernel_size
            if g == m.in_channels: return "depthwise"
            if (kh, kw) == (1, 1): return "pointwise"
            return "conv"
        if isinstance(m, (nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
            return "pool"
        if isinstance(m, nn.Embedding):
            return "embedding"
        if _looks_like_attention(m):
            return "attention"
        if isinstance(m, nn.Linear):
            return "gemm"
        return "nonlinear"

    def flops_of(self, m, x, y):
        xin = next((t for t in x if torch.is_tensor(t)), None)
        yout = y[0] if isinstance(y, (tuple, list)) else y
        if _looks_like_attention(m):
            params = _extract_attn_params(m, xin, yout)
            if params is None: return 0.0
            B, T, D, H, Dh, span = params
            f, _ = _attn_flops_bytes(B, T, H, Dh, span)
            return float(f)
        if not (torch.is_tensor(xin) and torch.is_tensor(yout)): return 0.0
        if isinstance(m, nn.Linear):
            M = xin.numel() // m.in_features
            fl = 2.0 * M * m.in_features * m.out_features
            if m.bias is not None: fl += M * m.out_features
            return fl
        if isinstance(m, nn.Conv2d):
            N, C, Ht, W = xin.shape; Cout, Hout, Wout = yout.shape[1:]
            Kh, Kw = m.kernel_size; g = m.groups
            return 2.0 * N * Cout * Hout * Wout * (C // g) * Kh * Kw
        return 0.0

    def hook(self, m, x, y):
        cat = self.classify(m); f = self.flops_of(m, x, y); b = 0
        for t in x:
            if torch.is_tensor(t): b += self.bytes_of(t)
        if torch.is_tensor(y): b += self.bytes_of(y)
        elif isinstance(y, (tuple, list)):
            b += sum(self.bytes_of(t) for t in y if torch.is_tensor(t))
        self.cat_flops[cat] += f
        self.cat_bytes[cat] += b

    def totals(self): return dict(self.cat_flops), dict(self.cat_bytes)

# =========================================================
# (3) OI 기반 레이턴시 예측 (선형 회귀)
# =========================================================
REG_THROUGHPUT_OI_LINEAR = {
    "rpi3": {
        "attention": {"a": -6.4330,   "b": 0.2440,  "R2": 0.4916, "n": 150},
        "conv":      {"a":  0.0552,   "b": 0.0487,  "R2": 0.9151, "n": 150},
        "depthwise": {"a": 0.0,       "b": 0.0870,  "R2": 0.002,  "n": 150},
        "gemm":      {"a": 11.5501,   "b": -0.0196, "R2": 0.1015, "n": 150},
    },
    "rpi4": {
        "attention": {"a": -13.0094,  "b": 0.5014,  "R2": 0.5278, "n": 150},
        "conv":      {"a": 0.5226,    "b": 0.0977,  "R2": 0.9528, "n": 150},
        "depthwise": {"a": 0.0000,    "b": 0.2041,  "R2": 0.0022, "n": 150},
        "gemm":      {"a": 13.2254,   "b": 0.1030,  "R2": 0.4939, "n": 150},
    },
    "rpi5": {
        "attention": {"a": -5.2551,   "b": 0.8862,  "R2": 0.2204, "n": 150},
        "conv":      {"a": 3.1018,    "b": 0.2145,  "R2": 0.7758, "n": 150},
        "depthwise": {"a": 0,         "b": 0.7547,  "R2": 0.0005, "n": 150},
        "gemm":      {"a": 58.5960,   "b": 0.1194,  "R2": 0.1144, "n": 150},
    }
}

GLOBAL_OI_CAP = 50.0
OI_CAPS = {"gemm": 80.0, "conv": 40.0, "depthwise": 3.0, "nonlinear": 50.0, "attention": 50.0}
STATIC_THR_FLOOR_GLOBAL = 0.20
STATIC_THR_FLOOR_BY_OP  = {"nonlinear": 0.50, "depthwise": 0.2, "conv": 0.20, "gemm": 1.00, "attention": 1.00}
STATIC_THR_CEIL_BY_OP   = {}
OP_ALIAS_FOR_REG = {
    "embedding": "gemm",
    "pool":      "conv",
}

def _cap_oi_static(oi, op, use_opwise_cap=True):
    cap = OI_CAPS.get(op, GLOBAL_OI_CAP) if use_opwise_cap else GLOBAL_OI_CAP
    if not math.isfinite(oi): return 0.0
    return max(0.0, min(oi, cap))

def _apply_thr_bounds_static(thr, op):
    floor = STATIC_THR_FLOOR_BY_OP.get(op, STATIC_THR_FLOOR_GLOBAL)
    ceil  = STATIC_THR_CEIL_BY_OP.get(op, None)
    if floor is not None: thr = max(thr, floor)
    if ceil  is not None: thr = min(thr, ceil)
    return thr

def estimate_latency_regression_linear(device_key, cat_flops, cat_bytes, use_opwise_cap=True):
    coeffs = REG_THROUGHPUT_OI_LINEAR.get(device_key, {})
    per = {}; total_ms = 0.0
    for op, flops in cat_flops.items():
        base_op = OP_ALIAS_FOR_REG.get(op, op)
        if base_op == "pointwise": continue
        bytes_ = cat_bytes.get(op, 0)
        if flops <= 0 or bytes_ <= 0: continue

        oi_raw = flops / bytes_
        oi_cap = _cap_oi_static(oi_raw, base_op, use_opwise_cap)
        gflops = flops / 1e9

        reg = coeffs.get(base_op, {"a": 0.0, "b": 0.0})
        thr = reg["a"] + reg["b"] * oi_cap
        thr = _apply_thr_bounds_static(thr, base_op)
        if not math.isfinite(thr) or thr <= 0:
            thr = _apply_thr_bounds_static(1e-3, base_op)

        t_ms = (gflops / thr) * 1e3
        per[op] = {
            "OI_raw": oi_raw, "OI_capped": oi_cap, "GFLOPs": gflops,
            "Throughput(GF/s)": thr, "Latency(ms)": t_ms,
            "R2": reg.get("R2", 0.0), "Bytes(MB)": bytes_ / 1e6,
        }
        total_ms += t_ms
    return per, total_ms

def pretty_print_estimate_linear(device_key, per, total_ms):
    print(f"\n== OI-based Latency Estimate (Linear + Static caps) | Device: {device_key} ==")
    print(f"{'Op':<12}{'OI(raw)':>12}{'OI(cap)':>10}{'GFLOPs':>10}{'GF/s':>12}{'R2':>8}{'Latency(ms)':>15}")
    print("-" * 90)
    for op, d in per.items():
        print(f"{op:<12}{d['OI_raw']:12.3f}{d['OI_capped']:10.3f}{d['GFLOPs']:10.3f}"
              f"{d['Throughput(GF/s)']:12.3f}{d['R2']:8.3f}{d['Latency(ms)']:15.3f}")
    print("-" * 90)
    print(f"{'TOTAL':<12}{'':>12}{'':>10}{'':>10}{'':>12}{'':>8}{total_ms:15.3f}")

# =========================================================
# (4) 전체 OI 측정 (MemTracker)
# =========================================================
class MemTracker:
    def __init__(self):
        self.total_read = 0; self.total_write = 0; self.handles = []

    def register_hooks(self, model):
        for m in model.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.Embedding, nn.ReLU,
                               nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d,
                               nn.MultiheadAttention)) or _looks_like_attention(m):
                self.handles.append(m.register_forward_hook(self.hook))

    def hook(self, m, x, y):
        in_b  = sum(t.element_size() * t.numel() for t in x if torch.is_tensor(t))
        out_b = y.element_size() * y.numel() if torch.is_tensor(y) else 0
        self.total_read += in_b; self.total_write += out_b

    def remove(self):
        for h in self.handles: h.remove()
        self.handles.clear()

    def total_bytes(self): return self.total_read + self.total_write

def estimate_oi_by_hook(model, x, total_flops):
    tr = MemTracker(); tr.register_hooks(model)
    with torch.no_grad(): model(x)
    bytes_ = tr.total_bytes(); tr.remove()
    oi = total_flops / bytes_ if bytes_ > 0 else 0
    print(f"\n== OI (overall) ==")
    print(f"Total Memory: {bytes_ / 1e6:.3f} MB")
    print(f"Total FLOPs : {total_flops / 1e9:.3f} GFLOPs")
    print(f"OI          : {oi:.3f} FLOPs/Byte")

# =========================================================
# (5) 실측 레이턴시 분해
# =========================================================
def compute_real_op_latency(measured_ms, per_op_meas):
    op_hook_times = {op: d["time_ms"] for op, d in per_op_meas.items() if d.get("time_ms", 0.0) > 0}
    total_hook_time = sum(op_hook_times.values())
    if total_hook_time <= 0:
        return {}
    return {op: measured_ms * (t / total_hook_time) for op, t in op_hook_times.items()}

def pretty_print_real_op_latency(real_op_times, measured_ms):
    print("\n== Real Inference Time Breakdown (per-operator) ==")
    print(f"{'Op':<12}{'Latency(ms)':>15}{'Ratio(%)':>12}")
    print("-" * 45)
    total = 0.0
    for op, t in sorted(real_op_times.items(), key=lambda x: -x[1]):
        ratio = (t / measured_ms) * 100.0 if measured_ms > 0 else 0.0
        print(f"{op:<12}{t:15.3f}{ratio:12.2f}")
        total += t
    print("-" * 45)
    print(f"{'TOTAL':<12}{total:15.3f}{100.0:12.2f}")

# =========================================================
# MAIN
# =========================================================
USE_VISION_TRANSFORMER = False
USE_SWINT = False

if __name__ == "__main__":
    if USE_MINI_LM:
        model = MiniTransformerLM(
            vocab_size=320, seq_len=SEQ_LEN, d_model=128,
            n_head=2, n_layer=2, dropout=0.0
        ).to(DEVICE).eval()
        x = torch.randint(0, 320, (1, SEQ_LEN), device=DEVICE)
    elif USE_VISION_TRANSFORMER:
        if USE_SWINT:
            model = swin_v2_t(weights=Swin_V2_T_Weights.IMAGENET1K_V1).to(DEVICE).eval()
        else:
            model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(DEVICE).eval()
        x = torch.randn(1, 3, 224, 224, device=DEVICE)
    else:
        model = models.efficientnet_b0().to(DEVICE).eval()
        x = torch.randn(1, 3, 224, 224, device=DEVICE)

    total_flops, measured_ms, per_op_meas = analyze_model_operations(model, x, num_runs=NUM_RUNS)

    hook = OpCategorizerHook(); hook.register(model)
    with torch.no_grad(): model(x)
    cat_flops, cat_bytes = hook.totals(); hook.remove()
    merged_flops, merged_bytes = merge_pointwise_into_conv_stats(cat_flops, cat_bytes)

    for dev in ["rpi3", "rpi4", "rpi5"]:
        per, total_ms = estimate_latency_regression_linear(dev, merged_flops, merged_bytes)
        pretty_print_estimate_linear(dev, per, total_ms)

    estimate_oi_by_hook(model, x, total_flops)
    print(f"\n[Measured on this host] Avg of {NUM_RUNS} runs: {measured_ms:.3f} ms")

    real_op_times = compute_real_op_latency(measured_ms, per_op_meas)
    pretty_print_real_op_latency(real_op_times, measured_ms)