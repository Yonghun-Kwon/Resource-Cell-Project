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


ATTN_FLOP_MODE = "full"     # "full" | "projections" | "core"  (full 권장)
ATTN_INCLUDE_SOFTMAX = True # softmax FLOPs(작음) 포함 여부
ATTN_INCLUDE_LAYERNORM = False  # LayerNorm FLOPs 포함 여부(선택)
ATTN_FLOP_SCALE = 1.0       # 마지막 보정 스케일(필요시 0.8~1.2 등)

# =========================================================
# (0) 예시: GEMM 위주의 MLP 모델 (옵션)
# =========================================================
class MLPModel(nn.Module):
    def __init__(self, in_dim=1024, hidden=4096, out_dim=1000, depth=4):
        super().__init__()
        layers = []
        for i in range(depth):
            layers += [nn.Linear(in_dim if i==0 else hidden, hidden),
                       nn.ReLU(inplace=True)]
        layers.append(nn.Linear(hidden, out_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

# =========================================================
# (Mini) 초경량 트랜스포머 언어모델 (PyTorch only)
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
        q = self.q_proj(x).view(B, T, H, Dh).transpose(1, 2).contiguous()   # B,H,T,Dh
        k = self.k_proj(x).view(B, T, H, Dh).transpose(1, 2).contiguous()
        v = self.v_proj(x).view(B, T, H, Dh).transpose(1, 2).contiguous()

        att = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)        # B,H,T,T
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v                                            # B,H,T,Dh
        y = y.transpose(1, 2).contiguous().view(B, T, C)       # B,T,C
        y = self.o_proj(y)
        y = self.resid_drop(y)
        return y

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
    def __init__(self,
                 vocab_size=320,
                 seq_len=128,
                 d_model=128,
                 n_head=2,
                 n_layer=2,
                 dropout=0.0):
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
        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, V)
        return logits

# =========================================================
# (A) pointwise / pool → conv, embedding → gemm 통합 유틸
# =========================================================
def merge_pointwise_into_conv_perop(per_op: dict) -> dict:
    merged = {k: v.copy() for k, v in per_op.items()}
    for extra in ("pointwise", "pool"):
        e = merged.pop(extra, None)
        if e is None:
            continue
        if "conv" not in merged:
            merged["conv"] = {"flops": 0.0, "time_ms": 0.0, "throughput": 0.0}
        merged["conv"]["flops"] += e.get("flops", 0.0)
        merged["conv"]["time_ms"] += e.get("time_ms", 0.0)
    if "conv" in merged:
        time_s = merged["conv"]["time_ms"] / 1000.0
        gflops_total = merged["conv"]["flops"]
        merged["conv"]["throughput"] = (gflops_total / time_s) if time_s > 0 else 0.0
    return merged

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
# (공통) Attention 감지/파라미터
# =========================================================
# =========================================================
# (공통) Attention 감지/파라미터/수식 (개선)
# =========================================================
def _looks_like_attention(m: nn.Module) -> bool:
    n = m.__class__.__name__.lower()
    if "attention" in n or "attn" in n:
        return True
    if isinstance(m, nn.MultiheadAttention):
        return True
    # torchvision ViT/Swin 내부 블록들 대비
    has_heads = any(hasattr(m, attr) for attr in (
        "num_heads", "n_heads", "num_attention_heads", "heads", "head_dim"
    ))
    has_qkv = hasattr(m, "qkv") or hasattr(m, "q_proj") or hasattr(m, "k_proj") or hasattr(m, "v_proj")
    has_proj_out = hasattr(m, "proj") or hasattr(m, "o_proj") or hasattr(m, "out_proj")
    if has_heads and (has_qkv or has_proj_out or "window" in n):
        return True
    # Swin window attention 흔적
    if hasattr(m, "window_size"):
        return True
    return False


def _extract_attn_params(m: nn.Module,
                         xin: Optional[torch.Tensor],
                         yout: Optional[torch.Tensor]):
    """
    반환: (B, T, D, H, Dh, span)
    - span: full self-attention이면 T, Swin같이 윈도우면 window_size^2
    """
    if xin is None:
        return None
    # xin 모양을 (B,T,D)로 해석 시도
    # MultiheadAttention은 (T,B,D)도 가능
    a, b, c = xin.shape[-3], xin.shape[-2], xin.shape[-1]
    batch_first = getattr(m, "batch_first", False)
    if isinstance(m, nn.MultiheadAttention) and not batch_first:
        T, B, D = a, b, c
    else:
        # 대부분 (B,T,D) 혹은 (T,B,D)이지만, B가 더 작다는 가정이 안전한 편
        if a <= b:   # (B,T,D)
            B, T, D = a, b, c
        else:        # (T,B,D)
            T, B, D = a, b, c

    # heads
    H = (getattr(m, "num_heads", None) or
         getattr(m, "n_heads", None) or
         getattr(m, "num_attention_heads", None) or
         getattr(m, "heads", None) or 1)
    H = int(H)
    Dh = max(1, D // H)

    # span 결정: Swin은 window_size 존재
    span = T
    win = getattr(m, "window_size", None)
    if isinstance(win, (tuple, list)) and len(win) >= 2:
        span = int(win[0]) * int(win[1])
    elif isinstance(win, int) and win > 0:
        span = min(T, int(win) * int(win))  # 안전하게

    return int(B), int(T), int(D), int(H), int(Dh), int(span)


def _attn_flops_bytes(B: int, T: int, H: int, Dh: int, span: int):
    """
    FLOPs:
      F = 8·B·T·D²  (Q,K,V,O proj, D=H·Dh)
        + 4·B·H·T·span·Dh  (QK^T + A·V)
      + (옵션) softmax, layernorm 미소항
    Bytes (대략):
      x/y + q/k/v/o + scores + probs + y
    """
    D = H * Dh
    # 1) Projections
    proj = 8.0 * B * T * D * D
    # 2) Core (QK^T + A·V)
    core = 4.0 * B * H * T * span * Dh

    f = 0.0
    if ATTN_FLOP_MODE == "full":
        f = proj + core
    elif ATTN_FLOP_MODE == "projections":
        f = proj
    elif ATTN_FLOP_MODE == "core":
        f = core
    else:
        f = proj + core  # fallback

    # 3) (옵션) softmax / layernorm FLOPs
    if ATTN_INCLUDE_SOFTMAX:
        # softmax ~ exp+sum+div: 대략 3·(B·H·T·span)
        f += 3.0 * B * H * T * span
    if ATTN_INCLUDE_LAYERNORM:
        # LN ~ 5·(B·T·D) (대략치)
        f += 5.0 * B * T * D

    f *= ATTN_FLOP_SCALE

    # ===== Bytes 대략 =====
    # x: B*T*D, y: B*T*D
    xio = (B*T*D + B*T*D)
    # q/k/v/o 각 B*T*D
    qkvo = 4 * B * T * D
    # scores/probs: B*H*T*span (읽고/쓰고 포함: x2 정도)
    scores = 2 * B * H * T * span
    # y head-merge: B*T*D (이미 y 포함되어 있지만 근사치 유지)
    ybytes = B * T * D

    b = (xio + qkvo + scores + ybytes) * (getattr(torch.empty((), dtype=torch.float32), "element_size")() or 4)
    return f, b

def flops_of_single_module(m, x, y):
    xin = next((t for t in x if torch.is_tensor(t)), None)
    yout = y[0] if isinstance(y, (tuple, list)) else y

    if _looks_like_attention(m):
        params = _extract_attn_params(m, xin, yout)
        if params is None:
            return 0.0
        B, T, D, H, Dh, span = params
        f, _ = _attn_flops_bytes(B, T, H, Dh, span)
        return float(f)

    if not (torch.is_tensor(xin) and torch.is_tensor(yout)):
        return 0.0
    if isinstance(m, nn.Linear):
        M = xin.numel() // m.in_features
        return 2.0 * M * m.in_features * m.out_features
    if isinstance(m, nn.Conv2d):
        N, C, Ht, W = xin.shape; Cout, Hout, Wout = yout.shape[1:]
        Kh, Kw = m.kernel_size; g = m.groups
        return 2.0 * N * Cout * Hout * Wout * (C // g) * Kh * Kw
    return 0.0

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

    def flops_of(m, x, y):
        # 타이밍 훅에서 쓸 간단 근사 (trend만 중요)
        xin = next((t for t in x if torch.is_tensor(t)), None)
        yout = y[0] if isinstance(y, (tuple, list)) else y
        if _looks_like_attention(m):
            params = _extract_attn_params(m, xin, yout)
            if params is None: return 0.0
            B, T, D, H, Dh, span = params
            return float(4.0 * B * H * T * span * Dh)  # 간단 근사
        if not (torch.is_tensor(xin) and torch.is_tensor(yout)):
            return 0.0
        if isinstance(m, nn.Linear):
            M = xin.numel() // m.in_features
            return 2.0 * M * m.in_features * m.out_features
        if isinstance(m, nn.Conv2d):
            N, C, Ht, W = xin.shape; Cout, Hout, Wout = yout.shape[1:]
            Kh, Kw = m.kernel_size; g = m.groups
            return 2.0 * N * Cout * Hout * Wout * (C // g) * Kh * Kw
        return 0.0

    class LayerAwareTimingHook:
        def __init__(self):
            self.start = {}
            self.cat_time_s = defaultdict(float)
            self.handles = []
            self.current_major = None

        def is_depthwise(self, m: nn.Conv2d) -> bool:
            return m.groups == m.in_channels == m.out_channels

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
            for h in self.handles:
                h.remove()
            self.handles.clear()
            self.start.clear()

        def classify_major(self, m):
            if isinstance(m, nn.Conv2d):
                if self.is_depthwise(m):
                    return "depthwise"
                return "conv"
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
            major = self.classify_major(m)
            if major is not None:
                self.current_major = major

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
                if self.current_major in ("conv", "depthwise"):
                    self.cat_time_s[self.current_major] += dt
                else:
                    self.cat_time_s["nonlinear"] += dt
                return

            if minor == "pool":
                if self.current_major in ("conv", "depthwise"):
                    self.cat_time_s[self.current_major] += dt
                else:
                    self.cat_time_s["conv"] += dt
                return

            self.cat_time_s["nonlinear"] += dt

        def results(self):
            return {
                op: {"time_ms": t * 1000.0}
                for op, t in self.cat_time_s.items()
            }

    model.eval()
    hook = LayerAwareTimingHook()
    hook.register(model)
    with profile(activities=[ProfilerActivity.CPU],
                 with_flops=True, record_shapes=False, profile_memory=False) as prof:
        with torch.no_grad():
            with record_function("profile_run"): _ = model(input_tensor)
    per_op_raw = hook.results()
    hook.remove()

    # PyTorch 내부 오퍼레이터 FLOPs 총합(참고용)
    total_flops = sum(evt.flops for evt in prof.key_averages() if getattr(evt, "flops", 0))

    per_op = per_op_raw

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
def get_attn_module(block):
    return (getattr(block, "mha", None)
            or getattr(block, "self_attention", None)
            or getattr(block, "attn", None)
            or getattr(block, "attention", None))

def attn_shapes_from_block(block, xin: torch.Tensor):
    attn = get_attn_module(block)
    if attn is None:
        raise AttributeError(f"{type(block)}: attention module not found")
    batch_first = bool(getattr(attn, "batch_first", True))
    D = int(getattr(attn, "embed_dim", xin.shape[-1]))
    H = int(getattr(attn, "num_heads", max(1, D // max(1, getattr(attn, "head_dim", D)))))
    # 입력에서 B,T 추출
    if batch_first:   # [B,T,D]
        B, T, Din = map(int, xin.shape[-3:]) if xin.dim()==3 else (int(xin.shape[0]), int(xin.shape[-2]), int(xin.shape[-1]))
    else:             # [T,B,D]
        T, B, Din = map(int, xin.shape[-3:])
    if D == 0: D = Din
    Dh = D // H
    span = T  # ViT-B/16: full attention
    return B, T, D, H, Dh, span

def flops_bytes_attention(block, xin: torch.Tensor):
    B, T, D, H, Dh, span = attn_shapes_from_block(block, xin)
    proj_f = 8.0 * B * T * D * D
    core_f = 4.0 * B * H * T * span * Dh     # = QK^T(2·…) + A·V(2·…)
    f = proj_f + core_f
    # bytes는 근사치라 그대로 두셔도 됩니다
    proj_b_one = ((B*T*D) + (D*D) + (B*T*D)) * BYTES_PER
    proj_b = 4 * proj_b_one
    core_b = ((B*H*T*span)*2 + (B*H*T*Dh)*3 + (B*H*T*Dh)) * BYTES_PER
    b = proj_b + core_b
    return {
        "attn_proj_FLOPs": proj_f,
        "attn_core_FLOPs": core_f,
        "attn_proj_Bytes": proj_b,
        "attn_core_Bytes": core_b,
        "total_FLOPs": f,
        "total_Bytes": b
    }, f, b

# =========================================================
# (2) FLOPs / Bytes 집계 훅 (embedding/pool/attention 포함)
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
        if isinstance(m, nn.Linear): return "gemm"
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
        self.cat_flops[cat] += f; self.cat_bytes[cat] += b
    def totals(self): return dict(self.cat_flops), dict(self.cat_bytes)

# =========================================================
# (3) 디바이스별 선형 회귀식 (GF/s = a + b * OI)
# =========================================================
# =========================================================
# (3) 디바이스별 선형 회귀식 (GF/s = a + b * OI)
#   ※ 최신 OI 기반 계수 (attention 포함)
# =========================================================
# REG_THROUGHPUT_OI_LINEAR = {
#     "rpi3": {
#         "attention": {"a": 19.0355, "b": -0.1873, "R2": 0.5571},
#         "conv":      {"a": 0.4331,  "b": 0.0392,  "R2": 0.8761},
#         "depthwise": {"a": 0.0382,  "b": 0.0767,  "R2": 0.2472},
#         "gemm":      {"a": 11.5620, "b": -0.0196, "R2": 0.3201},
#     },
#     "rpi4": {
#         "attention": {"a": 33.7565, "b": -0.3030, "R2": 0.3375},
#         "conv":      {"a": 0.6133,  "b": 0.0851,  "R2": 0.9249},
#         "depthwise": {"a": 0.3506,  "b": 0.0589,  "R2": 0.0407},
#         "gemm":      {"a": 13.2079, "b": 0.1029,  "R2": 0.4976},
#     },
#     "rpi5": {
#         "attention": {"a": 72.5994, "b": -0.4578, "R2": 0.0859},
#         "conv":      {"a": 2.9602,  "b": 0.2064,  "R2": 0.8285},
#         "depthwise": {"a": 3.0106,  "b": -0.5782, "R2": 0.1117},
#         "gemm":      {"a": 57.6281, "b": 0.1131, "R2": 0.2623},
#     },
# }
# REG_THROUGHPUT_OI_LINEAR = {
#     "rpi3": {
#         "attention": {"a": -43.0812,  "b": 12.4977,   "R2": 0.5146, "n": 150},
#         "conv":      {"a": -5.4056 ,   "b": 1.9866,    "R2": 0.8784, "n": 150},
#         "depthwise": {"a": -95.2139,  "b": 117.6976,  "R2": 0.6047, "n": 150},
#         "gemm":      {"a": 13.1533,   "b": -0.7377,   "R2": 0.0287, "n": 150},
#     },
#     "rpi4": {
#         "attention": {"a": -86.9795,  "b": 25.3412,   "R2": 0.5380, "n": 150},
#         "conv":      {"a": -12.0156,   "b": 4.3032,    "R2": 0.9288, "n": 150},
#         "depthwise": {"a": -155.9130, "b": 192.9012,  "R2": 0.4724, "n": 150},
#         "gemm":      {"a": -13.1620,  "b": 8.0604,    "R2": 0.6016, "n": 150},
#     },
#     "rpi5": {
#         "attention": {"a": -137.8216, "b": 45.2546,   "R2": 0.2293, "n": 150},
#         "conv":      {"a": -19.105,  "b": 8.5601,    "R2": 0.7806, "n": 150},
#         "depthwise": {"a": -119.7745, "b": 149.8485,  "R2": 0.0244, "n": 150},
#         "gemm":      {"a": 31.9861,   "b": 8.4167,    "R2": 0.1132, "n": 150},
#     },
# }
REG_THROUGHPUT_OI_LINEAR = {
    "rpi3": {
        "attention": {"a": -6.4330,   "b": 0.2440,   "R2": 0.4916,  "n": 150},
        "conv":      {"a":  0.0552,   "b": 0.0487,   "R2": 0.9151,  "n": 150},
        "depthwise": {"a": 0.0, "b": 0.0870,  "R2": 0.002,  "n": 150},
        "gemm":      {"a": 11.5501,   "b": -0.0196,  "R2": 0.1015,  "n": 150},
    },

    "rpi4": {
        "attention": {"a": -13.0094,  "b": 0.5014,   "R2": 0.5278,  "n": 150},
        "conv":      {"a": 0.5226,   "b": 0.0977,   "R2": 0.9528,  "n": 150},
        "depthwise": {"a": 0.0000, "b": 0.2041,  "R2": 0.0022,  "n": 150},
        "gemm":      {"a": 13.2254,   "b": 0.1030,   "R2": 0.4939,  "n": 150},
    },

    "rpi5": {
        "attention": {"a": -5.2551,   "b": 0.8862,   "R2": 0.2204,  "n": 150},
        "conv":      {"a": 3.1018,    "b": 0.2145,   "R2": 0.7758,  "n": 150},
        "depthwise": {"a": 0, "b": 0.7547,  "R2": 0.0005,  "n": 150},
        "gemm":      {"a": 58.5960,   "b": 0.1194,   "R2": 0.1144,  "n": 150},
    }
}
# 회귀 단계에서 카테고리 매핑

# =========================================================
# (4) 정적 한계값: OI cap + 처리량(GF/s) floor/ceil
# =========================================================
GLOBAL_OI_CAP = 50.0
OI_CAPS = {"gemm":80.0, "conv":40.0, "depthwise":2.5, "nonlinear":50.0, "attention":50.0}
STATIC_THR_FLOOR_GLOBAL = 0.20
STATIC_THR_FLOOR_BY_OP = {"nonlinear": 0.50, "depthwise": 0.2, "conv": 0.20, "gemm": 1.00, "attention": 1.00}
STATIC_THR_CEIL_GLOBAL = None
STATIC_THR_CEIL_BY_OP = {}
FIXED_OI_BY_OP = {
    "depthwise": 2.5,
}
def _cap_oi_static(oi, op, use_opwise_cap=True):
    cap = OI_CAPS.get(op, GLOBAL_OI_CAP) if use_opwise_cap else GLOBAL_OI_CAP
    if not math.isfinite(oi): return 0.0
    return max(0.0, min(oi, cap))

def _apply_thr_bounds_static(thr, op):
    floor = STATIC_THR_FLOOR_BY_OP.get(op, STATIC_THR_FLOOR_GLOBAL)
    ceil  = STATIC_THR_CEIL_BY_OP.get(op, STATIC_THR_CEIL_GLOBAL)
    if floor is not None: thr = max(thr, floor)
    if ceil is not None:  thr = min(thr, ceil)
    return thr

OP_ALIAS_FOR_REG = {
    #"attention": "gemm",
    "embedding": "gemm",
    "pool": "conv",
}
def estimate_latency_regression_linear(device_key: str,
                                       cat_flops: dict,
                                       cat_bytes: dict,
                                       use_opwise_cap: bool = True):
    coeffs = REG_THROUGHPUT_OI_LINEAR.get(device_key, {})
    per = {}; total_ms = 0.0

    for op, flops in cat_flops.items():

        base_op = OP_ALIAS_FOR_REG.get(op, op)
        if base_op == "pointwise":
            continue

        bytes_ = cat_bytes.get(op, 0)
        if flops <= 0 or bytes_ <= 0:
            continue

        # ====== OI 계산 ======
        # ====== OI 계산 ======
        if base_op in FIXED_OI_BY_OP:
            oi_raw = FIXED_OI_BY_OP[base_op]
        else:
            oi_raw = flops / bytes_

        oi_cap = _cap_oi_static(oi_raw, base_op, use_opwise_cap)


        gflops = flops / 1e9

        # ====== 선형 모델: thr = a + b * OI ======
        reg = coeffs.get(base_op, {"a": 0.0, "b": 0.0})
        thr = reg["a"] + reg["b"] * oi_cap

        # throughput 보정
        thr = _apply_thr_bounds_static(thr, base_op)
        if not math.isfinite(thr) or thr <= 0:
            thr = _apply_thr_bounds_static(1e-3, base_op)

        # latency = FLOPs / throughput
        t_ms = (gflops / thr) * 1e3

        per[op] = {
            "OI_raw": oi_raw,
            "OI_capped": oi_cap,
            "GFLOPs": gflops,
            "Throughput(GF/s)": thr,
            "Latency(ms)": t_ms,
            "R2": reg.get("R2", 0.0),
            "Bytes(MB)": bytes_ / 1e6,
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

def pretty_print_measured_perop(per_op: dict, title="Measured per-op (this host)"):
    print(f"\n== {title} ==")
    print(f"{'Op':<12}{'GFLOPs':>10}{'Time(ms)':>12}{'GF/s(measured)':>16}")
    print("-" * 56)
    tot_gf = tot_ms = 0.0
    for op, d in per_op.items():
        gf = d.get("flops", 0.0); ms = d.get("time_ms", 0.0)
        thr = (gf / (ms/1000.0)) if ms > 0 else 0.0
        print(f"{op:<12}{gf:10.3f}{ms:12.3f}{thr:16.3f}")
        tot_gf += gf; tot_ms += ms
    tot_thr = (tot_gf / (tot_ms/1000.0)) if tot_ms > 0 else 0.0
    print("-" * 56)
    print(f"{'TOTAL':<12}{tot_gf:10.3f}{tot_ms:12.3f}{tot_thr:16.3f}")

# =========================================================
# (5) 총합 OI 참고 훅 (LLM: Embedding/Attention 포함)
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
        in_b = sum(t.element_size() * t.numel() for t in x if torch.is_tensor(t))
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
def compute_real_op_latency(measured_ms: float, per_op_meas: dict):
    """
    실제 inference time(measured_ms)을 기준으로
    연산자별 latency를 비율로 분배
    """
    # hook으로 측정된 연산자별 시간(ms)
    op_hook_times = {
        op: d["time_ms"]
        for op, d in per_op_meas.items()
        if d.get("time_ms", 0.0) > 0
    }

    total_hook_time = sum(op_hook_times.values())

    if total_hook_time <= 0:
        return {}

    # 실제 inference time 기준으로 재분배
    real_op_times = {
        op: measured_ms * (t / total_hook_time)
        for op, t in op_hook_times.items()
    }

    return real_op_times
def pretty_print_real_op_latency(real_op_times: dict, measured_ms: float):
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
# (6) MAIN
# =========================================================
USE_VISION_TRANSFORMER = False   # True면 ViT/스윈 중 선택
USE_SWINT = False                 # True면 Swin v2 Tiny, False면 ViT-B/16

if __name__ == "__main__":
    if USE_MINI_LM:
        model = MiniTransformerLM(
            vocab_size=320, seq_len=SEQ_LEN, d_model=128,
            n_head=2, n_layer=2, dropout=0.0
        ).to(DEVICE).eval()
        x = torch.randint(0, 320, (1, SEQ_LEN), device=DEVICE)

    elif USE_VISION_TRANSFORMER:
        if USE_SWINT:
            weights = Swin_V2_T_Weights.IMAGENET1K_V1
            model = swin_v2_t(weights=weights).to(DEVICE).eval()
            x = torch.randn(1, 3, 224, 224, device=DEVICE)
        else:
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            model = vit_b_16(weights=weights).to(DEVICE).eval()
            x = torch.randn(1, 3, 224, 224, device=DEVICE)
    else:
        model = models.resnet50().to(DEVICE).eval()
        x = torch.randn(1, 3, 224, 224, device=DEVICE)

    total_flops, measured_ms, per_op_meas = analyze_model_operations(model, x, num_runs=NUM_RUNS)

    hook = OpCategorizerHook(); hook.register(model)
    with torch.no_grad(): model(x)
    cat_flops, cat_bytes = hook.totals(); hook.remove()
    merged_flops, merged_bytes = merge_pointwise_into_conv_stats(cat_flops, cat_bytes)

    for dev in ["rpi3", "rpi4", "rpi5"]:
        per, total_ms = estimate_latency_regression_linear(
            dev, merged_flops, merged_bytes, use_opwise_cap=True
        )
        pretty_print_estimate_linear(dev, per, total_ms)

    estimate_oi_by_hook(model, x, total_flops)
    print(f"\n[Measured on this host] Avg of {NUM_RUNS} runs: {measured_ms:.3f} ms")
    # === 실제 inference time 기준 연산자별 latency 계산 ===
    real_op_times = compute_real_op_latency(
        measured_ms,
        per_op_meas
    )
    pretty_print_real_op_latency(real_op_times, measured_ms)
