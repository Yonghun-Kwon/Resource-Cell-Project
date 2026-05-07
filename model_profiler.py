"""
PyTorch Model Profiler
MACs / memory / OI / latency / throughput — per user-specified operator types.
"""

import torch
import torch.nn as nn
import time
import csv
import os
from dataclasses import dataclass
from typing import List, Dict, Type, Callable, Optional, Tuple
import numpy as np


WARMUP_RUNS = 3
MEASURE_RUNS = 10
RESULTS_DIR  = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────
@dataclass
class OpProfile:
    layer_name:       str
    op_type:          str
    param_desc:       str
    macs:             float
    mem_bytes:        float
    latency_ms:       float
    throughput_gmacs: float
    oi:               float   # MACs / byte


# ─────────────────────────────────────────────
# MACs / memory counter registry
# CounterFn(module, input_tuple, output) -> (macs, mem_bytes, desc)
# ─────────────────────────────────────────────
CounterFn = Callable[[nn.Module, tuple, torch.Tensor], Tuple[float, float, str]]
_COUNTERS: Dict[Type[nn.Module], CounterFn] = {}


def register_counter(cls: Type[nn.Module]):
    def decorator(fn: CounterFn):
        _COUNTERS[cls] = fn
        return fn
    return decorator


@register_counter(nn.Linear)
def _count_linear(m, inp, out):
    x     = inp[0]
    batch = x.numel() // x.shape[-1]
    macs  = float(batch * m.in_features * m.out_features)
    mem   = (x.numel() + out.numel() + m.weight.numel()) * 4
    if m.bias is not None:
        mem += m.bias.numel() * 4
    return macs, float(mem), f"in={m.in_features},out={m.out_features},batch={batch}"


@register_counter(nn.Conv2d)
def _count_conv2d(m, inp, out):
    x            = inp[0]
    B, Cin, H, W = x.shape
    Cout         = out.shape[1]
    Hout, Wout   = out.shape[2], out.shape[3]
    kh = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
    kw = m.kernel_size[1] if isinstance(m.kernel_size, tuple) else m.kernel_size
    macs = float(B * Cout * Hout * Wout * (Cin // m.groups) * kh * kw)
    mem  = (x.numel() + out.numel() + m.weight.numel()) * 4
    if m.bias is not None:
        mem += m.bias.numel() * 4
    tag = " (DW)" if m.groups == Cin and Cin > 1 else ""
    return macs, float(mem), f"B={B},Cin={Cin},Cout={Cout},H={H},W={W},K={kh}x{kw}{tag}"


@register_counter(nn.MultiheadAttention)
def _count_mha(m, inp, out):
    q = inp[0]
    if m.batch_first:
        B, S, E = q.shape
    else:
        S, B, E = q.shape
    H = m.num_heads
    D = E // H
    # QK^T: B*H*S*S*D  +  AV: B*H*S*S*D  =  2*B*H*S*S*D
    macs = float(2 * B * H * S * S * D)
    mem  = float((3*B*H*S*D + 2*B*H*S*S + B*H*S*D) * 4)
    return macs, mem, f"B={B},H={H},S={S},D={D}"


@register_counter(nn.BatchNorm2d)
def _count_bn(m, inp, out):
    n    = inp[0].numel()
    macs = float(2 * n)   # normalize + scale&shift
    mem  = float(3 * n * 4 + 2 * m.num_features * 4)
    return macs, mem, f"C={m.num_features},N={n}"


@register_counter(nn.LayerNorm)
def _count_ln(m, inp, out):
    n      = inp[0].numel()
    norm_n = 1
    for s in m.normalized_shape:
        norm_n *= s
    macs = float(2 * n)
    mem  = float(3 * n * 4 + 2 * norm_n * 4)
    return macs, mem, f"shape={list(m.normalized_shape)},N={n}"


def _count_ew(m, inp, out):
    n = inp[0].numel()
    return float(n), float(3 * n * 4), f"N={n}"

for _cls in (nn.ReLU, nn.GELU, nn.Sigmoid, nn.Tanh, nn.SiLU,
             nn.Hardswish, nn.LeakyReLU, nn.ELU):
    _COUNTERS[_cls] = _count_ew


@register_counter(nn.MaxPool2d)
def _count_maxpool(m, inp, out):
    n  = out.numel()
    kh = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
    kw = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[1]
    return float(n * kh * kw), float((inp[0].numel() + n) * 4), \
           f"K={kh}x{kw},out={list(out.shape)}"


@register_counter(nn.AvgPool2d)
def _count_avgpool(m, inp, out):
    n  = out.numel()
    kh = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
    kw = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[1]
    return float(n * kh * kw), float((inp[0].numel() + n) * 4), \
           f"K={kh}x{kw},out={list(out.shape)}"


# ─────────────────────────────────────────────
# ModelProfiler
# ─────────────────────────────────────────────
class ModelProfiler:
    """
    Per-layer profiler for user-specified operator types.

    Parameters
    ----------
    target_ops : list of nn.Module subclasses to profile.
                 None → profile all registered types.
    warmup     : warm-up forward passes before measurement.
    runs       : timed forward passes (latency = mean).

    Usage
    -----
    profiler = ModelProfiler(target_ops=[nn.Conv2d, nn.Linear])
    profiler.profile(model, dummy_input)
    profiler.summary()
    profiler.save_csv("results/profile.csv")
    """

    SUPPORTED_OPS = list(_COUNTERS.keys())

    def __init__(
        self,
        target_ops: Optional[List[Type[nn.Module]]] = None,
        warmup: int = WARMUP_RUNS,
        runs:   int = MEASURE_RUNS,
    ):
        self.target_ops = set(target_ops) if target_ops else set(_COUNTERS.keys())
        self.warmup     = warmup
        self.runs       = runs
        self._profiles: List[OpProfile] = []

    # ── public API ────────────────────────────────────────────────────

    def profile(self, model: nn.Module, *inputs) -> "ModelProfiler":
        model.eval()

        targets: Dict[str, nn.Module] = {
            name: m for name, m in model.named_modules()
            if type(m) in self.target_ops and type(m) in _COUNTERS
        }
        if not targets:
            print("[Profiler] No matching operators found in model.")
            print(f"           Registered types: {[c.__name__ for c in _COUNTERS]}")
            return self

        print(f"[Profiler] Found {len(targets)} target layer(s): "
              f"{[f'{n}({type(m).__name__})' for n, m in targets.items()]}")

        # ── pass 1: collect MACs / memory (single forward) ───────────
        macs_mem: Dict[str, tuple] = {}
        handles = []
        for name, m in targets.items():
            def _hook(mod, inp, out, _n=name):
                macs_mem[_n] = _COUNTERS[type(mod)](mod, inp, out)
            handles.append(m.register_forward_hook(_hook))

        with torch.no_grad():
            model(*inputs)
        for h in handles:
            h.remove()

        # ── warm-up ───────────────────────────────────────────────────
        with torch.no_grad():
            for _ in range(self.warmup):
                model(*inputs)

        # ── pass 2: timed runs ────────────────────────────────────────
        raw_times: Dict[str, List[float]] = {n: [] for n in targets}
        t0_store:  Dict[str, float]       = {}
        handles = []
        for name, m in targets.items():
            def _pre(mod, inp, _n=name):
                t0_store[_n] = time.perf_counter()
            def _post(mod, inp, out, _n=name):
                raw_times[_n].append(time.perf_counter() - t0_store[_n])
            handles.append(m.register_forward_pre_hook(_pre))
            handles.append(m.register_forward_hook(_post))

        with torch.no_grad():
            for _ in range(self.runs):
                model(*inputs)
        for h in handles:
            h.remove()

        # ── build OpProfile list ──────────────────────────────────────
        self._profiles = []
        for name, m in targets.items():
            if name not in macs_mem:
                continue
            macs, mem, desc = macs_mem[name]
            times_ms = [t * 1000 for t in raw_times.get(name, [])]
            lat  = float(np.mean(times_ms)) if times_ms else 0.0
            tput = (macs / 1e9) / (lat / 1000) if lat > 1e-9 else 0.0
            oi   = macs / mem if mem > 0 else 0.0
            self._profiles.append(OpProfile(
                layer_name=name, op_type=type(m).__name__,
                param_desc=desc, macs=macs, mem_bytes=mem,
                latency_ms=lat, throughput_gmacs=tput, oi=oi,
            ))
        return self

    @property
    def profiles(self) -> List[OpProfile]:
        return list(self._profiles)

    def summary(self):
        if not self._profiles:
            print("No profiling data. Call .profile() first.")
            return
        W = 108
        print("\n" + "=" * W)
        print(f"  {'Layer':<32} {'Type':<20} {'MACs':>10} {'Mem(MB)':>9} "
              f"{'OI':>7} {'Lat(ms)':>9} {'Tput(GMACs/s)':>13}")
        print("-" * W)
        for p in self._profiles:
            if   p.macs >= 1e9: ms = f"{p.macs/1e9:.3f}G"
            elif p.macs >= 1e6: ms = f"{p.macs/1e6:.2f}M"
            else:               ms = f"{p.macs/1e3:.1f}K"
            mem_mb = p.mem_bytes / (1024 ** 2)
            print(f"  {p.layer_name:<32} {p.op_type:<20} {ms:>10} {mem_mb:>9.3f} "
                  f"{p.oi:>7.3f} {p.latency_ms:>9.3f} {p.throughput_gmacs:>13.4f}")
        print("=" * W)
        total_macs = sum(p.macs      for p in self._profiles)
        total_mem  = sum(p.mem_bytes for p in self._profiles)
        total_lat  = sum(p.latency_ms for p in self._profiles)
        print(f"  Total  MACs  : {total_macs/1e9:.4f} GMACs")
        print(f"  Total  Mem   : {total_mem/(1024**2):.2f} MB")
        print(f"  Σ Latency    : {total_lat:.3f} ms   |   Layers: {len(self._profiles)}")
        print("=" * W)

    def save_csv(self, path: str):
        if not self._profiles:
            print("No data to save.")
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "layer_name", "op_type", "param_desc",
                "macs", "mem_bytes", "latency_ms", "throughput_gmacs", "oi",
            ])
            writer.writeheader()
            for p in self._profiles:
                writer.writerow(p.__dict__)
        print(f"✅ Profile CSV saved → {path}")


# ─────────────────────────────────────────────
# ResNet50 profiling
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import torchvision.models as tvm

    model = tvm.resnet50(weights=None)
    model.eval()
    x = torch.randn(1, 3, 224, 224)

    # ── 1. Conv2d + BatchNorm2d + ReLU ──────────────────────────────
    print("\n" + "=" * 60)
    print("  ResNet50 — Conv2d / BatchNorm2d / ReLU")
    print("=" * 60)
    p1 = ModelProfiler(
        target_ops=[nn.Conv2d, nn.BatchNorm2d, nn.ReLU],
        warmup=3, runs=10,
    )
    p1.profile(model, x)
    p1.summary()
    p1.save_csv(os.path.join(RESULTS_DIR, "resnet50_conv_bn_relu.csv"))

    # ── 2. Linear (fc layer only) ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ResNet50 — Linear (fc)")
    print("=" * 60)
    p2 = ModelProfiler(target_ops=[nn.Linear], warmup=3, runs=10)
    p2.profile(model, x)
    p2.summary()
    p2.save_csv(os.path.join(RESULTS_DIR, "resnet50_linear.csv"))

    # ── 3. All ops combined ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ResNet50 — All registered ops")
    print("=" * 60)
    p3 = ModelProfiler(
        target_ops=[nn.Conv2d, nn.BatchNorm2d, nn.ReLU,
                    nn.Linear, nn.MaxPool2d, nn.AvgPool2d],
        warmup=3, runs=10,
    )
    p3.profile(model, x)
    p3.summary()
    p3.save_csv(os.path.join(RESULTS_DIR, "resnet50_all.csv"))
