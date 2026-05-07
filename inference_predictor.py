"""
Inference Time Predictor
두 가지 회귀 모델로 레이어별 지연시간 예측:
  - OI 모델  : log(OI)      → log(throughput)  → latency
  - Mem 모델 : log(mem_bytes) → log(latency_ms)   (메모리 접근량 직접 회귀)
"""

import torch
import torch.nn as nn
import csv
import math
import os
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

from model_profiler import ModelProfiler, OpProfile


RESULTS_DIR = "results"
BENCH_CSV   = os.path.join(RESULTS_DIR, "benchmark_results.csv")

# benchmark CSV는 FLOPs 기준 OI / throughput_gflops 사용
# MACs = FLOPs / factor → OI_flops = OI_macs * factor
FLOPS_FACTOR: Dict[str, float] = {
    "GEMM":        2.0,
    "Convolution": 2.0,
    "Depthwise":   2.0,
    "Attention":   2.0,
    "Elementwise": 1.0,
}


# ─────────────────────────────────────────────
# Op type → benchmark name
# ─────────────────────────────────────────────
def op_to_bench(profile: OpProfile) -> Optional[str]:
    t = profile.op_type
    if t == "Linear":
        return "GEMM"
    if t == "Conv2d":
        return "Depthwise" if "(DW)" in profile.param_desc else "Convolution"
    if t == "MultiheadAttention":
        return "Attention"
    if t in ("ReLU", "GELU", "Sigmoid", "Tanh", "SiLU",
             "Hardswish", "LeakyReLU", "ELU",
             "BatchNorm2d", "LayerNorm", "MaxPool2d", "AvgPool2d"):
        return "Elementwise"
    return None


# ─────────────────────────────────────────────
# 공통 회귀 결과 dataclass
# ─────────────────────────────────────────────
@dataclass
class RegressionModel:
    benchmark:  str
    precision:  str
    x_label:    str    # "oi" or "mem_bytes"
    y_label:    str    # "throughput_gflops" or "latency_ms"
    slope:      float
    intercept:  float
    r2:         float
    n_samples:  int


# ─────────────────────────────────────────────
# OI 모델: log(OI) → log(throughput_gflops)
# ─────────────────────────────────────────────
def load_oi_models(
    csv_path: str,
    precision: str = "Float32",
) -> Dict[str, RegressionModel]:
    rows = _read_csv(csv_path)
    models: Dict[str, RegressionModel] = {}
    print(f"\n  [OI 모델] precision={precision}")
    for bench in ("GEMM", "Convolution", "Depthwise", "Elementwise", "Attention"):
        pts = [(float(r["oi"]), float(r["throughput_gflops"]))
               for r in rows if r["benchmark"] == bench and r["precision"] == precision]
        if len(pts) < 2:
            continue
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        slope, intercept, r2 = _fit(np.log10(np.maximum(xs, 1e-9)),
                                     np.log10(np.maximum(ys, 1e-9)))
        models[bench] = RegressionModel(
            benchmark=bench, precision=precision,
            x_label="oi", y_label="throughput_gflops",
            slope=slope, intercept=intercept, r2=r2, n_samples=len(pts),
        )
        print(f"    {bench:<12}  slope={slope:.4f}  intercept={intercept:.4f}  R²={r2:.4f}")
    return models


# ─────────────────────────────────────────────
# Mem 모델: log(mem_bytes) → log(latency_ms)
# ─────────────────────────────────────────────
def load_mem_models(
    csv_path: str,
    precision: str = "Float32",
) -> Dict[str, RegressionModel]:
    rows = _read_csv(csv_path)
    models: Dict[str, RegressionModel] = {}
    print(f"\n  [Mem 모델] precision={precision}")
    for bench in ("GEMM", "Convolution", "Depthwise", "Elementwise", "Attention"):
        pts = [(float(r["mem_bytes"]), float(r["latency_ms"]))
               for r in rows if r["benchmark"] == bench and r["precision"] == precision]
        if len(pts) < 2:
            continue
        xs = np.array([p[0] for p in pts])
        ys = np.array([p[1] for p in pts])
        slope, intercept, r2 = _fit(np.log10(np.maximum(xs, 1e-9)),
                                     np.log10(np.maximum(ys, 1e-9)))
        models[bench] = RegressionModel(
            benchmark=bench, precision=precision,
            x_label="mem_bytes", y_label="latency_ms",
            slope=slope, intercept=intercept, r2=r2, n_samples=len(pts),
        )
        print(f"    {bench:<12}  slope={slope:.4f}  intercept={intercept:.4f}  R²={r2:.4f}")
    return models


# ─────────────────────────────────────────────
# 예측 함수
# ─────────────────────────────────────────────
def _predict_oi(macs: float, oi_macs: float,
                reg: RegressionModel, bench: str) -> float:
    """OI → throughput → latency 변환."""
    factor      = FLOPS_FACTOR.get(bench, 1.0)
    oi_flops    = oi_macs * factor
    tput_gflops = 10 ** (reg.slope * math.log10(max(oi_flops, 1e-9)) + reg.intercept)
    flops       = macs * factor
    return (flops / 1e9) / tput_gflops * 1000  # ms


def _predict_mem(mem_bytes: float, reg: RegressionModel) -> float:
    """mem_bytes → latency 직접 예측."""
    return 10 ** (reg.slope * math.log10(max(mem_bytes, 1e-9)) + reg.intercept)  # ms


# ─────────────────────────────────────────────
# Per-layer prediction result
# ─────────────────────────────────────────────
@dataclass
class LayerPrediction:
    layer_name:        str
    op_type:           str
    bench_type:        str
    macs:              float
    mem_bytes:         float
    oi_macs:           float
    actual_lat_ms:     float
    # OI 모델
    pred_oi_ms:        float
    err_oi_pct:        float
    r2_oi:             float
    # Mem 모델
    pred_mem_ms:       float
    err_mem_pct:       float
    r2_mem:            float


# ─────────────────────────────────────────────
# Main predictor
# ─────────────────────────────────────────────
def predict_model_latency(
    model:      nn.Module,
    inputs:     list,
    target_ops: Optional[list] = None,
    precision:  str = "Float32",
    bench_csv:  str = BENCH_CSV,
    warmup:     int = 3,
    runs:       int = 10,
) -> List[LayerPrediction]:
    print("\n══ 회귀 모델 학습 ════════════════════════════════════════")
    oi_models  = load_oi_models(bench_csv, precision)
    mem_models = load_mem_models(bench_csv, precision)

    print("\n══ 모델 프로파일링 ═══════════════════════════════════════")
    profiler = ModelProfiler(target_ops=target_ops, warmup=warmup, runs=runs)
    profiler.profile(model, *inputs)

    results: List[LayerPrediction] = []
    for p in profiler.profiles:
        bench = op_to_bench(p)
        if bench is None:
            continue
        oi_reg  = oi_models.get(bench)
        mem_reg = mem_models.get(bench)
        if oi_reg is None or mem_reg is None:
            continue

        pred_oi  = _predict_oi(p.macs, p.oi, oi_reg, bench)
        pred_mem = _predict_mem(p.mem_bytes, mem_reg)

        def _err(pred, actual):
            return (pred - actual) / actual * 100 if actual > 1e-9 else 0.0

        results.append(LayerPrediction(
            layer_name=p.layer_name, op_type=p.op_type, bench_type=bench,
            macs=p.macs, mem_bytes=p.mem_bytes, oi_macs=p.oi,
            actual_lat_ms=p.latency_ms,
            pred_oi_ms=pred_oi,   err_oi_pct=_err(pred_oi,  p.latency_ms), r2_oi=oi_reg.r2,
            pred_mem_ms=pred_mem, err_mem_pct=_err(pred_mem, p.latency_ms), r2_mem=mem_reg.r2,
        ))

    return results


# ─────────────────────────────────────────────
# Display & export
# ─────────────────────────────────────────────
def print_predictions(results: List[LayerPrediction]):
    W = 130
    print("\n" + "=" * W)
    print(f"  {'Layer':<34} {'Type':<10} {'Bench':<12} "
          f"{'Actual':>9} "
          f"{'OI-Pred':>9} {'OI-Err%':>8} {'R²(OI)':>7}  "
          f"{'Mem-Pred':>9} {'Mem-Err%':>9} {'R²(Mem)':>8}")
    print("-" * W)
    for p in results:
        print(f"  {p.layer_name:<34} {p.op_type:<10} {p.bench_type:<12} "
              f"{p.actual_lat_ms:>9.4f} "
              f"{p.pred_oi_ms:>9.4f} {p.err_oi_pct:>+8.2f}% {p.r2_oi:>7.4f}  "
              f"{p.pred_mem_ms:>9.4f} {p.err_mem_pct:>+9.2f}% {p.r2_mem:>8.4f}")
    print("=" * W)

    total_actual  = sum(p.actual_lat_ms for p in results)
    total_oi      = sum(p.pred_oi_ms    for p in results)
    total_mem     = sum(p.pred_mem_ms   for p in results)
    err_oi_total  = (total_oi  - total_actual) / total_actual * 100 if total_actual else 0
    err_mem_total = (total_mem - total_actual) / total_actual * 100 if total_actual else 0
    print(f"  Σ Actual    : {total_actual:.4f} ms")
    print(f"  Σ OI-Pred   : {total_oi:.4f} ms  (error: {err_oi_total:+.2f}%)")
    print(f"  Σ Mem-Pred  : {total_mem:.4f} ms  (error: {err_mem_total:+.2f}%)")
    print("=" * W)

    # bench-type별 요약
    bench_types = sorted(set(p.bench_type for p in results))
    print(f"\n  {'Bench':<14} {'N':>4}  {'Actual':>10}  "
          f"{'OI-Pred':>9} {'OI-Err%':>8}  {'Mem-Pred':>9} {'Mem-Err%':>9}")
    print("  " + "-" * 70)
    for bt in bench_types:
        sub  = [p for p in results if p.bench_type == bt]
        a    = sum(p.actual_lat_ms for p in sub)
        poi  = sum(p.pred_oi_ms    for p in sub)
        pmem = sum(p.pred_mem_ms   for p in sub)
        eoi  = (poi  - a) / a * 100 if a else 0
        emem = (pmem - a) / a * 100 if a else 0
        print(f"  {bt:<14} {len(sub):>4}  {a:>10.4f}  "
              f"{poi:>9.4f} {eoi:>+8.2f}%  {pmem:>9.4f} {emem:>+9.2f}%")


def save_predictions_csv(results: List[LayerPrediction], path: str):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "layer_name", "op_type", "bench_type",
            "macs", "mem_bytes", "oi_macs",
            "actual_lat_ms",
            "pred_oi_ms",  "err_oi_pct",  "r2_oi",
            "pred_mem_ms", "err_mem_pct", "r2_mem",
        ])
        writer.writeheader()
        for p in results:
            writer.writerow(p.__dict__)
    print(f"\n✅ Prediction CSV saved → {path}")


# ─────────────────────────────────────────────
# 내부 유틸
# ─────────────────────────────────────────────
def _read_csv(path: str) -> List[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _fit(log_x: np.ndarray, log_y: np.ndarray):
    slope, intercept = np.polyfit(log_x, log_y, 1)
    pred   = slope * log_x + intercept
    ss_res = np.sum((log_y - pred) ** 2)
    ss_tot = np.sum((log_y - log_y.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return slope, intercept, r2


# ─────────────────────────────────────────────
# Main: ResNet50
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import torchvision.models as tvm

    model = tvm.resnet50(weights=None)
    model.eval()
    x = torch.randn(1, 3, 224, 224)

    results = predict_model_latency(
        model      = model,
        inputs     = [x],
        target_ops = [nn.Conv2d, nn.BatchNorm2d, nn.ReLU,
                      nn.Linear, nn.MaxPool2d, nn.AvgPool2d],
        precision  = "Float32",
    )

    print_predictions(results)
    save_predictions_csv(
        results,
        os.path.join(RESULTS_DIR, "resnet50_latency_prediction.csv"),
    )
