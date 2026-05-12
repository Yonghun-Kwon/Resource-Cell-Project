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


RESULTS_DIR    = "results"
BENCH_CSV      = os.path.join(RESULTS_DIR, "benchmark_results.csv")
REGRESSION_CSV = os.path.join(RESULTS_DIR, "regression_results.csv")
COMBINED_CSV   = os.path.join(RESULTS_DIR, "combined_regression_results.csv")

# mdb.py 벤치마크 이름 → 내부 op 타입 매핑
MDB_TO_BENCH: Dict[str, str] = {
    "GEMM_ReLU":    "GEMM",
    "Conv_BN_ReLU": "Convolution",
    "DPE_Block":    "Depthwise",
    "Attn_GELU":    "Attention",
}

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
# OI 모델: mdb.py가 저장한 regression_results.csv 로드
#   columns: benchmark, precision, slope, intercept, r2, ...
# ─────────────────────────────────────────────
def load_oi_models(
    csv_path: str,
    precision: str = "Float32",
) -> Dict[str, RegressionModel]:
    rows = _read_csv(csv_path)
    models: Dict[str, RegressionModel] = {}
    print(f"\n  [OI 모델] precision={precision}")
    for r in rows:
        if r["precision"] != precision:
            continue
        bench = MDB_TO_BENCH.get(r["benchmark"])
        if bench is None:
            continue
        m = RegressionModel(
            benchmark=bench, precision=precision,
            x_label="oi", y_label="throughput_gflops",
            slope=float(r["slope"]),
            intercept=float(r["intercept"]),
            r2=float(r["r2"]),
            n_samples=0,
        )
        models[bench] = m
        print(f"    {bench:<12}  slope={m.slope:.4f}  intercept={m.intercept:.4f}  R²={m.r2:.4f}")
    return models


# ─────────────────────────────────────────────
# Mem 모델: benchmark_results.csv에서 직접 피팅
#   mdb.py 이름(GEMM_ReLU 등) → 내부 이름으로 변환 후 그룹화
# ─────────────────────────────────────────────
def load_mem_models(
    csv_path: str,
    precision: str = "Float32",
) -> Dict[str, RegressionModel]:
    rows = _read_csv(csv_path)
    models: Dict[str, RegressionModel] = {}
    print(f"\n  [Mem 모델] precision={precision}")

    from collections import defaultdict
    grouped: Dict[str, list] = defaultdict(list)
    for r in rows:
        if r["precision"] != precision:
            continue
        bench = MDB_TO_BENCH.get(r["benchmark"])
        if bench is None:
            continue
        grouped[bench].append((float(r["mem_bytes"]), float(r["latency_ms"])))

    for bench, pts in grouped.items():
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
# Combined 모델: (OI, mem_bytes) → latency_ms
# combined_regression_results.csv 로드
# ─────────────────────────────────────────────
@dataclass
class CombinedModel:
    benchmark: str
    precision: str
    formula:   str    # "log-log" | "semi-log" | "linear" | "mixed"
    coef_oi:   float  # a
    coef_mem:  float  # b
    intercept: float  # c
    r2:        float


# 형태별 변환 규칙 (mdb.py와 동일하게 유지)
_FORMULA_MODES: Dict[str, tuple] = {
    "log-log":  ("log", "log", "log"),
    "semi-log": ("log", "lin", "log"),
    "linear":   ("lin", "lin", "lin"),
    "mixed":    ("log", "lin", "lin"),
}


def load_combined_models(
    csv_path: str,
    precision: str = "Float32",
) -> Dict[str, CombinedModel]:
    rows = _read_csv(csv_path)
    models: Dict[str, CombinedModel] = {}
    print(f"\n  [Combined 모델] precision={precision}")
    for r in rows:
        if r["precision"] != precision:
            continue
        bench   = MDB_TO_BENCH.get(r["benchmark"], r["benchmark"])
        formula = r.get("formula", "log-log")   # 구버전 CSV 호환
        m = CombinedModel(
            benchmark=bench, precision=precision,
            formula=formula,
            coef_oi=float(r["coef_oi"]),
            coef_mem=float(r["coef_mem"]),
            intercept=float(r["intercept"]),
            r2=float(r["r2"]),
        )
        models[bench] = m
        warn = ""
        if abs(m.coef_oi) > 20 or abs(m.coef_mem) > 20:
            warn = "  ⚠️  계수 과대 — 다중공선성 의심, 클램프 적용됨"
        print(f"    {bench:<12}  formula={m.formula:<10}  "
              f"a={m.coef_oi:.4f}  b={m.coef_mem:.4f}  "
              f"c={m.intercept:.4f}  R²={m.r2:.4f}{warn}")
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


def _predict_combined(oi_macs: float, mem_bytes: float,
                       m: CombinedModel, bench: str) -> float:
    """
    formula에 따라 분기해서 latency(ms) 예측.

    formula 변환 규칙:
      log-log  : log(lat) = a·log(OI_flops) + b·log(mem) + c
      semi-log : log(lat) = a·log(OI_flops) + b·mem      + c
      linear   : lat      = a·OI_flops      + b·mem      + c
      mixed    : lat      = a·log(OI_flops) + b·mem      + c

    log 출력 형태는 [-3, 6] 클램프 → 오버플로우 방지.
    """
    factor   = FLOPS_FACTOR.get(bench, 1.0)
    oi_flops = max(oi_macs * factor, 1e-9)

    x1m, x2m, ym = _FORMULA_MODES.get(m.formula, ("log", "log", "log"))

    x1 = math.log10(oi_flops)  if x1m == "log" else oi_flops
    x2 = math.log10(max(mem_bytes, 1e-9)) if x2m == "log" else mem_bytes

    y_raw = m.coef_oi * x1 + m.coef_mem * x2 + m.intercept

    if ym == "log":
        if not math.isfinite(y_raw):
            y_raw = 6.0
        y_raw = max(-3.0, min(y_raw, 6.0))
        return 10 ** y_raw
    return max(y_raw, 1e-9)


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
    # Combined 모델 (OI + mem_bytes)
    pred_combined_ms:  float
    err_combined_pct:  float
    r2_combined:       float
    formula:           str     # 선택된 회귀 형태


# ─────────────────────────────────────────────
# Main predictor
# ─────────────────────────────────────────────
def predict_model_latency(
    model:          nn.Module,
    inputs:         list,
    target_ops:     Optional[list] = None,
    precision:      str = "Float32",
    bench_csv:      str = BENCH_CSV,
    regression_csv: str = REGRESSION_CSV,
    combined_csv:   str = COMBINED_CSV,
    warmup:         int = 3,
    runs:           int = 10,
) -> List[LayerPrediction]:
    print("\n══ 회귀 모델 학습 ════════════════════════════════════════")
    oi_models       = load_oi_models(regression_csv, precision)
    mem_models      = load_mem_models(bench_csv, precision)
    combined_models = load_combined_models(combined_csv, precision)

    print("\n══ 모델 프로파일링 ═══════════════════════════════════════")
    profiler = ModelProfiler(target_ops=target_ops, warmup=warmup, runs=runs)
    profiler.profile(model, *inputs)

    def _err(pred, actual):
        return (pred - actual) / actual * 100 if actual > 1e-9 else 0.0

    results: List[LayerPrediction] = []
    for p in profiler.profiles:
        bench = op_to_bench(p)
        if bench is None:
            continue
        oi_reg   = oi_models.get(bench)
        mem_reg  = mem_models.get(bench)
        comb_reg = combined_models.get(bench)
        if oi_reg is None or mem_reg is None or comb_reg is None:
            continue

        pred_oi   = _predict_oi(p.macs, p.oi, oi_reg, bench)
        pred_mem  = _predict_mem(p.mem_bytes, mem_reg)
        pred_comb = _predict_combined(p.oi, p.mem_bytes, comb_reg, bench)

        results.append(LayerPrediction(
            layer_name=p.layer_name, op_type=p.op_type, bench_type=bench,
            macs=p.macs, mem_bytes=p.mem_bytes, oi_macs=p.oi,
            actual_lat_ms=p.latency_ms,
            pred_oi_ms=pred_oi,     err_oi_pct=_err(pred_oi,   p.latency_ms), r2_oi=oi_reg.r2,
            pred_mem_ms=pred_mem,   err_mem_pct=_err(pred_mem,  p.latency_ms), r2_mem=mem_reg.r2,
            pred_combined_ms=pred_comb, err_combined_pct=_err(pred_comb, p.latency_ms),
            r2_combined=comb_reg.r2,
            formula=comb_reg.formula,
        ))

    return results


# ─────────────────────────────────────────────
# Display & export
# ─────────────────────────────────────────────
def print_predictions(results: List[LayerPrediction]):
    W = 150
    print("\n" + "=" * W)
    print(f"  {'Layer':<34} {'Type':<10} {'Bench':<12} {'Actual':>9} "
          f"{'OI-Pred':>9} {'OI-Err%':>8} "
          f"{'Mem-Pred':>9} {'Mem-Err%':>9} "
          f"{'Comb-Pred':>10} {'Comb-Err%':>10}")
    print("-" * W)
    for p in results:
        print(f"  {p.layer_name:<34} {p.op_type:<10} {p.bench_type:<12} "
              f"{p.actual_lat_ms:>9.4f} "
              f"{p.pred_oi_ms:>9.4f} {p.err_oi_pct:>+8.2f}% "
              f"{p.pred_mem_ms:>9.4f} {p.err_mem_pct:>+9.2f}% "
              f"{p.pred_combined_ms:>10.4f} {p.err_combined_pct:>+10.2f}%")
    print("=" * W)
    total_actual = sum(p.actual_lat_ms     for p in results)
    total_oi     = sum(p.pred_oi_ms        for p in results)
    total_mem    = sum(p.pred_mem_ms       for p in results)
    total_comb   = sum(p.pred_combined_ms  for p in results)
    def _e(t): return (t - total_actual) / total_actual * 100 if total_actual else 0
    print(f"  Σ Actual    : {total_actual:.4f} ms")
    print(f"  Σ OI-Pred   : {total_oi:.4f} ms  (error: {_e(total_oi):+.2f}%)")
    print(f"  Σ Mem-Pred  : {total_mem:.4f} ms  (error: {_e(total_mem):+.2f}%)")
    print(f"  Σ Comb-Pred : {total_comb:.4f} ms  (error: {_e(total_comb):+.2f}%)")

    # 선택된 formula 분포
    from collections import Counter
    formula_dist = Counter(p.formula for p in results)
    print(f"  Formula 분포: { {k: v for k, v in formula_dist.items()} }")
    print("=" * W)

    # bench-type별 요약
    bench_types = sorted(set(p.bench_type for p in results))
    print(f"\n  {'Bench':<14} {'N':>4}  {'Actual':>10}  "
          f"{'OI-Pred':>9} {'OI-Err%':>8}  "
          f"{'Mem-Pred':>9} {'Mem-Err%':>9}  "
          f"{'Comb-Pred':>10} {'Comb-Err%':>10}")
    print("  " + "-" * 90)
    for bt in bench_types:
        sub   = [p for p in results if p.bench_type == bt]
        a     = sum(p.actual_lat_ms    for p in sub)
        poi   = sum(p.pred_oi_ms       for p in sub)
        pmem  = sum(p.pred_mem_ms      for p in sub)
        pcomb = sum(p.pred_combined_ms for p in sub)
        def _be(t): return (t - a) / a * 100 if a else 0
        print(f"  {bt:<14} {len(sub):>4}  {a:>10.4f}  "
              f"{poi:>9.4f} {_be(poi):>+8.2f}%  "
              f"{pmem:>9.4f} {_be(pmem):>+9.2f}%  "
              f"{pcomb:>10.4f} {_be(pcomb):>+10.2f}%")


def save_predictions_csv(results: List[LayerPrediction], path: str):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "layer_name", "op_type", "bench_type",
            "macs", "mem_bytes", "oi_macs",
            "actual_lat_ms",
            "pred_oi_ms",       "err_oi_pct",       "r2_oi",
            "pred_mem_ms",      "err_mem_pct",      "r2_mem",
            "pred_combined_ms", "err_combined_pct", "r2_combined",
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