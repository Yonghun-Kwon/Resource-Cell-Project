#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os
import matplotlib.pyplot as plt

FILES = {
    "rpi3_conv": "rpi3_conv.csv",
    "rpi4_conv": "rpi4_conv.csv",
    "rpi5_conv": "rpi5_conv.csv",
    "rpi3_dw":   "rpi3_depth.csv",
    "rpi4_dw":   "rpi4_depth.csv",
    "rpi5_dw":   "rpi5_depth.csv",
    "rpi3_gemm": "rpi3_gemm.csv",
    "rpi4_gemm": "rpi4_gemm.csv",
    "rpi5_gemm": "rpi5_gemm.csv",
    "rpi3_att": "rpi3_att.csv",
    "rpi4_att": "rpi4_att.csv",
    "rpi5_att": "rpi5_att.csv",
}
def load_data(path):
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".csv":
        df = pd.read_csv(path)
    elif ext in [".xls", ".xlsx"]:
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
    df.columns = [c.strip().lower() for c in df.columns]
    return df

def fit_linear_oi_from_df(df):

    possible_throughput = [c for c in df.columns if "through" in c or "thr" in c]
    possible_flops      = [c for c in df.columns if "flop" in c]
    possible_mem        = [c for c in df.columns if "mem" in c or "gb" in c]
    if not (possible_throughput and possible_flops and possible_mem):
        return None, 0, df

    thr_col, flop_col, mem_col = possible_throughput[0], possible_flops[0], possible_mem[0]
    df = df[[flop_col, mem_col, thr_col]].rename(
        columns={flop_col: "GFLOPs", mem_col: "GB", thr_col: "thr"}
    )
    df = df.dropna()
    df = df[(df["GB"] > 0) & (df["GFLOPs"] > 0) & (df["thr"] > 0)].copy()
    if len(df) < 2:
        return None, len(df), df

    df["OI"] = df["GFLOPs"] / df["GB"]

    X = df[["OI"]].to_numpy()
    y = df["thr"].to_numpy()
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)

    return (model.intercept_, model.coef_[0], r2), len(df), df

OI_POINTS = np.array([1.0, 5.0, 10.0])

rows = []
results = []
for key, path in FILES.items():
    if not os.path.exists(path):
        print(f"[warning] no files: {path}")
        continue

    df = load_data(path)
    fit, n, df_clean = fit_linear_oi_from_df(df)
    if fit is None:
        results.append((key, np.nan, np.nan, np.nan, n))
        continue

    a, b, r2 = fit
    results.append((key, a, b, r2, n))
    thr_pred = a + b * OI_POINTS

    rows.append({
        "device_op": key,
        f"thr@OI={OI_POINTS[0]}": float(thr_pred[0]),
        f"thr@OI={OI_POINTS[1]}": float(thr_pred[1]),
        f"thr@OI={OI_POINTS[2]}": float(thr_pred[2]),
    })

pred_df = pd.DataFrame(rows)
if not pred_df.empty:
    print("\n=== Predicted Throughput at Selected OI (GFLOP/s) ===")
    print(pred_df.sort_values("device_op").to_string(index=False))

out = pd.DataFrame(results, columns=["device_op","intercept","slope","R2","n_used"])
if not out.empty:
    print("\n=== Regression Summary (x = OI) ===")
    print(out.sort_values("device_op").round(4).to_string(index=False))