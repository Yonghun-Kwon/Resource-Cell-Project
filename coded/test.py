# -*- coding: utf-8 -*-
"""
5입력->5타깃 MLP 학습 후, 학습 데이터에 대한
타깃값과 예측값을 함께(나란히) 출력 및 CSV 저장.
"""

import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

# ===== 경로 =====
DATA_PATH = Path("oi_rpi5.csv")          # 필요 시 경로 수정
OUT_CSV   = Path("./train_pred_with_targets.csv")   # 저장 파일명

# ===== 1) 데이터 로드 =====
df = pd.read_csv(DATA_PATH)

# ===== 2) 5입력/5타깃 컬럼 자동 감지 =====
def detect_io_columns(df: pd.DataFrame):
    cats = ["depth", "gemm", "conv", "pointwise", "elementwise"]
    # 숫자형만 추림
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    # pred_*/meas_* 패턴 우선
    pred_cols = [c for c in num_cols if re.match(r"(?i)^pred[_\\-\\s]?", c)]
    meas_cols = [c for c in num_cols if re.match(r"(?i)^meas[_\\-\\s]?", c)]

    def order_by_cats(cols, prefix):
        lower = [c.lower() for c in cols]
        chosen = []
        for k in cats:
            cand = None
            # exact: prefix_cat
            for i, lc in enumerate(lower):
                if lc == f"{prefix}_{k}":
                    cand = cols[i]; break
            # fallback: 포함 관계
            if cand is None:
                for i, lc in enumerate(lower):
                    if (prefix in lc) and (k in lc):
                        cand = cols[i]; break
            if cand is not None and cand not in chosen:
                chosen.append(cand)
        # 5개 못 채우면 앞에서부터 보충
        if len(chosen) < 5:
            for c in cols:
                if c not in chosen:
                    chosen.append(c)
                if len(chosen) == 5:
                    break
        return chosen[:5]

    if len(pred_cols) >= 5 and len(meas_cols) >= 5:
        X_cols = order_by_cats(pred_cols, "pred")
        Y_cols = order_by_cats(meas_cols, "meas")
    else:
        # 차선: 숫자형 앞 10개 → 5입력/5타깃
        if len(num_cols) < 10:
            raise RuntimeError("숫자형 컬럼이 10개 미만이라 5입력/5타깃 자동 감지 실패")
        X_cols = num_cols[:5]
        Y_cols = num_cols[5:10]

    if len(X_cols) != 5 or len(Y_cols) != 5:
        raise RuntimeError("입력/타깃 컬럼 수가 5개가 아닙니다.")
    return X_cols, Y_cols

X_cols, Y_cols = detect_io_columns(df)

# ===== 3) 전처리 =====
work = df.copy()
for c in X_cols + Y_cols:
    work[c] = pd.to_numeric(work[c], errors="coerce")
work = work.dropna(subset=X_cols + Y_cols).reset_index(drop=True)

X = work[X_cols].to_numpy(dtype=float)
Y = work[Y_cols].to_numpy(dtype=float)

# ===== 4) MLP 구성(새 초기화) & 학습 =====
mlp = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    solver="adam",
    alpha=1e-3,
    learning_rate="adaptive",
    learning_rate_init=1e-3,
    max_iter=8000,
    early_stopping=True,
    validation_fraction=0.2,
    n_iter_no_change=80,
    tol=1e-4,
    random_state=None,  # 매 실행 새 초기화
    shuffle=True,
)

model = Pipeline([
    ("x_scaler", StandardScaler()),
    ("reg", TransformedTargetRegressor(
        regressor=mlp,
        transformer=StandardScaler()  # 다중 타깃 표준화
    ))
])

model.fit(X, Y)

# ===== 5) 학습 데이터로 다시 예측 =====
Y_hat = model.predict(X)  # shape (N, 5)

# ===== 6) 타깃값과 예측값을 나란히 출력 =====
#    (열 순서: target1, yhat1, target2, yhat2, ...)
pairs = []
for j, tgt_col in enumerate(Y_cols):
    pairs += [tgt_col, f"yhat_{tgt_col}"]

out_df = pd.DataFrame(Y_hat, columns=[f"yhat_{c}" for c in Y_cols])
show_df = pd.concat([work[Y_cols].reset_index(drop=True), out_df], axis=1)[pairs]

# 콘솔 출력(헤더 1회)
print("\n=== 학습 데이터에 대한 타깃값 & 예측값 (전 행) ===")
print(show_df.to_string(index=False))

# 원하면 입력/타깃/예측/오차까지 모두 저장
full_save = work.copy()
for j, tgt_col in enumerate(Y_cols):
    full_save[f"yhat_{tgt_col}"] = Y_hat[:, j]
    full_save[f"abs_err_{tgt_col}"] = np.abs(full_save[tgt_col] - full_save[f"yhat_{tgt_col}"])
full_save.to_csv(OUT_CSV, index=False)
print(f"\nSaved with targets & predictions: {OUT_CSV}")
