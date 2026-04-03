# benchmark.py
import time
import torch
import psutil
import math
import torch.nn as nn
import pandas as pd
from sklearn.linear_model import LinearRegression

DEVICE = "cpu"
RUNS = 32
N_STEPS = 10       # 여기서는 scale용이지만, run_benchmark에서는 고정 크기만 사용
BYTES_PER = 4
REBUILD_MODEL_EVERY_RUN = False

RATIOS = {
    "conv":        [1],
    "dw":          [1],
    "gemm":        [1],
    "ln_attn_ln":  [1],
}

SIZES_FIT = {
    "gemm": dict(M=256,  K=256,  N=256),
    "conv": dict(B=1, Cin=6,  Cout=12,  H=224, W=224, K=3),
    "dw":   dict(B=1, C=8,  H=64,  W=64,  K=3),
    "vit":  dict(B=1, T=57,  H=4,  Dh=64),        # Attention용
    "ln_attn_ln": dict(B=1, T=160, H=4,  Dh=64),
}

SIZES_MISS = {
    "gemm": dict(M=1024, K=1024, N=1024),
    "conv": dict(B=1, Cin=24, Cout=48, H=224, W=224, K=3),
    "dw":   dict(B=1, C=96,  H=256, W=256, K=3),
    "vit":  dict(B=1, T=632, H=8,  Dh=64),        # Attention용
    "ln_attn_ln": dict(B=1, T=632, H=8,  Dh=64),
}

def flops_conv3x3(B,Cin,Cout,H,W,K=3): return 2*B*Cout*H*W*Cin*K*K
def bytes_conv3x3(B,Cin,Cout,H,W,K=3): return (B*Cin*H*W + B*Cout*H*W + Cout*Cin*K*K)*BYTES_PER
def flops_dw(B,C,H,W,K=3): return 2*B*C*H*W*K*K
def bytes_dw(B,C,H,W,K=3): return (B*C*H*W + B*C*H*W + C*K*K)*BYTES_PER
def flops_gemm(M,K,N): return 2*M*K*N
def bytes_gemm(M,K,N): return (M*K + K*N + 2*M*N)*BYTES_PER

def flops_conv3x3(B, Cin, Cout, H, W, K=3):
    return 2 * B * Cout * H * W * Cin * K * K

def bytes_conv3x3(B, Cin, Cout, H, W, K=3):
    return (B * Cin * H * W + B * Cout * H * W + Cout * Cin * K * K) * BYTES_PER

def flops_dw(B, C, H, W, K=3):
    return 2 * B * C * H * W * K * K

def bytes_dw(B, C, H, W, K=3):
    return (B * C * H * W + B * C * H * W + C * K * K) * BYTES_PER

def flops_gemm(M, K, N):
    return 2 * M * K * N

def bytes_gemm(M, K, N):
    return (M * K + K * N + 2 * M * N) * BYTES_PER

def flops_attn_total(B, T, H, Dh):
    D = H * Dh
    proj = 8.0 * B * T * D * D
    core = 4.0 * B * H * T * T * Dh
    return proj + core


def flops_bytes_layernorm(B, T, D):
    # FLOPs: ~5*D per token
    f = 5.0 * B * T * D
    # Bytes: 입력 + 출력 + (감마/베타 읽기) 대략
    b = ((B * T * D) + (B * T * D) + 2 * D) * BYTES_PER
    return f, b


def bytes_attn_total(B, T, H, Dh):
    D = H * Dh
    x_io  = (B * T * D + B * T * D)         # x, y
    qkv_o = (3 * B * T * D + B * T * D)     # q,k,v,o
    scores = (B * H * T * T) * 2           # score read/write (근사)
    return (x_io + qkv_o + scores) * BYTES_PER

@torch.no_grad()
def run_model(model, x, runs=RUNS):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model(x)
        # CUDA가 있다면 sync, CPU는 필요 없음
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times)


# ================ Conv 블록 ================

@torch.no_grad()
def bench_conv_block(B, Cin, Cout, H, W, K=3, runs=RUNS, device=DEVICE):
    ratios = RATIOS["conv"]

    def build_net():
        layers = []
        in_ch = Cin
        for _ in ratios:
            out_ch = max(1, int(Cout))
            layers += [
                nn.Conv2d(in_ch, out_ch, K, padding=1, bias=False),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        return nn.Sequential(*layers).to(device)

    if REBUILD_MODEL_EVERY_RUN:
        times = []
        for _ in range(runs):
            net = build_net()
            x = torch.randn(B, Cin, H, W, device=device)
            times.append(run_model(net, x, 1))
        avg = sum(times) / len(times)
    else:
        net = build_net()
        x = torch.randn(B, Cin, H, W, device=device)
        avg = run_model(net, x, runs)

    f = flops_conv3x3(B, Cin, Cout, H, W, K) * len(ratios)
    b = bytes_conv3x3(B, Cin, Cout, H, W, K) * len(ratios)
    g = f / avg / 1e9
    return avg, f, b, g


# ================ Depthwise 블록 ================

@torch.no_grad()
def bench_depthwise_block(B, C, H, W, K=3, runs=RUNS, device=DEVICE):
    ratios = RATIOS["dw"]

    def build_net():
        layers = []
        in_ch = C
        for _ in ratios:
            layers += [
                nn.Conv2d(in_ch, in_ch, K, padding=1, groups=in_ch, bias=False),
                nn.ReLU(inplace=True),
            ]
        return nn.Sequential(*layers).to(device)

    if REBUILD_MODEL_EVERY_RUN:
        times = []
        for _ in range(runs):
            net = build_net()
            x = torch.randn(B, C, H, W, device=device)
            times.append(run_model(net, x, 1))
        avg = sum(times) / len(times)
    else:
        net = build_net()
        x = torch.randn(B, C, H, W, device=device)
        avg = run_model(net, x, runs)

    f = flops_dw(B, C, H, W, K) * len(ratios)
    b = bytes_dw(B, C, H, W, K) * len(ratios)
    g = f / avg / 1e9
    return avg, f, b, g


# ================ GEMM 블록 ================

@torch.no_grad()
def bench_gemm_block(M, K, N, runs=RUNS, device=DEVICE):
    ratios = RATIOS["gemm"]

    def build_net():
        layers = []
        in_dim = K
        for r in ratios:
            out_dim = max(1, int(N * (0.8 + 0.1 * r)))  # 대략 0.9~1.3배
            layers += [
                nn.Linear(in_dim, out_dim, bias=True),
                nn.ReLU(inplace=True),
                nn.Softmax(dim=-1),
            ]
            in_dim = out_dim
        return nn.Sequential(*layers).to(device)

    if REBUILD_MODEL_EVERY_RUN:
        times = []
        for _ in range(runs):
            net = build_net()
            x = torch.randn(M, K, device=device)
            times.append(run_model(net, x, 1))
        avg = sum(times) / len(times)
    else:
        net = build_net()
        x = torch.randn(M, K, device=device)
        avg = run_model(net, x, runs)

    f = flops_gemm(M, K, N) * len(ratios)
    b = bytes_gemm(M, K, N) * len(ratios)
    g = f / avg / 1e9
    return avg, f, b, g


# ================ Self-Attention 블록 ================

class PureSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.h = n_head
        self.dh = d_model // n_head
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x: [B, T, D], D = H * Dh
        B, T, D = x.shape
        H, Dh = self.h, self.dh

        q = self.q(x).view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()
        k = self.k(x).view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()
        v = self.v(x).view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)   # [B,H,T,T]
        att = torch.softmax(scores, dim=-1)
        y = att @ v
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        y = self.o(y)
        return y

class LNAttnLNBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, eps: float = 1e-5):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=eps)
        self.attn = PureSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        x = self.ln1(x)
        x = self.attn(x)
        x = self.ln2(x)
        return x

@torch.no_grad()
def bench_ln_attn_ln_block(B, T, H, Dh, runs=RUNS, device=DEVICE):
    ratios = RATIOS["ln_attn_ln"]
    D = H * Dh

    def build_net():
        layers=[]
        for _ in ratios:
            layers += [LNAttnLNBlock(d_model=D, n_head=H)]
        return nn.Sequential(*layers).to(device)

    if REBUILD_MODEL_EVERY_RUN:
        times=[]
        for _ in range(runs):
            net = build_net()
            x = torch.randn(B, T, D, device=device)
            times.append(run_model(net, x, 1))
        avg = sum(times)/len(times)
    else:
        net = build_net()
        x = torch.randn(B, T, D, device=device)
        avg = run_model(net, x, runs)

    # FLOPs/Bytes: LN + Attn + LN 한 번을 묶어서 계산 후 ratios만큼 곱
    f_ln1, b_ln1 = flops_bytes_layernorm(B, T, D)
    f_attn       = flops_attn_total(B, T, H, Dh)
    b_attn       = bytes_attn_total(B, T, H, Dh)
    f_ln2, b_ln2 = flops_bytes_layernorm(B, T, D)

    f_one = f_ln1 + f_attn + f_ln2
    b_one = b_ln1 + b_attn + b_ln2

    f = f_one * len(ratios)
    b = b_one * len(ratios)
    g = f / avg / 1e9
    return avg, f, b, g
@torch.no_grad()
def bench_attention_block(B, T, H, Dh, runs=RUNS, device=DEVICE):
    ratios = RATIOS["attn"]
    D = H * Dh

    def build_net():
        layers = []
        for _ in ratios:
            layers += [
                PureSelfAttention(d_model=D, n_head=H),
                nn.LayerNorm(D),
                nn.ReLU(inplace=True),
            ]
        return nn.Sequential(*layers).to(device)

    if REBUILD_MODEL_EVERY_RUN:
        times = []
        for _ in range(runs):
            net = build_net()
            x = torch.randn(B, T, D, device=device)
            times.append(run_model(net, x, 1))
        avg = sum(times) / len(times)
    else:
        net = build_net()
        x = torch.randn(B, T, D, device=device)
        avg = run_model(net, x, runs)

    f = flops_attn_total(B, T, H, Dh) * len(ratios)
    b = bytes_attn_total(B, T, H, Dh) * len(ratios)
    g = f / avg / 1e9
    return avg, f, b, g

def bench_ln_attn_ln_block(B, T, H, Dh, runs=RUNS, device=DEVICE):
    ratios = RATIOS["ln_attn_ln"]
    D = H * Dh

    def build_net():
        layers=[]
        for _ in ratios:
            layers += [LNAttnLNBlock(d_model=D, n_head=H)]
        return nn.Sequential(*layers).to(device)

    if REBUILD_MODEL_EVERY_RUN:
        times=[]
        for _ in range(runs):
            net = build_net()
            x = torch.randn(B, T, D, device=device)
            times.append(run_model(net, x, 1))
        avg = sum(times)/len(times)
    else:
        net = build_net()
        x = torch.randn(B, T, D, device=device)
        avg = run_model(net, x, runs)

    # FLOPs/Bytes: LN + Attn + LN 한 번을 묶어서 계산 후 ratios만큼 곱
    f_ln1, b_ln1 = flops_bytes_layernorm(B, T, D)
    f_attn       = flops_attn_total(B, T, H, Dh)
    b_attn       = bytes_attn_total(B, T, H, Dh)
    f_ln2, b_ln2 = flops_bytes_layernorm(B, T, D)

    f_one = f_ln1 + f_attn + f_ln2
    b_one = b_ln1 + b_attn + b_ln2

    f = f_one * len(ratios)
    b = b_one * len(ratios)
    g = f / avg / 1e9
    return avg, f, b, g

def run_model(model,x,runs=RUNS):
    times=[]
    for _ in range(runs):
        t0=time.perf_counter(); _=model(x); t1=time.perf_counter()
        times.append(t1-t0)
    return sum(times)/len(times)
# --- 기존 Conv+BN+ReLU 블록 대신 단일 Conv 레이어 ---

class ConvOnlyBlock(nn.Module):
    def __init__(self, Cin, Cout, K=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(Cin, Cout, kernel_size=K, stride=stride, padding=padding, bias=False)

    def forward(self, x):
        return self.conv(x)
def scale_linear(v_fit,v_miss,step,steps):
    return int(round(v_fit+(v_miss-v_fit)*(step/(steps-1))))
def interpolate_cfg(cfg_fit,cfg_miss,step,steps):
    return {k:scale_linear(cfg_fit[k],cfg_miss[k],step,steps) for k in cfg_fit}
def fit_linear_oi_from_df(df):
    """
    필요한 열 후보:
      - throughput: 'through' 또는 'thr' 포함
      - flops: 'flop' 포함 (GFLOPs 단위 가정)
      - memory: 'mem' 또는 'gb' 포함 (GB 단위 가정)
    """
    cols = list(df.columns)

    def has_sub(col, key):
        return key in col.lower()

    possible_throughput = [c for c in cols if has_sub(c, "through") or has_sub(c, "thr")]
    possible_flops      = [c for c in cols if has_sub(c, "flop")]
    possible_mem        = [c for c in cols if has_sub(c, "mem") or has_sub(c, "gb")]

    if not (possible_throughput and possible_flops and possible_mem):
        # 어떤 열이든 하나라도 못 찾으면 바로 종료
        return None, 0, df

    thr_col, flop_col, mem_col = possible_throughput[0], possible_flops[0], possible_mem[0]

    df = df[[flop_col, mem_col, thr_col]].rename(
        columns={flop_col: "GFLOPs", mem_col: "GB", thr_col: "thr"}
    )
    df = df.dropna()
    df = df[(df["GB"] > 0) & (df["GFLOPs"] > 0) & (df["thr"] > 0)].copy()
    if len(df) < 2:
        return None, len(df), df

    # OI = GFLOPs / GB
    df["OI"] = df["GFLOPs"] / df["GB"]

    from sklearn.linear_model import LinearRegression
    X = df[["OI"]].to_numpy()
    y = df["thr"].to_numpy()
    model = LinearRegression().fit(X, y)
    r2 = model.score(X, y)
    return (model.intercept_, model.coef_[0], r2), len(df), df

def run_scaled_suite():
    ops = [
        ("GEMM",       bench_gemm_block,       "gemm"),
        ("CONV",       bench_conv_block,       "conv"),
        ("DEPTH",      bench_depthwise_block,  "dw"),
        ("LN-ATTN-LN", bench_ln_attn_ln_block, "ln_attn_ln"),
    ]
    print(f"REBUILD_MODEL_EVERY_RUN={REBUILD_MODEL_EVERY_RUN}")

    # 각 연산별 요약(regression 포함)을 모아둘 수 있음 (원하면 run_benchmark에서 활용)
    summaries = {}

    for name, fn, key in ops:
        records = []  # 이 연산에 대해 step별 결과 기록
        print(f"\n=== {name} ===")
        print("op     step      GF/s      GFLOPs        GB      latency(ms)")

        for step in range(N_STEPS):
            cfg = interpolate_cfg(SIZES_FIT[key], SIZES_MISS[key], step, N_STEPS)
            avg, f, b, g = fn(**cfg)
            gflops = f / 1e9
            gbytes = b / 1e9
            lat_ms = avg * 1000.0

            # 기존 라인 출력
            print(f"{name:<9}{step+1:<5}{g:10.2f}{gflops:10.3f}{gbytes:10.3f}{lat_ms:12.2f}")

            # 회귀용 기록 (GFLOPs, GB, throughput)
            records.append({
                "GFLOPs": gflops,
                "GB": gbytes,
                "thr": g,          # throughput = GF/s
            })

        # --- 여기서 이 연산에 대한 선형 회귀 수행 ---
        df = pd.DataFrame(records)
        coef, n_used, df_used = fit_linear_oi_from_df(df)
        if coef is not None:
            alpha, beta, r2 = coef
            print(f"[REG] {name}: thr = {alpha:.4f} + {beta:.44f} * OI "
                  f"(R²={r2:.4f}, n={n_used})")
            summaries[name] = {
                "alpha": alpha,
                "beta": beta,
                "r2": r2,
            }
        else:
            print(f"[REG] {name}: 회귀에 사용할 유효 데이터가 부족합니다. (n={n_used})")
            summaries[name] = None

    return summaries

def run_benchmark():
    summaries = run_scaled_suite()
    print("\n===== Linear Regression per Operation (thr vs OI) =====")
    for name, reg in summaries.items():
        if reg is None:
            print(f"{name:9s}: (no regression, not enough data)")
        else:
            alpha = reg["alpha"]
            beta  = reg["beta"]
            r2    = reg["r2"]
            print(f"{name:9s}: thr = {alpha:.4f} + {beta:.4f} * OI (R²={r2:.4f})")
    return summaries

def get_resource():
    """
    C++ benchmark::resource::getResource에 해당하지만,
    /proc, perf_event 대신 psutil 기반으로 구현.
    반환: cpu_usage(0~1), available_mem(MB), battery(%)
    """
    # CPU 사용률 (약간의 interval 줌)
    cpu_percent = psutil.cpu_percent(interval=0.5)
    cpu_usage = cpu_percent / 100.0

    vm = psutil.virtual_memory()
    available_mb = vm.available / (1024 ** 2)

    try:
        batt = psutil.sensors_battery()
        battery = batt.percent if batt is not None else 100.0
    except Exception:
        battery = 100.0

    return cpu_usage, available_mb, battery
