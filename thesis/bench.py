#!/usr/bin/env python3
import torch, time
import torch.nn as nn
import math
DEVICE = "cpu"
RUNS = 128
N_STEPS = 10
BYTES_PER = 4
REBUILD_MODEL_EVERY_RUN = True

RATIOS = {
    "conv":       1,
    "dw":         1,
    "gemm":       1,
    "ln_attn_ln": 1,
}


SIZES_FIT = {
    "gemm": dict(M=256, K=256, N=256),
    "conv": dict(B=1, Cin=6,  Cout=12,  H=224,  W=224,  K=3),
    "dw":   dict(B=1, C=8,    H=64,  W=64,  K=3),
    "ln_attn_ln": dict(B=1, T=160,  H=4,  Dh=64),
}

SIZES_MISS = {
    "gemm": dict(M=1024, K=1024, N=1024),
    "conv": dict(B=1, Cin=24, Cout=48, H=224, W=224, K=3),
    "dw":   dict(B=1, C=96,  H=256, W=256, K=3),
    "ln_attn_ln": dict(B=1, T=632, H=8,  Dh=64),
}

# -------------------------------------------
# FLOPs / Bytes 
# -------------------------------------------
def flops_conv3x3(B,Cin,Cout,H,W,K=3): return 2*B*Cout*H*W*Cin*K*K
def bytes_conv3x3(B,Cin,Cout,H,W,K=3): return (B*Cin*H*W + B*Cout*H*W + Cout*Cin*K*K)*BYTES_PER
def flops_dw(B,C,H,W,K=3): return 2*B*C*H*W*K*K
def bytes_dw(B,C,H,W,K=3): return (B*C*H*W + B*C*H*W + C*K*K)*BYTES_PER
def flops_gemm(M,K,N): return 2*M*K*N
def bytes_gemm(M,K,N): return (M*K + K*N + 2*M*N)*BYTES_PER

def flops_attn_total(B,T,H,Dh):
    D = H*Dh
    proj = 8.0 * B * T * D * D
    core = 4.0 * B * H * T * T * Dh
    return proj + core

def flops_bytes_layernorm(B, T, D):
    f = 5.0 * B * T * D
    b = ((B*T*D) + (B*T*D) + 2*D) * BYTES_PER
    return f, b

def bytes_attn_total(B, T, H, Dh):
    D = H * Dh
    x_io  = (B*T*D + B*T*D)
    qkv_o = (3*B*T*D + B*T*D)
    scores = (B*H*T*T) * 2
    return (x_io + qkv_o + scores) * BYTES_PER

def run_model(model, x, runs=RUNS):
    times = []
    for _ in range(runs):
        t0 = time.perf_counter(); _ = model(x); t1 = time.perf_counter()
        times.append(t1 - t0)
    return sum(times) / len(times)

# -------------------------------------------
# Pure Self-Attention (Q,K,V,O 분리형)
# -------------------------------------------
class PureSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.h  = n_head
        self.dh = d_model // n_head
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        H, Dh = self.h, self.dh
        q = self.q(x).view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()
        k = self.k(x).view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()
        v = self.v(x).view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)
        att = torch.softmax(scores, dim=-1)
        y = att @ v
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        return self.o(y)

# -------------------------------------------
# LN → Attn → LN 블록
# -------------------------------------------
class LNAttnLNBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, eps: float = 1e-5):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=eps)
        self.attn = PureSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        return self.ln2(self.attn(self.ln1(x)))

# -------------------------------------------
# Conv 블록
# -------------------------------------------
@torch.no_grad()
def bench_conv_block(B,Cin,Cout,H,W,K=3,runs=RUNS,device=DEVICE):
    n_layers = RATIOS["conv"]
    if REBUILD_MODEL_EVERY_RUN:
        times=[]
        for _ in range(runs):
            layers=[]; in_ch = Cin
            for _ in range(n_layers):
                out_ch = max(1, int(Cout))
                layers += [
                    nn.Conv2d(in_ch, out_ch, K, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                ]
                in_ch = out_ch
            layers += [nn.AdaptiveAvgPool2d(1)]
            net=nn.Sequential(*layers).to(device)
            x=torch.randn(B,Cin,H,W,device=device)
            t0=time.perf_counter(); _=net(x); t1=time.perf_counter()
            times.append(t1-t0)
        avg=sum(times)/len(times)
    else:
        layers=[]; in_ch = Cin
        for _ in range(n_layers):
            out_ch = max(1, int(Cout))
            layers += [
                nn.Conv2d(in_ch, out_ch, K, padding=1, bias=False),
                nn.ReLU(inplace=True)
            ]
            in_ch = out_ch
        net=nn.Sequential(*layers).to(device)
        x=torch.randn(B,Cin,H,W,device=device)
        avg=run_model(net,x,runs)

    f=flops_conv3x3(B,Cin,Cout,H,W,K)*n_layers
    b=bytes_conv3x3(B,Cin,Cout,H,W,K)*n_layers
    g=f/avg/1e9
    return avg,f,b,g

# -------------------------------------------
# Depthwise 블록
# -------------------------------------------
@torch.no_grad()
def bench_depthwise_block(B,C,H,W,K=3,runs=RUNS,device=DEVICE):
    n_layers = RATIOS["dw"]
    def build_net():
        layers=[]
        for _ in range(n_layers):
            layers += [
                nn.Conv2d(C, C, K, padding=1, groups=C, bias=False),
                nn.ReLU(inplace=True)
            ]
        return nn.Sequential(*layers).to(device)

    if REBUILD_MODEL_EVERY_RUN:
        times=[]
        for _ in range(runs):
            net=build_net()
            x=torch.randn(B,C,H,W,device=device)
            t0=time.perf_counter(); _=net(x); t1=time.perf_counter()
            times.append(t1-t0)
        avg=sum(times)/len(times)
    else:
        net=build_net()
        x=torch.randn(B,C,H,W,device=device)
        avg=run_model(net,x,runs)

    f=flops_dw(B,C,H,W,K)*n_layers
    b=bytes_dw(B,C,H,W,K)*n_layers
    g=f/avg/1e9
    return avg,f,b,g

# -------------------------------------------
# GEMM 블록
# -------------------------------------------
@torch.no_grad()
def bench_gemm_block(M,K,N,runs=RUNS,device=DEVICE):
    n_layers = RATIOS["gemm"]
    def build_net():
        layers=[]; in_dim = K
        for _ in range(n_layers):
            out_dim = max(1, int(N))
            layers += [nn.Linear(in_dim, out_dim, bias=True),
                       nn.ReLU(inplace=True),
                       nn.Softmax(dim=-1)]
            in_dim = out_dim
        return nn.Sequential(*layers).to(device)

    if REBUILD_MODEL_EVERY_RUN:
        times=[]
        for _ in range(runs):
            net=build_net()
            x=torch.randn(M,K,device=device)
            t0=time.perf_counter(); _=net(x); t1=time.perf_counter()
            times.append(t1-t0)
        avg=sum(times)/len(times)
    else:
        net=build_net()
        x=torch.randn(M,K,device=device)
        avg=run_model(net,x,runs)

    f=flops_gemm(M,K,N)*n_layers
    b=bytes_gemm(M,K,N)*n_layers
    g=f/avg/1e9
    return avg,f,b,g

# -------------------------------------------
# LN-ATTN-LN 블록
# -------------------------------------------
@torch.no_grad()
def bench_ln_attn_ln_block(B, T, H, Dh, runs=RUNS, device=DEVICE):
    n_layers = RATIOS["ln_attn_ln"]
    D = H * Dh

    def build_net():
        return nn.Sequential(*[LNAttnLNBlock(d_model=D, n_head=H) for _ in range(n_layers)]).to(device)

    if REBUILD_MODEL_EVERY_RUN:
        times=[]
        for _ in range(runs):
            net = build_net()
            x = torch.randn(B, T, D, device=device)
            t0=time.perf_counter(); _=net(x); t1=time.perf_counter()
            times.append(t1-t0)
        avg = sum(times)/len(times)
    else:
        net = build_net()
        x = torch.randn(B, T, D, device=device)
        avg = run_model(net, x, runs)

    f_ln1, b_ln1 = flops_bytes_layernorm(B, T, D)
    f_attn        = flops_attn_total(B, T, H, Dh)
    b_attn        = bytes_attn_total(B, T, H, Dh)
    f_ln2, b_ln2  = flops_bytes_layernorm(B, T, D)

    f = (f_ln1 + f_attn + f_ln2) * n_layers
    b = (b_ln1 + b_attn + b_ln2) * n_layers
    g = f / avg / 1e9
    return avg, f, b, g

# -------------------------------------------
# 스케일 보간 & 실행
# -------------------------------------------
def scale_linear(v_fit,v_miss,step,steps):
    return int(round(v_fit+(v_miss-v_fit)*(step/(steps-1))))
def interpolate_cfg(cfg_fit,cfg_miss,step,steps):
    return {k:scale_linear(cfg_fit[k],cfg_miss[k],step,steps) for k in cfg_fit}

def run_scaled_suite():
    ops = [
        ("DEPTH",      bench_depthwise_block,      "dw"),
        ("CONV",       bench_conv_block,            "conv"),
        ("GEMM",       bench_gemm_block,            "gemm"),
        ("LN-ATTN-LN", bench_ln_attn_ln_block,      "ln_attn_ln"),
    ]
    print(f"REBUILD_MODEL_EVERY_RUN={REBUILD_MODEL_EVERY_RUN}")
    print(f"{'name':<9}{'step':<5}{'GFLOP/s':>10}{'GFLOPs':>10}{'GBytes':>10}{'avg_ms':>12}")
    for name, fn, key in ops:
        for step in range(N_STEPS):
            cfg = interpolate_cfg(SIZES_FIT[key], SIZES_MISS[key], step, N_STEPS)
            avg, f, b, g = fn(**cfg)
            print(f"{name:<9}{step+1:<5}{g:10.2f}{f/1e9:10.3f}{b/1e9:10.3f}{avg*1000:12.2f}")

if __name__ == "__main__":
    run_scaled_suite()