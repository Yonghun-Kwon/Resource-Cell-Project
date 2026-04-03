#!/usr/bin/env python3
import torch, time
import torch.nn as nn
import math
import numpy as np
DEVICE = "cpu"
RUNS = 128
N_STEPS = 10
BYTES_PER = 4
REBUILD_MODEL_EVERY_RUN = True

# -------------------------------------------
# 계단형 비율(층 수 제어)
# -------------------------------------------
RATIOS = {
    "conv":  [1],
    "dw":    [1],
    "gemm":  [1],
    "attn":  [1],   # ← 어텐션 계단 수
    "ln_attn_ln": [1],
}

# -------------------------------------------
# ★ 크기 설정 (Fit~Miss)
# - CONV: 약간 축소
# - GEMM: 확대
# - DEPTHWISE: 확대
# - ATTN: T/H/Dh 스윕으로 T^2 효과 확인
# -------------------------------------------
SIZES_FIT = {
    "gemm": dict(M=256, K=256, N=256),
    "conv": dict(B=1, Cin=6,  Cout=12,  H=224,  W=224,  K=3),
    "dw":   dict(B=1, C=8,    H=64,  W=64,  K=3),
    "vit": dict(B=1, T=57,  H=4,  Dh=64),  
    # ATTN ≈ GEMM(FIT) 맞춤: B=1, H=4, Dh=64, T=57  → D=256
    # F_attn = 8*1*57*256^2 + 4*1*4*57^2*64 = 33,211,392  (GEMM 대비 -1.02%)
    "ln_attn_ln": dict(B=1, T=160,  H=4,  Dh=64),   # D = 256
}

SIZES_MISS = {
    "gemm": dict(M=1024, K=1024, N=1024),
    "conv": dict(B=1, Cin=24, Cout=48, H=224, W=224, K=3),
    "dw":   dict(B=1, C=96,  H=256, W=256, K=3),
    "vit": dict(B=1, T=632, H=8,  Dh=64),
    # ATTN ≈ GEMM(MISS) 맞춤: B=1, H=8, Dh=64, T=632 → D=512
    # F_attn = 8*1*632*512^2 + 4*1*8*632^2*64 = 2,143,420,416 (GEMM 대비 -0.19%)
    "ln_attn_ln": dict(B=1, T=632, H=8,  Dh=64),   # D = 512
}
# -------------------------------------------
# FLOPs / Bytes 유틸
# -------------------------------------------
def flops_conv3x3(B,Cin,Cout,H,W,K=3): return 2*B*Cout*H*W*Cin*K*K
def bytes_conv3x3(B,Cin,Cout,H,W,K=3): return (B*Cin*H*W + B*Cout*H*W + Cout*Cin*K*K)*BYTES_PER
def flops_dw(B,C,H,W,K=3): return 2*B*C*H*W*K*K
def bytes_dw(B,C,H,W,K=3): return (B*C*H*W + B*C*H*W + C*K*K)*BYTES_PER
def flops_gemm(M,K,N): return 2*M*K*N
def bytes_gemm(M,K,N): return (M*K + K*N + 2*M*N)*BYTES_PER

# --- Attention FLOPs/Bytes (근사) ---
# 총 FLOPs ≈ 8·B·T·D^2 (Q,K,V,O proj) + 4·B·H·T^2·Dh (QK^T + A·V)
def flops_attn_total(B,T,H,Dh):
    D = H*Dh
    proj = 8.0 * B * T * D * D
    core = 4.0 * B * H * T * T * Dh
    return proj + core


def flops_bytes_layernorm(B, T, D):
    # FLOPs: ~5*D per token (mean, var, normalize, scale, bias)
    f = 5.0 * B * T * D
    # Bytes: 입력 + 출력 + (감마/베타 읽기) 대략
    b = ((B*T*D) + (B*T*D) + 2*D) * BYTES_PER
    return f, b


# 총 Bytes(대략) ≈ (입출력/투영/스코어) 합산
def bytes_attn_total(B, T, H, Dh):
    D = H * Dh
    x_io  = (B*T*D + B*T*D)           # x, y
    qkv_o = (3*B*T*D + B*T*D)         # q,k,v,o
    scores = (B*H*T*T) * 2            # score read/write (근사)
    return (x_io + qkv_o + scores) * BYTES_PER

def run_model(model, x, runs=RUNS):
    times=[]
    for _ in range(runs):
        t0=time.perf_counter(); _=model(x); t1=time.perf_counter()
        times.append(t1-t0)
    return sum(times)/len(times)

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
        # x: [B, T, D], D = H * Dh
        B, T, D = x.shape
        H, Dh = self.h, self.dh

        # 1) Q/K/V 투사
        q = self.q(x).view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()  # [B,H,T,Dh]
        k = self.k(x).view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()
        v = self.v(x).view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()

        # 2) 점수 = QK^T / sqrt(Dh) → softmax
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)                 # [B,H,T,T]
        att = torch.softmax(scores, dim=-1)

        # 3) A·V → [B,H,T,Dh]
        y = att @ v

        # 4) 헤드 결합 → 출력 투사
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, D)               # [B,T,D]
        y = self.o(y)
        return y

# -------------------------------------------
# LN → Attn → LN 묶음 벤치
# -------------------------------------------
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
    
@torch.no_grad()
def bench_conv(B=1, Cin=3, Cout=8, H=64, W=64, K=3, runs=RUNS, device=DEVICE):
    # ---- 모델 구성 (단일 Conv 레이어) ----
    net = nn.Conv2d(Cin, Cout, K, padding=1, bias=False).to(device)
    x = torch.randn(B, Cin, H, W, device=device)
    torch.set_grad_enabled(False)

    # ---- 반복 실행 ----
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = net(x)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()
        times.append(t1 - t0)
    avg = sum(times) / len(times)  # 평균 초 단위

    # ---- FLOPs / Bytes / GF/s 계산 ----
    f = 2 * B * Cin * Cout * K * K * H * W        # FLOPs
    b = (B * Cin * H * W + B * Cout * H * W) * 4  # Bytes (float32)
    g = f / avg / 1e9                             # GFLOP/s

  

    return avg, f, b, g

# -------------------------------------------
# Conv 블록
# -------------------------------------------
@torch.no_grad()
def bench_conv_block(B,Cin,Cout,H,W,K=3,runs=RUNS,device=DEVICE):
    ratios = RATIOS["conv"]
    if REBUILD_MODEL_EVERY_RUN:
        times=[]
        for _ in range(runs):
            layers=[]; in_ch = Cin
            for _ in ratios:
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
            times.append(run_model(net,x,1))
        avg=sum(times)/len(times)
    else:
        layers=[]; in_ch = Cin
        for _ in ratios:
            out_ch = max(1, int(Cout))
            layers += [
                nn.Conv2d(in_ch, out_ch, K, padding=1, bias=False),
                #nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            in_ch = out_ch
        #layers += [nn.AdaptiveAvgPool2d(1)]
        net=nn.Sequential(*layers).to(device)
        x=torch.randn(B,Cin,H,W,device=device)
        avg=run_model(net,x,runs)

    f=flops_conv3x3(B,Cin,Cout,H,W,K)*len(ratios)
    b=bytes_conv3x3(B,Cin,Cout,H,W,K)*len(ratios)
    g=f/avg/1e9
    return avg,f,b,g

# -------------------------------------------
# Depthwise 블록
# -------------------------------------------
@torch.no_grad()
def bench_depthwise_block(B,C,H,W,K=3,runs=RUNS,device=DEVICE):
    ratios = RATIOS["dw"]
    def build_net():
        layers=[]; in_ch = C
        for _ in ratios:
            layers += [
                nn.Conv2d(in_ch, in_ch, K, padding=1, groups=in_ch, bias=False),
                #nn.BatchNorm2d(in_ch),
                nn.ReLU(inplace=True)
            ]
        return nn.Sequential(*layers).to(device)

    if REBUILD_MODEL_EVERY_RUN:
        times=[]
        for _ in range(runs):
            net=build_net()
            x=torch.randn(B,C,H,W,device=device)
            times.append(run_model(net,x,1))
        avg=sum(times)/len(times)
    else:
        net=build_net()
        x=torch.randn(B,C,H,W,device=device)
        avg=run_model(net,x,runs)

    f=flops_dw(B,C,H,W,K)*len(ratios)
    b=bytes_dw(B,C,H,W,K)*len(ratios)
    g=f/avg/1e9
    return avg,f,b,g

# -------------------------------------------
# GEMM 블록
# -------------------------------------------
@torch.no_grad()
def bench_gemm_block(M,K,N,runs=RUNS,device=DEVICE):
    ratios = RATIOS["gemm"]
    def build_net():
        layers=[]; in_dim = K
        for r in ratios:
            out_dim = max(1, int(N * (0.8 + 0.1*r)))  # 0.9~1.3배
            layers += [nn.Linear(in_dim, out_dim, bias=True,),
                       nn.ReLU(inplace=True),
                       nn.Softmax(dim=-1)]
            in_dim = out_dim
        return nn.Sequential(*layers).to(device)

    if REBUILD_MODEL_EVERY_RUN:
        times=[]
        for _ in range(runs):
            net=build_net()
            x=torch.randn(M,K,device=device)
            times.append(run_model(net,x,1))
        avg=sum(times)/len(times)
    else:
        net=build_net()
        x=torch.randn(M,K,device=device)
        avg=run_model(net,x,runs)

    f=flops_gemm(M,K,N)*len(ratios)
    b=bytes_gemm(M,K,N)*len(ratios)
    g=f/avg/1e9
    return avg,f,b,g

# -------------------------------------------
# ATTENTION 블록 (QKV fused + SDPA + out proj)
# -------------------------------------------
class PureSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.h  = n_head
        self.dh = d_model // n_head
        # Q, K, V, O 각각 분리된 Linear (bias=False로 통일)
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x: [B, T, D], D = H * Dh
        B, T, D = x.shape
        H, Dh = self.h, self.dh

        # 1) Q/K/V 투사
        q = self.q(x).view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()  # [B,H,T,Dh]
        k = self.k(x).view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()
        v = self.v(x).view(B, T, H, Dh).permute(0, 2, 1, 3).contiguous()

        # 2) 점수 = QK^T / sqrt(Dh)  → [B,H,T,T]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(Dh)

        # (옵션) 마스크가 필요하면 여기에 causal/패딩 마스크 적용 가능
        # mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        # scores = scores.masked_fill(mask, float('-inf'))

        # 3) softmax
        att = torch.softmax(scores, dim=-1)

        # 4) A·V → [B,H,T,Dh]
        y = att @ v

        # 5) 헤드 결합 → 출력 투사
        y = y.permute(0, 2, 1, 3).contiguous().view(B, T, D)  # [B,T,D]
        y = self.o(y)
       
        return y
    
@torch.no_grad()
def bench_attention_block(B, T, H, Dh, runs=RUNS, device=DEVICE):
    ratios = RATIOS["attn"]
    D = H * Dh

    def build_net():
        layers = []
        for _ in ratios:
            # 여기서 PureSelfAttention 사용!
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

    # FLOPs/Bytes 근사식은 기존 함수 사용 (Q/K/V/O + QK^T + A·V)
    f = flops_attn_total(B, T, H, Dh) * len(ratios)
    b = bytes_attn_total(B, T, H, Dh) * len(ratios)
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
    ops=[
        #("DEPTH",      bench_depthwise_block,      "dw"),
        #("CONV",      bench_conv_block,      "conv"),
        ("GEMM",      bench_gemm_block,      "gemm"),
        #("ATTN",      bench_attention_block, "attn"),
        #("LN-ATTN-LN", bench_ln_attn_ln_block, "ln_attn_ln"),   # ★ 추가
    ]
    print(f"REBUILD_MODEL_EVERY_RUN={REBUILD_MODEL_EVERY_RUN}")
    for name,fn,key in ops:
        for step in range(N_STEPS):
            cfg=interpolate_cfg(SIZES_FIT[key],SIZES_MISS[key],step,N_STEPS)
            avg,f,b,g=fn(**cfg)
            print(f"{name:<9}{step+1:<5}{g:10.2f}{f/1e9:10.3f}{b/1e9:10.3f}{avg*1000:12.2f}")


if __name__=="__main__":
    run_scaled_suite()
