import torch
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import threading

class CPUMonitor:
    def __init__(self, interval=0.05):
        self.interval = interval
        self.usage = []
        self.running = False
        self.thread = None

    def _monitor(self):
        while self.running:
            self.usage.append(psutil.cpu_percent(interval=None))
            time.sleep(self.interval)

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def average(self):
        return sum(self.usage) / len(self.usage) if self.usage else 0

# 1. STREAM Triad (메모리 대역폭 측정)
def measure_stream_triad(N=10_000_000, repeats=10):
    a = torch.ones(N, dtype=torch.float32)
    b = torch.ones(N, dtype=torch.float32) * 2.0
    c = torch.ones(N, dtype=torch.float32) * 3.0
    scalar = 2.0

    for _ in range(3):
        a = b + scalar * c

    best_time = float('inf')
    for _ in range(repeats):
        start = time.time()
        a = b + scalar * c
        end = time.time()
        best_time = min(best_time, end - start)

    bytes_accessed = 3 * N * 4  # float32 = 4 bytes
    return bytes_accessed / best_time / 1e9  # GB/s

# 2. Conv2D 연산 성능 측정
def measure_conv_workload(channels=64, size=112, repeats=3):
    conv = torch.nn.Conv2d(channels, channels, 3, padding=1)
    x = torch.rand((1, channels, size, size))
    with torch.no_grad():
        for _ in range(2):
            _ = conv(x)

    best_time = float('inf')
    for _ in range(repeats):
        start = time.time()
        _ = conv(x)
        end = time.time()
        best_time = min(best_time, end - start)

    K = 3
    flops = 2 * channels * size * size * K * K * channels
    return flops / best_time / 1e9  # GFLOPs/s

def run_oi_sweep(fixed_flops=1_000_000_000, num_threads=4):
    base_size = 1024 * 1024
    max_mb = 300
    step_mb = 3

    a = torch.ones(1_000_000, dtype=torch.float32)
    b = torch.ones(1_000_000, dtype=torch.float32)
    input_bytes = (a.numel() + b.numel()) * 4

    print(f"{'TotalMB':>8} {'OI':>10} {'Time(ms)':>10} {'CPU(%)':>8}")
    print('-' * 42)

    def dot_workload(repeat, a, b, dummy=None):
        with torch.no_grad():
            for _ in range(repeat):
                _ = torch.dot(a, b)
            if dummy is not None:
                dummy += 1

    for dummy_mb in range(0, max_mb + 1, step_mb):
        dummy_bytes = dummy_mb * base_size
        dummy = torch.ones(dummy_bytes // 4, dtype=torch.float32) if dummy_bytes > 0 else None

        ops_per_iter = a.numel() * 2
        total_repeat = fixed_flops // ops_per_iter
        repeat_per_thread = total_repeat // num_threads

        # 워밍업
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(dot_workload, repeat_per_thread, a, b, dummy) for _ in range(num_threads)]
            for f in as_completed(futures):
                pass

        # 본 측정 + CPU 사용률 측정
        monitor = CPUMonitor(interval=0.05)
        monitor.start()
        start = time.time()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(dot_workload, repeat_per_thread, a, b, dummy) for _ in range(num_threads)]
            for f in as_completed(futures):
                pass
        end = time.time()
        monitor.stop()

        avg_cpu = monitor.average()
        total_bytes = input_bytes + dummy_bytes
        oi = fixed_flops / total_bytes
        total_mb = total_bytes / 1e6
        elapsed_ms = (end - start) * 1000

        print(f"{total_mb:8.2f} {oi:10.2f} {elapsed_ms:10.2f} {avg_cpu:8.2f}")


def analyze_model(flops, bytes, peak_gflops, peak_bw_gbs):
    oi = flops / bytes
    ridge = peak_gflops / peak_bw_gbs
    achieved_gflops = oi * peak_bw_gbs if oi < ridge else peak_gflops
    latency = flops / (achieved_gflops * 1e9)
    print(f"  OI:          {oi:.2f} FLOPs/Byte")
    print(f"  Ridge Point: {ridge:.2f} FLOPs/Byte")
    print(f"  Est. Latency:{latency*1000:.2f} ms")
    print(f"  Bound:       {'Memory' if oi < ridge else 'Compute'}-bound")

# main
if __name__ == "__main__":
   
    for i in range (0, 20):
        time.sleep(1)
        mem_bw = measure_stream_triad()
        compute_peak = measure_conv_workload()
        ridge_point = compute_peak / mem_bw
        print(f"  Mem_CPU_Ridge : {mem_bw:.2f} {compute_peak:.2f} {ridge_point:.2f}")  
    run_oi_sweep(fixed_flops=1000000000)

