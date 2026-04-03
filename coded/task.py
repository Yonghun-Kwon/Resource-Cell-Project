import torch
import time

def run_inference(input_tensor, model):
    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        _ = model(input_tensor)
        end = time.perf_counter()
    latency = end - start
    print(f"[TASK] Inference completed in {latency:.6f} seconds")
    return latency

def benchmark_run():
    # 단순한 행렬 곱으로 workload 발생
    A = torch.randn(1024, 1024)
    B = torch.randn(1024, 1024)
    torch.matmul(A, B)  # warm-up

    start = time.perf_counter()
    for _ in range(10):
        torch.matmul(A, B)
    end = time.perf_counter()

    elapsed = end - start
    cell_count = 1024 * 1024 * 1024 * 2 * 10 / 1e9  # 간단한 flop 추정
    print(f"[TASK] Benchmark run time: {elapsed:.4f}s, estimated cell count: {cell_count:.2f}G")
    return int(cell_count)