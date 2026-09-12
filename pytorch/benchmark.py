import torch

import mycuda


def measure_ms(fn, iterations: int) -> float:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def tflops(n: int, milliseconds: float) -> float:
    return 2.0 * n**3 / (milliseconds * 1.0e9)


def main() -> None:
    torch.set_float32_matmul_precision("high")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("    N  iter  mycuda_ms  mycuda_TF  torch_ms  torch_TF  ratio")
    for n, iterations in [(1024, 200), (2048, 100), (4096, 30), (8192, 10)]:
        a = torch.randn(n, n, device="cuda")
        b = torch.randn(n, n, device="cuda")
        custom_ms = measure_ms(lambda: mycuda.gemm(a, b), iterations)
        torch_ms = measure_ms(lambda: torch.mm(a, b), iterations)
        custom_tf = tflops(n, custom_ms)
        torch_tf = tflops(n, torch_ms)
        print(
            f"{n:5d} {iterations:5d} {custom_ms:10.3f} {custom_tf:10.2f} "
            f"{torch_ms:9.3f} {torch_tf:9.2f} {100 * custom_tf / torch_tf:6.1f}%"
        )


if __name__ == "__main__":
    main()
