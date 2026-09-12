import torch

import mycuda


def assert_matches(a: torch.Tensor, b: torch.Tensor) -> None:
    actual = mycuda.gemm(a, b)
    expected = torch.mm(a, b)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    assert actual.is_contiguous()
    assert actual.device == a.device
    assert actual.dtype == torch.float32


def test_forward() -> None:
    for m, k, n in [
        (1, 1, 1),
        (16, 8, 16),
        (65, 33, 67),
        (128, 256, 128),
        (257, 129, 193),
        (7, 0, 9),
        (0, 5, 9),
        (7, 5, 0),
    ]:
        a = torch.randn(m, k, device="cuda", dtype=torch.float32)
        b = torch.randn(k, n, device="cuda", dtype=torch.float32)
        assert_matches(a, b)


def test_non_contiguous() -> None:
    a = torch.randn(37, 73, device="cuda").transpose(0, 1)
    b = torch.randn(41, 37, device="cuda").transpose(0, 1)
    assert not a.is_contiguous() and not b.is_contiguous()
    assert_matches(a, b)


def test_current_stream() -> None:
    stream = torch.cuda.Stream()
    a = torch.randn(129, 257, device="cuda")
    b = torch.randn(257, 131, device="cuda")
    with torch.cuda.stream(stream):
        actual = mycuda.gemm(a, b)
        expected = torch.mm(a, b)
    stream.synchronize()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)


def test_autograd() -> None:
    a = torch.randn(33, 17, device="cuda", requires_grad=True)
    b = torch.randn(17, 29, device="cuda", requires_grad=True)
    grad_output = torch.randn(33, 29, device="cuda")
    mycuda.gemm(a, b).backward(grad_output)
    expected_a = grad_output @ b.detach().transpose(0, 1)
    expected_b = a.detach().transpose(0, 1) @ grad_output
    torch.testing.assert_close(a.grad, expected_a)
    torch.testing.assert_close(b.grad, expected_b)


def test_dispatch_and_compile() -> None:
    a = torch.randn(31, 23, device="cuda")
    b = torch.randn(23, 19, device="cuda")
    direct = torch.ops.mycuda.gemm(a, b)
    torch.testing.assert_close(direct, torch.mm(a, b), rtol=1e-2, atol=1e-2)

    torch.library.opcheck(torch.ops.mycuda.gemm.default, (a, b))

    compiled = torch.compile(mycuda.gemm, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(a, b), direct)


def test_validation() -> None:
    bad_cases = [
        (torch.randn(2, 3), torch.randn(3, 4)),
        (torch.randn(2, 3, device="cuda", dtype=torch.float16),
         torch.randn(3, 4, device="cuda", dtype=torch.float16)),
        (torch.randn(2, 3, 1, device="cuda"),
         torch.randn(3, 4, device="cuda")),
        (torch.randn(2, 3, device="cuda"),
         torch.randn(5, 4, device="cuda")),
    ]
    for a, b in bad_cases:
        try:
            mycuda.gemm(a, b)
        except (RuntimeError, NotImplementedError):
            pass
        else:
            raise AssertionError("invalid inputs should be rejected")


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    major, minor = torch.cuda.get_device_capability()
    if major < 8:
        raise RuntimeError("the Tensor Core kernel requires SM80 or newer")
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("high")
    print(f"PyTorch {torch.__version__}, GPU {torch.cuda.get_device_name(0)} (SM{major}{minor})")
    test_forward()
    test_non_contiguous()
    test_current_stream()
    test_autograd()
    test_dispatch_and_compile()
    test_validation()
    print("All PyTorch extension tests passed.")


if __name__ == "__main__":
    main()
