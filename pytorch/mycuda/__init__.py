"""PyTorch bindings for the MyCUDA Tensor Core GEMM kernel."""

from pathlib import Path

import torch


def _load_native_library() -> None:
    candidates = sorted(Path(__file__).parent.glob("_C*.so"))
    if not candidates:
        raise ImportError(
            "MyCUDA native extension is not built. Run "
            "`python -m pip install -e . --no-build-isolation` in the "
            "pytorch directory."
        )
    torch.ops.load_library(str(candidates[0]))


_load_native_library()


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return ``a @ b`` using the MyCUDA TF32 Tensor Core kernel.

    Inputs must be two-dimensional CUDA float32 tensors on the same device.
    Non-contiguous inputs are accepted and made contiguous by the native op.
    """

    return torch.ops.mycuda.gemm(a, b)


@torch.library.register_fake("mycuda::gemm")
def _fake_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    torch._check(a.dim() == 2 and b.dim() == 2, lambda: "inputs must be 2-D")
    torch._check(a.dtype == torch.float32 and b.dtype == torch.float32,
                 lambda: "inputs must be float32")
    torch._check(a.device.type == "cuda" and b.device.type == "cuda",
                 lambda: "inputs must be CUDA tensors")
    torch._check(a.device == b.device, lambda: "inputs must be on one device")
    torch._check(a.shape[1] == b.shape[0], lambda: "matrix shapes must align")
    return a.new_empty((a.shape[0], b.shape[1]))


def _setup_context(ctx, inputs, output) -> None:
    del output
    a, b = inputs
    ctx.save_for_backward(a, b)


def _backward(ctx, grad_output: torch.Tensor):
    a, b = ctx.saved_tensors
    grad_a = grad_output @ b.transpose(0, 1) if ctx.needs_input_grad[0] else None
    grad_b = a.transpose(0, 1) @ grad_output if ctx.needs_input_grad[1] else None
    return grad_a, grad_b


torch.library.register_autograd(
    "mycuda::gemm", _backward, setup_context=_setup_context
)


__all__ = ["gemm"]
