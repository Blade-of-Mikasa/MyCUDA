from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).resolve().parent
KERNEL_DIR = ROOT.parent / "src" / "kernel"


setup(
    name="mycuda-pytorch",
    version="0.1.0",
    description="PyTorch operator for the MyCUDA Tensor Core GEMM kernel",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="mycuda._C",
            sources=[str(ROOT / "csrc" / "gemm_op.cu")],
            include_dirs=[str(KERNEL_DIR)],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": ["-O3", "-std=c++20", "-lineinfo"],
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtension.with_options(no_python_abi_suffix=True)
    },
    zip_safe=False,
)
