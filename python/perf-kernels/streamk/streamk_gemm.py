from typing import Optional

import torch
import triton
import triton.language as tl
import random

from persistent_streamk_kernel import persistent_streamk_gemm
from streamk_kernel import streamk_gemm

torch.manual_seed(123)
random.seed(123)

total_sm = 304
print(f"total SMs: {total_sm}")

class matmul(torch.autograd.Function):

    _debug = True

    @staticmethod
    def set_debug(debug: bool):
        matmul._debug = debug

    @staticmethod
    def _call(kernel_callable, *positional_args, **keyword_args):
        # Get grid size from the keyword arguments or provide a default value
      #  grids = keyword_args.get("total_programs_streamk", total_sm)
        grids = positional_args[6]
        print("grids =", grids)

        # Call the kernel with the appropriate arguments
        kernel_instance = kernel_callable[(grids,)](*positional_args, **keyword_args)

        if matmul._debug:
            print(f"{kernel_instance.n_regs} registers used, {kernel_instance.n_spills} spills")
            # Uncomment these lines if needed to print specific assembly instructions
            # print(kernel_instance.asm['ttgir'])
            # print(kernel_instance.asm['amdgcn'])

        return keyword_args.get("c")  # Adjust return value as per your requirements

    @staticmethod
    def forward(ctx, kernel_callable, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, grid: int, *args):
        """
        Forward method of the matmul class.

        :param ctx: Context for backward computation.
        :param kernel_callable: The callable kernel function to invoke.
        :param a, b, c: Input tensors.
        :param grid: Grid size for the kernel.
        :param args: Additional kernel arguments.
        """
        # Unpack additional arguments
        kwargs = args[0]

        # Ensure the tensor dimensions are compatible
        assert a.shape[1] == b.shape[0], "incompatible dimensions"

        # Calculate strides
        stride_am, stride_ak = a.stride()
        stride_bk, stride_bn = b.stride()
        stride_cm, stride_cn = c.stride()

        # Extract dimensions
        M, K = a.shape
        _, N = b.shape

        # Debugging information
        if matmul._debug:
            print(f"M: {M}, N: {N}, K: {K}")
            print(f"stride_am: {stride_am}, stride_ak: {stride_ak}")
            print(f"stride_bk: {stride_bk}, stride_bn: {stride_bn}")
            print(f"stride_cm: {stride_cm}, stride_cn: {stride_cn}")
            print(f"Additional kwargs: {kwargs}")

        # Call the specified kernel using the updated `_call` method
        matmul._call(
            kernel_callable,
            a, b, c,  # Core positional arguments for the kernel
            M, N, K, grid,  # Dimensions and hardware information
            stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,  # Strides
            **kwargs
        )
        return c

# Mapping of kernel names to callable kernel functions
kernel_map = {
    "simple": streamk_gemm,
#    "two_tile": another_kernel_function,
    "persistent": persistent_streamk_gemm
}

kernel_callable = kernel_map.get("persistent")
print(kernel_callable)

m, n, k = 4096, 4096, 8192  # some problem size to test
A = torch.randn(m, k, device="cuda", dtype=torch.float16)
B = torch.randn(n, k, device="cuda", dtype=torch.float16).T
C = torch.zeros((m, n), device="cuda", dtype=A.dtype)
BLK_M = 64
BLK_N = 64
BLK_K = 64
gsize_m = 4
two_tiles = 'True'
num_stages = 0
num_warps = 4
waves_per_eu = 0
mfmaInstrSize = 16
kpack = 2
grid = 304
print("kernel_name =", kernel_callable)

C = matmul.apply(
    kernel_callable,  # Your specific kernel function
    A,
    B,
    C,
    grid,
    {
        'BLOCK_SIZE_M': BLK_M,
        'BLOCK_SIZE_N': BLK_N,
        'BLOCK_SIZE_K': BLK_K,
        'GROUP_SIZE_M': gsize_m,
        'num_stages': num_stages,
        'num_warps': num_warps,
        'waves_per_eu': waves_per_eu,
        'matrix_instr_nonkdim': mfmaInstrSize,
        'kpack': kpack,
    }
)

expected = A @ B
print("C matrix from custom kernel:")
print(C)
print("Expected C matrix:")
print(expected)
assert torch.allclose(C, expected, atol=1), f"max: {(C - expected).abs().max().item()}\n{C}\n{expected}"
print("pass validation test")
