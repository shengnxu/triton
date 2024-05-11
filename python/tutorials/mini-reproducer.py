
import torch

import triton
import triton.language as tl
import sys
import argparse
import pytest

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128,  'BLOCK_SIZE_K': 16, },
                      num_warps=4) ,
    ] if torch.version.hip is None else [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 16, },
                      num_warps=4, num_stages=0),
    ],
    key=['M', 'K'],
    reset_to_zero=['bias_ptr'],
)
@triton.jit
def reduction_kernel(
        # Pointers to matrices
        a_ptr, 
        # bias ptr
        bias_ptr,
        # Matrix dimensions
        M, K,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am, stride_ak,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr,  BLOCK_SIZE_K: tl.constexpr,
):
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid // num_pid_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    bias_gradient = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    num_k = tl.cdiv(K, BLOCK_SIZE_K)
    #print('num_k=', num_k)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        bias_gradient_partial = tl.sum(a, axis=1)
        bias_gradient += bias_gradient_partial

        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak

    offs_bias_gradient = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    bias_gradient_ptrs = bias_ptr + offs_bias_gradient
    tl.store(bias_gradient_ptrs, bias_gradient, mask=(offs_bias_gradient<M))


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.


def reduction(a):
    # Check constraints.
    M, K = a.shape
    # Allocates output.
    bias_gradient = torch.empty((M,), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']),)
    reduction_kernel[grid](
        a, 
        bias_gradient,
        M, K,  #
        a.stride(0), a.stride(1),  #
    )
    return bias_gradient


# %%
# Unit Test
# ---------
#
# We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS).
@pytest.mark.parametrize("M, K, in_dtype, out_dtype",
[ (*shape, in_dtype, out_dtype)
    for shape in [#(128, 256, 32), 
        (3, 16),
                   ]
    for in_dtype, out_dtype in [#('int8', 'int8'),
                                ('float32', 'float32')]]
)
def test_correctness(M, K, in_dtype, out_dtype):
    torch.manual_seed(0)
    a = torch.randn((M, K), device='cuda', dtype=torch.float32)
    triton_bias_gradient = reduction(a)
    torch_bias_gradient = a.sum(axis=1)
    print(f"a={a}")
    print(f"triton_bias_gradient={triton_bias_gradient}")
    print(f"torch_bias_gradient={torch_bias_gradient}")
    rtol = 0 if torch.version.hip is None else 1e-2
    if torch.allclose(triton_bias_gradient, torch_bias_gradient, atol=1e-2, rtol=rtol):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
        assert torch.allclose(triton_bias_gradient, torch_bias_gradient, atol=1e-2, rtol=rtol)


