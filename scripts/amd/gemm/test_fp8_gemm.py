#!/usr/bin/python3
import torch
from torch.testing import assert_close

import triton
import triton._C.libtriton.triton as _triton
import triton.language as tl
from triton.runtime.jit import JITFunction, TensorWrapper, reinterpret
import triton.ops as to
import traceback
import argparse

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1}, num_stages=1, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    compute_type:tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=compute_type)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b, out_dtype=compute_type)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    c = accumulator
    # c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def matmul(a, b, c_type=torch.float32, a_is_fp8 = False):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape

    if a_is_fp8:
        print("ConvertedTof8")
        in_dtype = tl.float8e5
        a_input = triton.reinterpret(a, in_dtype)
    else:
        a_input = a

    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=c_type)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    if c_type == torch.float16:
        comp_type = tl.float16
    else:
        comp_type = tl.float32


    matmul_kernel[grid](
        a_input, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        compute_type = comp_type,
    )

    return c

def get_variant_golden(a, b):
    SIZE_M = a.shape[0]
    SIZE_K = a.shape[1]
    SIZE_N = b.shape[1]
    assert a.shape[1] == b.shape[0]
    zero_M_K = torch.zeros((SIZE_M, SIZE_K)).cuda()
    zero_3M_K = torch.zeros((3 * SIZE_M, SIZE_K)).cuda()
    zero_K_N = torch.zeros((SIZE_K, SIZE_N)).cuda()
    zero_3K_N = torch.zeros((3 * SIZE_K, SIZE_N)).cuda()
    a_padded = torch.cat((a, zero_M_K, zero_M_K), 0)
    a_padded = torch.cat((a_padded, zero_3M_K, zero_3M_K), 1)
    b_padded = torch.cat((b, zero_K_N, zero_K_N), 0)
    b_padded = torch.cat((b_padded, zero_3K_N, zero_3K_N), 1)
    c_padded = torch.matmul(a_padded, b_padded)
    return c_padded[:SIZE_M, :SIZE_N]

def test_gemm(SIZE_M, SIZE_N, SIZE_K, ab_type, c_type, a_is_f8 = False):
    print("testing sizes: M: {}, N: {}, K: {}, ab type: {}, c type: {}, a_is_f8: {}".format(SIZE_M, SIZE_N, SIZE_K, ab_type, c_type, a_is_f8))

    @triton.jit
    def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input = tl.load(input_ptr + offsets, mask=mask)
        output = input
        tl.store(output_ptr + offsets, output, mask=mask)

    if a_is_f8:
        a_type = tl.float8e5
        f8_tensor = torch.randn((SIZE_M, SIZE_K), dtype=torch.float32, device='cuda') * 10
        f8_tensor = f8_tensor.to(torch.int8)
        # f32_to_f8 doesn't handle nan, so we make sure f8_tensor doesn't contain any nan
        all_exp_ones = (f8_tensor & 0b01111100) == 128 - 2**a_type.fp_mantissa_width
        f8_tensor[all_exp_ones] = 0
        n_elements = f8_tensor.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        a_f16 = torch.empty_like(f8_tensor, dtype=torch.float16)
        copy_kernel[grid](triton.reinterpret(f8_tensor, a_type), a_f16, n_elements, BLOCK_SIZE=1024)
        b_f16 = torch.randn((SIZE_K, SIZE_N), device = 'cuda', dtype=torch.float16)

        golden = torch.matmul(a_f16, b_f16)
        c = matmul(f8_tensor, b_f16, c_type, True)

        print(f'gold = {golden}')
        print(f'c = {c}')

        golden_abs_err = 0.5
        golden_rel_err = 0.0
    elif ab_type == torch.int8:
        a = torch.randn((SIZE_M, SIZE_K), device='cuda', dtype=torch.float32).to(torch.int8)
        b = torch.randn((SIZE_K, SIZE_N), device='cuda', dtype=torch.float32).to(torch.int8)
        golden = torch.matmul(a.to(torch.float64), b.to(torch.float64))
        c = matmul(a, b, c_type)
    
        golden_abs_err = 0.5
        golden_rel_err = 0.0
    else:
        a = torch.randn((SIZE_M, SIZE_K), device='cuda', dtype=ab_type)
        b = torch.randn((SIZE_K, SIZE_N), device='cuda', dtype=ab_type)
        golden = torch.matmul(a, b)
        c = matmul(a, b, c_type)
    
        golden_variant = get_variant_golden(a, b)
        golden_diff = golden - golden_variant
        golden_abs_err = torch.max(torch.abs(golden_diff)).item()
        golden_rel_err = torch.max(torch.abs(golden_diff / golden)).item()

    torch.set_printoptions(profile="full")
    assert_close(c.to(torch.float64), golden.to(torch.float64), rtol=max(1e-3, 10 * golden_rel_err), atol=max(1e-3, 10 * golden_abs_err), check_dtype=False)

def parse_arguments():
    parser = argparse.ArgumentParser(
        prog = "test gemm tuning",
        description= "Tuning MFMA GEMM implementation",
        allow_abbrev=False,
    )

    parser.add_argument("-m", type=int, required=True, default=argparse.SUPPRESS)
    parser.add_argument("-n", type=int, required=True, default=argparse.SUPPRESS)
    parser.add_argument("-k", type=int, required=True, default=argparse.SUPPRESS)
    parser.add_argument("--fp8", action='store_true', default=False)

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    M = args.m
    N = args.n
    K = args.k
    a_is_fp8 = False
    if args.fp8:
        a_is_fp8 = True

    try:
        test_gemm(M, N, K, torch.float16, torch.float32, a_is_f8 = a_is_fp8)
    except:
        traceback.print_exc()
        print("FAILED!")
    else:
        print("PASSED!")

if __name__ == "__main__":
    main()
