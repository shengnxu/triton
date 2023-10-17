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
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1}, num_stages=1, num_warps=4),
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

def matmul(a, b, c, c_type=torch.float32):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
    )

    if c_type == torch.float16:
        comp_type = tl.float16
    else:
        comp_type = tl.float32

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        compute_type = comp_type,
    )


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


name_to_torch_types = {
    'int8': torch.int8,
    'int32': torch.int32,
    'fp16': torch.float16,
    'fp32': torch.float32,
    'bf32': torch.bfloat16,
    'fp8e4b8': torch.float8_e4m3fnuz,
    'fp8e5b16': torch.float8_e5m2fnuz,
    'fp8e4': torch.float8_e4m3fn,
    'fp8e5': torch.float8_e5m2,
}

name_to_triton_types = {
    'int8': tl.int8,
    'int32': tl.int32,
    'fp16': tl.float16,
    'fp32': tl.float32,
    'bf16': tl.bfloat16,
    'fp8e4b8': tl.float8e4b8,
    'fp8e5b16': tl.float8e5b16,
    'fp8e4': tl.float8e4nv,
    'fp8e5': tl.float8e5,
}

def gen_input(M, N, d_type, seed, device='cuda'):
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    raw_data = torch.randn((M, N), dtype=torch.float32, device='cuda')
    if 'fp8' in d_type: # d_type is float8
        assert d_type in name_to_torch_types
        torch_f8 = raw_data.to(name_to_torch_types[d_type])
        # input = torch_f8
        input = triton.reinterpret(torch_f8, name_to_triton_types[d_type])
        input_f16 = torch_f8.to(torch.float16)
    else:
        input = raw_data.to(name_to_torch_types[d_type])
        input_f16 = input.to(torch.float16)

    return input, input_f16


def test_gemm(SIZE_M, SIZE_N, SIZE_K, a_type, b_type, c_type):
    print(f"testing sizes: M: {SIZE_M}, N: {SIZE_N}, K: {SIZE_K}, a_type: {a_type}, b_type: {b_type}, c_type: {c_type}")
    a, a_f16 = gen_input(SIZE_M, SIZE_K, a_type, 10, device='cuda')
    b, b_f16 = gen_input(SIZE_K, SIZE_N, b_type, 11, device='cuda')

    c = torch.empty((SIZE_M, SIZE_N), device=a.device, dtype=name_to_torch_types[c_type])
    golden = torch.matmul(a_f16, b_f16)
    matmul(a, b, c, name_to_torch_types[c_type])

    print(f'gold = {golden}')
    print(f'c = {c}')
    torch.testing.assert_close(c, golden, atol=8e-2, rtol=6e-2, check_dtype=False)


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog = "test gemm tuning",
        description= "Tuning MFMA GEMM implementation",
        allow_abbrev=False,
    )

    parser.add_argument("-m", type=int, required=True, default=argparse.SUPPRESS)
    parser.add_argument("-n", type=int, required=True, default=argparse.SUPPRESS)
    parser.add_argument("-k", type=int, required=True, default=argparse.SUPPRESS)
    parser.add_argument("-dta", type=str, default='fp16', help="float8 type: fp8e4/fp8e5")
    parser.add_argument("-dtb", type=str, default='fp16', help="float8 type: fp8e4/fp8e5")
    parser.add_argument("-dtc", type=str, default='fp16', help="output type: fp16/fp32/int32")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    M = args.m
    N = args.n
    K = args.k
    a_type = "fp16"
    b_type = "fp16"
    # fp8_type = torch.float8_e4m3fnuz
    # fp8_type = torch.float8_e5m2fnuz
    # fp8_type = torch.float8_e5m2
    # fp8_type = torch.float8_e4m3fn
    c_type = "fp16"

    if args.dta != "fp16":
        assert args.dta in name_to_torch_types
        a_type = args.dta

    if args.dtb != "fp16":
        assert args.dtb in name_to_torch_types
        b_type = args.dtb

    if args.dtc != "fp16":
        assert args.dtc in name_to_torch_types
        c_type = args.dtc

    try:
        test_gemm(M, N, K, a_type, b_type, c_type)
    except:
        traceback.print_exc()
        print("FAILED!")
    else:
        print("PASSED!")

if __name__ == "__main__":
    main()
