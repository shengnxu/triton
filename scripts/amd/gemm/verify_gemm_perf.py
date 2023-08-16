#!/usr/bin/env python3
import argparse
import sys

import torch
from torch.testing import assert_close

import triton
import triton.language as tl
from triton.runtime.jit import JITFunction, TensorWrapper, reinterpret
import traceback

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    out_dtype: tl.constexpr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr
):
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a_mask = (offs_m[:, None] < M) and (offs_k[None, :] < K)
    b_mask = (offs_k[:, None] < K) and (offs_n[None, :] < N)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=out_dtype)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        a = tl.load(a_ptrs, mask=a_mask)
        b = tl.load(b_ptrs, mask=b_mask)
        accumulator += tl.dot(a, b, out_dtype=out_dtype)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask)


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


def test_gemm(SIZE_M, 
              SIZE_N, 
              SIZE_K, 
              num_warps, 
              BLOCK_SIZE_M, 
              BLOCK_SIZE_N, 
              BLOCK_SIZE_K, 
              SPLIT_K, 
              a_is_fp8, 
              out_dtype,
              check_result):

    @triton.jit
    def copy_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        input = tl.load(input_ptr + offsets, mask=mask)
        output = input
        tl.store(output_ptr + offsets, output, mask=mask)

    if a_is_fp8:
        print("fp8_for_a_is_used-------------")
        a_type = tl.float8e4
        a = torch.randn((SIZE_M, SIZE_K), dtype=torch.float32, device='cuda') * 10
        a = a.to(torch.int8)
        # f32_to_f8 doesn't handle nan, so we make sure f8_tensor doesn't contain any nan
        all_exp_ones = (a & 0b01111100) == 128 - 2**a_type.fp_mantissa_width
        a[all_exp_ones] = 0
        a_fp16 = torch.empty_like(a, dtype=torch.float16)
        n_elements = a.numel()
        grid1 = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
        copy_kernel[grid1](triton.reinterpret(a, a_type), a_fp16, n_elements, BLOCK_SIZE=1024)
        a_triton_input = triton.reinterpret(a, a_type)
    else:
        a = torch.randn((SIZE_M, SIZE_K), device='cuda', dtype=torch.float16)
        a_fp16 = a
        a_triton_input = a

    b = torch.randn((SIZE_K, SIZE_N), device='cuda', dtype=torch.float16)
    c = torch.zeros((SIZE_M, SIZE_N), device=a.device, dtype=out_dtype)

    dot_dtype = tl.float16 if out_dtype is torch.float16 else tl.float32

    grid = lambda META: (
        triton.cdiv(SIZE_M, BLOCK_SIZE_M) * triton.cdiv(SIZE_N, BLOCK_SIZE_N),
        SPLIT_K
    )
    matmul_kernel[grid](a_ptr=a_triton_input, b_ptr=b, c_ptr=c,
                        stride_am=a.stride(0), stride_ak=a.stride(1),
                        stride_bk=b.stride(0), stride_bn=b.stride(1),
                        stride_cm=c.stride(0), stride_cn=c.stride(1),
                        M=a.shape[0], N=b.shape[1], K=a.shape[1],
                        out_dtype=dot_dtype,
                        BLOCK_SIZE_M=BLOCK_SIZE_M,
                        BLOCK_SIZE_N=BLOCK_SIZE_N,
                        BLOCK_SIZE_K=BLOCK_SIZE_K,
                        SPLIT_K=SPLIT_K,
                        num_warps=num_warps,
                        num_stages=1)
    # golden = torch.matmul(a_fp16, b)

    # It's not easy to get a proper error threshold in different size
    # Here the gemm calculation is padded to a different size in order to get
    # a variant version of the golden result. And the error between golden and
    # golden_variant provide reference on selecting the proper rtol / atol.
    # golden_variant = get_variant_golden(a, b)
    # golden_diff = golden - golden_variant
    # golden_abs_err = torch.max(torch.abs(golden_diff)).item()
    # golden_rel_err = torch.max(torch.abs(golden_diff / golden)).item()
    # print(f'golden_abs_err = {golden_abs_err}, golden_rel_err = {golden_rel_err}')
    # torch.set_printoptions(profile="full")
    # assert_close(c, golden, rtol=3e-2, atol=3e-2)
    # print("after_assert_close")

    # assert_close(c, golden)
    # assert_close(c, golden, rtol=max(1e-4, 1.5 * golden_rel_err), atol=max(1e-4, 1.5 * golden_abs_err), check_dtype=False)


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="test gemm tuning",
        description="Tuning infra for triton gemm",
        allow_abbrev=False,
    )

    parser.add_argument("-m", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-n", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-k", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-blockM", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-blockN", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-blockK", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-splitK", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-num_warps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--fp8", action='store_true', default=False)
    parser.add_argument("--compare", action='store_true', default=False)

    parsed_args = parser.parse_args(args)

    M = parsed_args.m
    N = parsed_args.n
    K = parsed_args.k
    num_warps = parsed_args.num_warps
    BLOCK_M = parsed_args.blockM
    BLOCK_N = parsed_args.blockN
    BLOCK_K = parsed_args.blockK
    SPLIT_K = parsed_args.splitK
    a_is_fp8 = parsed_args.fp8
    print(f'a_is_f8 = {a_is_fp8}')
    check_ret = parsed_args.compare
    test_gemm(M, N, K, num_warps, BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_K, 
              a_is_fp8, torch.float16, check_ret)


if __name__ == '__main__':
    sys.exit(main())
