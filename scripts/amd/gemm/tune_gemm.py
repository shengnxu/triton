#!/usr/bin/env python3
import argparse
import sys

import torch
from torch.testing import assert_close

import triton
import triton.language as tl
from triton.runtime.jit import JITFunction, TensorWrapper, reinterpret
import yaml
import traceback

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    dot_outtype: tl.constexpr,
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
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=dot_outtype)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        a = tl.load(a_ptrs, mask=a_mask)
        b = tl.load(b_ptrs, mask=b_mask)
        accumulator += tl.dot(a, b, out_dtype=dot_outtype)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask)


def triton_matmul(a, b, c, block_m, block_n, block_k, split_k, num_warps, out_dtype, a_is_fp8):
    size_m = a.shape[0]
    size_n = b.shape[1]
    size_k = a.shape[1]

    if a_is_fp8:
        a_input = triton.reinterpret(a, tl.float8e4)
    else:
        a_input = a

    grid = lambda META: (
        triton.cdiv(size_m, block_m) * triton.cdiv(size_n, block_n),
        split_k
    )

    dot_outtype = tl.float16 if out_dtype is torch.float16 else tl.float32

    matmul_kernel[grid](a_ptr=a_input, b_ptr=b, c_ptr=c,
                        stride_am=a.stride(0), stride_ak=a.stride(1),
                        stride_bk=b.stride(0), stride_bn=b.stride(1),
                        stride_cm=c.stride(0), stride_cn=c.stride(1),
                        dot_outtype=dot_outtype,
                        M=size_m, N=size_n, K=size_k,
                        BLOCK_SIZE_M=block_m,
                        BLOCK_SIZE_N=block_n,
                        BLOCK_SIZE_K=block_k,
                        SPLIT_K=split_k,
                        num_warps=num_warps,
                        num_stages=1)


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

def get_tune_configs(SIZE_M, SIZE_N, SIZE_K):
    block_range = [32, 64, 128]
    split_k_range = [1, 2, 4, 5, 8, 10]
    num_warps_range = [1, 2, 4, 8, 16]
    # num_warps_range = [1, 2, 4]

    result = []
    for block_m in block_range:
        if SIZE_M <= 32 and block_m != 32:
            continue

        for block_n in block_range:
            if SIZE_N <=32 and block_n != 32:
                continue

            for block_k in block_range:
                for split_k in split_k_range:
                    leap = split_k * block_k
                    modv = SIZE_K % leap
                    if modv != 0:
                        continue

                    for num_warps in num_warps_range:
                        config = {}
                        config["block_m"] = block_m
                        config["block_n"] = block_n
                        config["block_k"] = block_k
                        config["split_k"] = split_k
                        config["num_warps"] = num_warps
                        result.append(config)

    return result


def tune_gemm(SIZE_M, SIZE_N, SIZE_K, c_type, a_use_fp8, output_file):
    b = torch.randn((SIZE_K, SIZE_N), device='cuda', dtype=torch.float16)
    c = torch.zeros((SIZE_M, SIZE_N), device=b.device, dtype=c_type)

    if a_use_fp8:
        a_type = tl.float8e4
        a = torch.randn((SIZE_M, SIZE_K), dtype=torch.float32, device='cuda') * 10
        a = a.to(torch.int8)
        # f32_to_f8 doesn't handle nan, so we make sure f8_tensor doesn't contain any nan
        all_exp_ones = (a & 0b01111100) == 128 - 2**a_type.fp_mantissa_width
        a[all_exp_ones] = 0
    else:
        a = torch.randn((SIZE_M, SIZE_K), device='cuda', dtype=torch.float16)

    tuning_configs = get_tune_configs(SIZE_M, SIZE_N, SIZE_K)

    min_time = 1024 * 1024 * 1024
    best_config = ''
    index = 0
    for config in tuning_configs:
        block_m = config["block_m"]
        block_n = config["block_n"]
        block_k = config["block_k"]
        split_k = config["split_k"]
        num_warps = config["num_warps"]

        try:
            perf_config = f'{block_m},{block_n},{block_k},{split_k},{num_warps}'
            print(f'{index}: perf_config: {perf_config}')
            c.zero_()
            exec_time = triton.testing.do_bench(
                lambda: triton_matmul(a, b, c, 
                                      block_m, block_n, block_k, 
                                      split_k, num_warps, c_type, 
                                      a_use_fp8))
        except Exception:
            traceback.print_exc()
            print("Exception happened in matmul, skip")
            continue

        print(f'{index}: perf_config: {perf_config}, time: {exec_time}\n')
        # It's not easy to get a proper error threshold in different size
        # Here the gemm calculation is padded to a different size in order to get
        # a variant version of the golden result. And the error between golden and
        # golden_variant provide reference on selecting the proper rtol / atol.
        # golden_variant = get_variant_golden(a, b)
        # golden_diff = golden - golden_variant
        # golden_abs_err = torch.max(torch.abs(golden_diff)).item()
        # golden_rel_err = torch.max(torch.abs(golden_diff / golden)).item()
        # torch.set_printoptions(profile="full")
        # try:
        #     assert_close(c, golden, rtol=max(0.05, 1.5 * golden_rel_err), atol=max(0.05, 1.5 * golden_abs_err), check_dtype=False)
        # except AssertionError:
        #     print(f"abs_error = {golden_abs_err}")
        #     print(f"rel_error = {golden_rel_err}")
        #     print('result mismatch, skip')
        #     continue

        if exec_time < min_time:
            min_time = exec_time
            best_config = perf_config
        index += 1

    strr = f'Best Result: {SIZE_M},{SIZE_N},{SIZE_K} best parameters: {best_config} --> {min_time}'
    print(strr)
    ofile = open(output_file, 'a')
    ofile.write(f'{strr}\n')
    ofile.close()


def parse_args():
    parser = argparse.ArgumentParser(
        prog="test gemm tuning",
        description="Tuning infra for triton gemm",
        allow_abbrev=False,
    )

    parser.add_argument("gemm_size_file", type=str, help='yaml file to indicate matrix size')
    parser.add_argument("--outfile", type=str, default='gemm_tune_outs', help='output file to store tune results')
    parser.add_argument("--fp8", action='store_true', default=False)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    matrix_size_file = args.gemm_size_file
    a_use_fp8 = args.fp8
    out_file = args.outfile
    with open(matrix_size_file) as file:
        matrix_sizes = yaml.safe_load(file)

    for sizes in matrix_sizes:
        M = sizes['M']
        N = sizes['N']
        K = sizes['K']

        tune_gemm(M, N, K, torch.float32, a_use_fp8, out_file)

if __name__ == '__main__':
    sys.exit(main())
