#!/usr/bin/env python3
import argparse
import sys

import pytest
import torch
from torch.testing import assert_close

import triton
import triton.language as tl
import yaml
import os


@triton.jit
def matmul_kernel_splitK(
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
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr, EVEN_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    pid_z = tl.program_id(1)
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
    offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    if torch.version.hip is None:
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    else:
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            k_remaining = K - k * (BLOCK_SIZE_K * SPLIT_K)
            a = tl.load(a_ptrs, mask=offs_k[None, :] < k_remaining, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, c, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, c, mask=c_mask)


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
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
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
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    if torch.version.hip is None:
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
        a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    else:
        offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
        offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
        a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`.
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.

def use_split_k(size_m, size_n, size_k):
    return (size_m < 64 or size_n < 64) and size_k > 1024

def triton_matmul(a, b, c, block_m, block_n, block_k, split_k, group_m, num_warps):
    size_m = a.shape[0]
    size_n = b.shape[1]
    size_k = a.shape[1]

    if use_split_k(size_m, size_n, size_k):
        grid_splitK = lambda META: (
            triton.cdiv(size_m, META['BLOCK_SIZE_M']) * triton.cdiv(size_n, META['BLOCK_SIZE_N']),
            META['SPLIT_K']
        )
        matmul_kernel_splitK[grid](a_ptr=a, b_ptr=b, c_ptr=c,
                            stride_am=a.stride(0), stride_ak=a.stride(1),
                            stride_bk=b.stride(0), stride_bn=b.stride(1),
                            stride_cm=c.stride(0), stride_cn=c.stride(1),
                            M=size_m, N=size_n, K=size_k,
                            BLOCK_SIZE_M=block_m,
                            BLOCK_SIZE_N=block_n,
                            BLOCK_SIZE_K=block_k,
                            SPLIT_K=split_k,
                            num_stages=1,
                            GROUP_SIZE_M=group_m,
                            num_warps=num_warps)
    else:
        grid = lambda META: (
            triton.cdiv(size_m, META['BLOCK_SIZE_M']) * triton.cdiv(size_n, META['BLOCK_SIZE_N']),
        )
        matmul_kernel[grid](a_ptr=a, b_ptr=b, c_ptr=c,
                            stride_am=a.stride(0), stride_ak=a.stride(1),
                            stride_bk=b.stride(0), stride_bn=b.stride(1),
                            stride_cm=c.stride(0), stride_cn=c.stride(1),
                            M=size_m, N=size_n, K=size_k,
                            BLOCK_SIZE_M=block_m,
                            BLOCK_SIZE_N=block_n,
                            BLOCK_SIZE_K=block_k,
                            num_stages=1,
                            GROUP_SIZE_M=group_m,
                            num_warps=num_warps)

    return c

# def matmul(a, b, activation=""):
#     # Check constraints.
#     assert a.shape[1] == b.shape[0], "Incompatible dimensions"
#     assert a.is_contiguous(), "Matrix A must be contiguous"
#     assert b.is_contiguous(), "Matrix B must be contiguous"
#     M, K = a.shape
#     K, N = b.shape
#     # Allocates output.
#     c = torch.empty((M, N), device=a.device, dtype=a.dtype)
#     # 1D launch kernel where each block gets its own program.

#     if (M < 64 or N < 64) and K > 1024:
#         grid_splitK = lambda META: (
#             triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
#             META['SPLIT_K']
#         )
#         matmul_kernel_splitK[grid_splitK](
#             a, b, c,
#             M, N, K,
#             a.stride(0), a.stride(1),
#             b.stride(0), b.stride(1),
#             c.stride(0), c.stride(1),
#             ACTIVATION=activation
#         )
#     else:
#         grid = lambda META: (
#             triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),
#         )
#         matmul_kernel[grid](
#             a, b, c,
#             M, N, K,
#             a.stride(0), a.stride(1),
#             b.stride(0), b.stride(1),
#             c.stride(0), c.stride(1),
#             ACTIVATION=activation
#         )

#     return c


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


# if no tuning config is available, get these default small tuning space
# to run the gemm performance
def get_default_tuning_configs(SIZE_M, SIZE_N, SIZE_K, use_split_k):
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'SPLIT_K': 1}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'SPLIT_K': 8}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'SPLIT_K': 10}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'SPLIT_K': 8}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 1, 'SPLIT_K': 10}, num_stages=1, num_warps=1),
    ] if use_split_k else [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=1, num_warps=2),
    ]

    # add M, N, K to each tuning config
    for i, c in enumerate(configs):
        c["M"] = SIZE_M
        c["N"] = SIZE_N
        c["K"] = SIZE_K
        configs[i] = c

    return configs

def get_full_tuning_sapce(SIZE_M, SIZE_N, SIZE_K, use_split_k):
    # tune space
    block_range = [32, 64, 128]
    split_k_range = [1, 2, 4, 5, 8, 10]
    num_warps_range = [1, 2, 4, 8, 16]
    group_m_range = [1, 4, 8]

    full_tuning_sapce = []
    for block_m in block_range:
        if SIZE_M <= 32 and block_m != 32:
            continue

        for block_n in block_range:
            if SIZE_N <=32 and block_n != 32:
                continue

            for block_k in block_range:
                for num_warps in num_warps_range:
                    for group_m in group_m_range:
                        if not use_split_k:
                            tuning_config = {}
                            tuning_config["M"] = SIZE_M
                            tuning_config["N"] = SIZE_N
                            tuning_config["K"] = SIZE_K
                            tuning_config["BLOCK_M"] = block_m
                            tuning_config["BLOCK_N"] = block_n
                            tuning_config["BLOCK_K"] = block_k
                            tuning_config["SPLIT_K"] = 1
                            tuning_config["GROUP_M"] = group_m
                            tuning_config["NUM_WARPS"] = num_warps
                            full_tuning_sapce.append(tuning_config)
                        else:
                            for split_k in split_k_range:
                                leap = split_k * block_k
                                modv = SIZE_K % leap
                                if modv != 0:
                                    continue
                        
                                tuning_config = {}
                                tuning_config["M"] = SIZE_M
                                tuning_config["N"] = SIZE_N
                                tuning_config["K"] = SIZE_K
                                tuning_config["BLOCK_M"] = block_m
                                tuning_config["BLOCK_N"] = block_n
                                tuning_config["SPLIT_K"] = split_k
                                tuning_config["BLOCK_K"] = block_k
                                tuning_config["GROUP_M"] = group_m
                                tuning_config["NUM_WARPS"] = num_warps
                                full_tuning_sapce.append(tuning_config)
    return full_tuning_sapce


def get_gemm_tuning_cache_file():
    TRITON_DIR = os.getenv('TRITON_DIR')
    file_path_name = TRITON_DIR + "/scripts/amd/gemm/gemm_tuning_config.yaml"
    return file_path_name


# get the full tuning space to tune the input GEMM size
def get_tuning_space(SIZE_M, SIZE_N, SIZE_K, use_split_k, force_tuning, force_no_tuning):
    use_split_k = use_split_k(SIZE_M, SIZE_N, SIZE_K)
    if force_tuning:
        return get_full_tuning_sapce(SIZE_M, SIZE_N, SIZE_K, use_split_k)
    else:
        # read from the cache tuning config to get the tuning config
        if force_tuning:
            return get_full_tuning_sapce(SIZE_M, SIZE_N, SIZE_K, use_split_k)
        else:
            tuning_config_cache_file = get_gemm_tuning_cache_file()
            if os.path.isfile(tuning_config_cache_file):                
                with open(tuning_config_cache_file, "r") as config_file:
                    configs = yaml.safe_load(config_file)
                    for config in configs:
                        M = config["M"]
                        N = config["N"]
                        K = config["K"]
                        if SIZE_M == M and SIZE_N == N and SIZE_K == K:
                            return [config]
        
        # input matrix size is not cached
        # force_no_tuning is et
        if force_no_tuning:
            return get_default_tuning_configs(SIZE_M, SIZE_N, SIZE_K, use_split_k)
        else:
            return get_full_tuning_sapce(SIZE_M, SIZE_N, SIZE_K, use_split_k)
        
def update_cached_tuning_config(best_config):
    tuning_configs = []

    tuning_config_cache_file = get_gemm_tuning_cache_file()
    if os.path.isfile(tuning_config_cache_file):
        with open(tuning_config_cache_file, "r") as config_file:
            tuning_configs = yaml.safe_load(config_file)
    
    M = best_config["M"]
    N = best_config["N"]
    K = best_config["K"]
    b_found = False
    for i, config in enumerate(tuning_configs):
        if M == config["M"] and N == config["N"] and K == config["K"]:
            tuning_configs[i] = best_config
            b_found = True
            break

    # matrix size not in the cache, so add tuning config for the 
    # input size
    if not b_found:
        tuning_configs.append(best_config)

    with open(tuning_config_cache_file, "w") as config_file:
        yaml.dump(tuning_configs, config_file)

# def tune_gemm(SIZE_M, SIZE_N, SIZE_K, num_warps, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, SPLIT_K, kpack, mPerWave):
def tune_gemm(SIZE_M, SIZE_N, SIZE_K, tuning_configs, compare_result, force_no_tuning):
    a = torch.randn((SIZE_M, SIZE_K), device='cuda', dtype=torch.float16)
    b = torch.randn((SIZE_K, SIZE_N), device='cuda', dtype=torch.float16)
    c = torch.zeros((SIZE_M, SIZE_N), device=a.device, dtype=torch.float32)

    warmup_iter_num = 10
    repeat_num = 50
    # heuristics: large input matrices repeat less number of times
    if SIZE_M >= 512 and SIZE_N >= 512 and SIZE_K >= 512:
        warmup_iter_num = 5
        repeat_num = 20

    index = 0
    best_config = {}
    min_time = 1024 * 1024 * 1024
    for tune_config in tuning_configs:
        block_m = tune_config["BLOCK_M"]
        block_n = tune_config["BLOCK_N"]
        block_k = tune_config["BLOCK_K"]
        split_k = tune_config["SPLIT_K"]
        group_m = tune_config["GROUP_M"]
        num_warps = tune_config["NUM_WARPS"]
        
        try:
            perf_config_str = f'{block_m},{block_n},{block_k},{split_k},{num_warps}'
            c.zero_()
            exec_time = triton.testing.do_bench(lambda: triton_matmul(a, b, c, block_m, block_n, block_k, split_k, group_m, num_warps), warmup=warmup_iter_num, rep=repeat_num)
        except Exception:
            print("Exception happened in matmul, skip")
            continue

        if compare_result:
            # call pytorch function to get golden
            golden = torch.matmul(a, b)

            # It's not easy to get a proper error threshold in different size
            # Here the gemm calculation is padded to a different size in order to get
            # a variant version of the golden result. And the error between golden and
            # golden_variant provide reference on selecting the proper rtol / atol.
            golden_variant = get_variant_golden(a, b)
            golden_diff = golden - golden_variant
            golden_abs_err = torch.max(torch.abs(golden_diff)).item()
            golden_rel_err = torch.max(torch.abs(golden_diff / golden)).item()
            torch.set_printoptions(profile="full")
            try:
                assert_close(c, golden, rtol=max(0.05, 1.5 * golden_rel_err), atol=max(0.05, 1.5 * golden_abs_err), check_dtype=False)
            except AssertionError:
                print(f"abs_error = {golden_abs_err}")
                print(f"rel_error = {golden_rel_err}")
                print('result mismatch, skip')
                continue

        if exec_time < min_time:
            min_time = exec_time
            best_config = tune_config
        print(f"{index}: m = {SIZE_M}, n = {SIZE_N}, k = {SIZE_K}, tune_config = {perf_config_str}, min_time = {min_time} ms", )
        index += 1

    if not force_no_tuning:
        update_cached_tuning_config(best_config)

    flops = 2 * SIZE_M * SIZE_N * SIZE_K / min_time / 1.0e9
    best_config_str = f'{best_config["BLOCK_M"]},{best_config["BLOCK_N"]},{best_config["BLOCK_K"]},{best_config["SPLIT_K"]},{best_config["NUM_WARPS"]}'
    strr = f'Best Result: {SIZE_M},{SIZE_N},{SIZE_K} best parameters: {best_config_str} --> {flops} TFLOPS, {min_time * 1000000}'
    print(strr)


def parse_args():
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,
    )

    parser.add_argument("-m", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-n", type=int, default=argparse.SUPPRESS)
    parser.add_argument("-k", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--compare", action='store_false', default=False)
    # default behavior is: If tuning config is available for an input size,
    # Use the tuning config to run performance directly. Otherwise, if
    # tuning config is unavailable, we will tune the input size and cache
    # the tuning output. 
    # with "--force_tuning" set to True, this script tunes the input matrix size
    # no matter whether tuning config is availalbe. Then cache the tuning
    # output (will overwrite the existing tuning config if existing) 
    # With "--force_no_tuning" set to True, this script will use the cached
    # tuning config to run GEMM. If no tuing config is available, will use
    # a very small tuning configs to run a result, but not cache the output.
    # Note: these two flag cannot be True at the same time. 
    parser.add_argument("--force_tuning", action='store_true', default=False)
    parser.add_argument("--force_no_tuning", action='store_true', default=False)

    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    M = args.m
    N = args.n
    K = args.k

    force_tuning = False
    if args.force_tuning:
        force_tuning = True

    force_no_tuning = False
    if args.force_no_tuning:
        force_no_tuning = True

    if force_tuning and force_no_tuning:
        print("Flags \"--force_tuning\" and \"--force_no_tuning\" cannot be set at the same time!")
        sys.exit(1)

    compare_result=False
    if args.compare:
        compare_result=True

    tuning_configs = get_tuning_space(M, N, K, force_tuning, force_no_tuning)
    tune_gemm(M, N, K, tuning_configs, compare_result, force_no_tuning)

if __name__ == '__main__':
    sys.exit(main())
