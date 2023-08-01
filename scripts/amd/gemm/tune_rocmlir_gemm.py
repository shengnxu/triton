import argparse
import sys

import torch
from torch.testing import assert_close

import triton
import triton.language as tl
import yaml
import traceback

def getLenPerMfmaGroup(lenPerWave):
    return min(lenPerWave, 64)

def blocksPerMfmaInsn(mPerWave, nPerWave):
    smallSize = min(mPerWave, nPerWave)
    if smallSize < 16: return 4, 16
    elif smallSize < 32: return 16, 1
    else: return 32, 1

def isTuningRunnable(blockM, blockN, blockK, kPack, mPerWave, numWarps):
    nPerWave = blockM * blockN // numWarps // mPerWave
    if nPerWave > blockN:
        nPerWave = blockN
        mPerWave = blockM // numWarps
    
    mPerMfmaGroup = getLenPerMfmaGroup(mPerWave)
    nPerMfmaGroup = getLenPerMfmaGroup(nPerWave)

    if mPerMfmaGroup > nPerMfmaGroup:
        mPerMfmaGroup = nPerMfmaGroup
    else:
        nPerMfmaGroup = mPerMfmaGroup
    
    assert mPerMfmaGroup == nPerMfmaGroup
    mfmaNonKDim, blocksMfma = blocksPerMfmaInsn(mPerMfmaGroup, nPerMfmaGroup)

    rowGroupSize = 4
    waveSize = 64
    rowsPerMfmaOutput = min(waveSize // mfmaNonKDim, mfmaNonKDim // rowGroupSize)

    tmp = rowsPerMfmaOutput * mfmaNonKDim
    blocksPerMfmaOutput = (waveSize + tmp - 1) // tmp
    tmp1 = rowGroupSize * rowsPerMfmaOutput * blocksPerMfmaOutput
    rowGroupsPerBlock = (mfmaNonKDim + tmp1 - 1) // tmp1
    inputSpanLen = mfmaNonKDim * blocksPerMfmaOutput
    inputSpanPerMfmaIn = waveSize / inputSpanLen

    blocksInOutRegs = blocksMfma // blocksPerMfmaOutput
    inputSpanLen = mfmaNonKDim * blocksPerMfmaOutput
    inputSpansPerMfmaIn = waveSize // inputSpanLen

    isKReduction = (blocksInOutRegs == 1) and (inputSpansPerMfmaIn > 1)
    K = blockK // kPack

    kPerThread = K // inputSpansPerMfmaIn if isKReduction else K

    return kPerThread > 0

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
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

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        a = tl.load(a_ptrs, mask=a_mask)
        b = tl.load(b_ptrs, mask=b_mask)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask)


@triton.jit
def matmul_kernel_large(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
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

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = pid_z * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K * SPLIT_K)):
        a = tl.load(a_ptrs, mask=a_mask)
        b = tl.load(b_ptrs, mask=b_mask)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * SPLIT_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        tl.store(c_ptrs, accumulator, mask=c_mask)
    else:
        tl.atomic_add(c_ptrs, accumulator, mask=c_mask)


def triton_matmul(a, b, c, block_m, block_n, block_k, split_k, num_warps, kpack, mPerWave):
    size_m = a.shape[0]
    size_n = b.shape[1]
    size_k = a.shape[1]

    grid = lambda META: (
        triton.cdiv(size_m, block_m) * triton.cdiv(size_n, block_n),
        split_k
    )

    matmul_kernel[grid](a_ptr=a, b_ptr=b, c_ptr=c,
                        stride_am=a.stride(0), stride_ak=a.stride(1),
                        stride_bk=b.stride(0), stride_bn=b.stride(1),
                        stride_cm=c.stride(0), stride_cn=c.stride(1),
                        M=a.shape[0], N=b.shape[1], K=a.shape[1],
                        BLOCK_SIZE_M=block_m,
                        BLOCK_SIZE_N=block_n,
                        BLOCK_SIZE_K=block_k,
                        SPLIT_K=split_k,
                        num_warps=num_warps,
                        kpack=kpack, mPerWave=mPerWave
                        )

def triton_matmul_large(a, b, c, block_m, block_n, block_k, split_k, num_warps, kpack, mPerWave, group_size_m):
    size_m = a.shape[0]
    size_n = b.shape[1]
    size_k = a.shape[1]

    grid = lambda META: (
        triton.cdiv(size_m, block_m) * triton.cdiv(size_n, block_n),
        split_k
    )

    matmul_kernel_large[grid](a_ptr=a, b_ptr=b, c_ptr=c,
                        stride_am=a.stride(0), stride_ak=a.stride(1),
                        stride_bk=b.stride(0), stride_bn=b.stride(1),
                        stride_cm=c.stride(0), stride_cn=c.stride(1),
                        M=a.shape[0], N=b.shape[1], K=a.shape[1],
                        BLOCK_SIZE_M=block_m,
                        BLOCK_SIZE_N=block_n,
                        BLOCK_SIZE_K=block_k,
                        SPLIT_K=split_k,
                        GROUP_SIZE_M=group_size_m,
                        num_warps=num_warps,
                        kpack=kpack, mPerWave=mPerWave
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


def get_tuning_configs(SIZE_M, SIZE_N, SIZE_K):
    block_range = [16, 32, 64, 128]
    split_k_range = [1, 2, 5, 8, 10, 12, 18, 24]
    num_warps_range = [1, 2, 4, 8]
    group_m_range = [4, 8, 12]
    kpack_range = [4, 8]
    mper_wave_range = [16, 32]

    small_m = 0
    if SIZE_M <= 16:
        group_m_range = [1]
        small_m = 1

    results = []
    for block_m in block_range:
        if SIZE_M <= 16 and block_m != 16:
            continue

        for group_m in group_m_range:
            if small_m == 0:
                num_block_m =  SIZE_M // block_m
                if num_block_m < group_m:
                    continue
            
            for block_n in block_range:
                if SIZE_N <= 16 and block_n != 16:
                    continue

                for block_k in block_range:
                    for split_k in split_k_range:
                        leap = split_k * block_k
                        modv = SIZE_K % leap
                        if modv != 0:
                            continue

                        for mper_wave in mper_wave_range:
                            if mper_wave > block_m:
                                continue
                            for num_warps in num_warps_range:
                                max_num_warps = block_m * block_n // 16 // mper_wave
                                if num_warps > max_num_warps:
                                    continue

                                for kpack in kpack_range:
                                    config = {}
                                    config["block_m"] = block_m
                                    config["block_n"] = block_n
                                    config["block_k"] = block_k
                                    config["split_k"] = split_k
                                    config["mper_wave"] = mper_wave
                                    config["kpack"] = kpack
                                    config["num_warps"] = num_warps
                                    config["group_m"] = group_m
                                    results.append(config)
    return results


# def tune_gemm(SIZE_M, SIZE_N, SIZE_K, num_warps, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, SPLIT_K, kpack, mPerWave):
def tune_gemm(SIZE_M, SIZE_N, SIZE_K, output_file):
    print(f'Tune input size: {SIZE_M}, {SIZE_N}, {SIZE_K}')
    a = torch.randn((SIZE_M, SIZE_K), device='cuda', dtype=torch.float16)
    b = torch.randn((SIZE_K, SIZE_N), device='cuda', dtype=torch.float16)
    c = torch.zeros((SIZE_M, SIZE_N), device=a.device, dtype=torch.float32)

    # call pytorch function to get golden
    golden = torch.matmul(a, b)

    small_m = 0
    if SIZE_M <= 16:
        small_m = 1

    min_time = 1024 * 1024 * 1024
    best_config = ''
    index = 0
    
    tuning_configs = get_tuning_configs(SIZE_M, SIZE_N, SIZE_K)
    for config in tuning_configs:
        block_m = config["block_m"]
        block_n = config["block_n"]
        block_k = config["block_k"]
        split_k = config["split_k"]
        mper_wave = config["mper_wave"]
        kpack = config["kpack"]
        num_warps = config["num_warps"]
        group_m = config["group_m"]

        perf_config = f'{block_m},{block_n},{block_k},{split_k},{mper_wave},{kpack},{num_warps}'                                    
        if not isTuningRunnable(block_m, block_n, block_k, kpack, mper_wave, num_warps):
            print(f'Config: {perf_config} skipped for {SIZE_M},{SIZE_N},{SIZE_K}')
            continue

        try:
            perf_config = f'{block_m},{block_n},{block_k},{split_k},{group_m},{mper_wave},{kpack},{num_warps}'
            print(f'{index}: perf_config: {perf_config}')
            c.zero_()
            if small_m == 0:
                exec_time = triton.testing.do_bench(lambda: triton_matmul(a, b, c, block_m, block_n, block_k, split_k, num_warps, kpack, mper_wave))
            else:
                exec_time = triton.testing.do_bench(lambda: triton_matmul_large(a, b, c, block_m, block_n, block_k, split_k, num_warps, kpack, mper_wave, group_m))
        except Exception:
            print("Exception happened in matmul, skip")
            traceback.print_exc()
            continue

        print(f'{index}: perf_config: {perf_config}, time: {exec_time}')


        if exec_time < min_time:
            min_time = exec_time
            best_config = perf_config
        index += 1

        # It's not easy to get a proper error threshold in different size
        # Here the gemm calculation is padded to a different size in order to get
        # a variant version of the golden result. And the error between golden and
        # golden_variant provide reference on selecting the proper rtol / atol.
        golden_variant = get_variant_golden(a, b)
        golden_diff = golden - golden_variant
        golden_abs_err = torch.max(torch.abs(golden_diff)).item()
        golden_rel_err = torch.max(torch.abs(golden_diff / golden)).item()
        torch.set_printoptions(profile="full")
        # try:
        #     assert_close(c, golden, rtol=max(0.05, 1.5 * golden_rel_err), atol=max(0.05, 1.5 * golden_abs_err), check_dtype=False)
        # except AssertionError:
        #     print(f"abs_error = {golden_abs_err}")
        #     print(f"rel_error = {golden_rel_err}")
        #     print('result mismatch, skip')
        #     continue

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
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    matrix_size_file = args.gemm_size_file
    out_file = args.outfile
    with open(matrix_size_file) as file:
        matrix_sizes = yaml.safe_load(file)

    for sizes in matrix_sizes:
        M = sizes['M']
        N = sizes['N']
        K = sizes['K']

        tune_gemm(M, N, K, out_file)

if __name__ == '__main__':
    sys.exit(main())
