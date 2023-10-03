import argparse
import sys
import yaml
import os
import subprocess



def get_full_tuning_space():
    configs = []

    block_mn_range = [16, 32, 64, 128]
    block_k_range = [16, 32, 64, 128]
    split_k_range = [1, 2, 4, 5, 6, 8, 10, 12, 16, 18, 24]
    num_warps_range = [1, 2, 4, 8]
    group_m_range = [1, 4, 8]
    # For now we see better perf with num_stages=0 for all gemm configs we care
    # But keep this explicit so that we do not forget we may need to set it to
    # other values in the future
    num_stage_range = [1, 0]

    for block_m in block_mn_range:
        for block_n in block_mn_range:
            for block_k in block_k_range:
                for num_warps in num_warps_range:
                    for group_m in group_m_range:
                        for split_k in split_k_range:
                            for num_stages in num_stage_range:
                                configs.append({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': block_k, 'GROUP_SIZE_M': group_m, 'SPLIT_K': split_k, 'num_warps': num_warps, 'num_stages': num_stages})

    return configs


def prune_configs(M, N, K, configs):

    pruned_configs = []
    mfma_type = os.getenv('MFMA_TYPE')
    if mfma_type == '16':
        mfma = 16
    else:
        mfma = 32

    for config in configs:
        BLOCK_SIZE_M = config.get("BLOCK_SIZE_M")
        BLOCK_SIZE_N = config.get("BLOCK_SIZE_N")
        BLOCK_SIZE_K = config.get("BLOCK_SIZE_K")
        SPLIT_K = config.get("SPLIT_K")
        GROUP_M = config.get("GROUP_SIZE_M")
        if M <= mfma and BLOCK_SIZE_M != mfma:
            continue
        if N <= mfma and BLOCK_SIZE_N != mfma:
            continue
        # skip large split_k when not necessary
        if SPLIT_K != 1 and not need_split_k(M, N, K):
            continue
        # skip split_k that leads to EVEN_K = false
        leap = SPLIT_K * BLOCK_SIZE_K
        modv = K % leap
        if modv != 0:
            continue
        # skip large GROUP_M
        if GROUP_M * BLOCK_SIZE_M > M and GROUP_M != 1:
            continue
        pruned_configs.append(config)

    return pruned_configs


def need_split_k(SIZE_M, SIZE_N, SIZE_K):
    return (SIZE_M < 64 or SIZE_N < 64) and SIZE_K > 1024


def run_bash_command(commandstring):
    proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash', stdout = subprocess.PIPE)
    return proc.stdout.splitlines()


def construct_funcName_from_config(M, N, K, config):
    block_m = config.get('BLOCK_SIZE_M')
    block_n = config.get('BLOCK_SIZE_N')
    block_k = config.get('BLOCK_SIZE_K')
    group_m = config.get('GROUP_SIZE_M')
    split_k = config.get('SPLIT_K')
    num_warps = config.get('num_warps')
    num_stages = config.get('num_stages')
    funcName = f"matmul_M{M}N{N}K{K}BM{block_m}BN{block_n}BK{block_k}GM{group_m}SK{split_k}nW{num_warps}nS{num_stages}"

    kernel_call_str = f"""    try:
        {funcName}[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_SIZE_M = {block_m},
            BLOCK_SIZE_N = {block_n},
            BLOCK_SIZE_K = {block_k},
            GROUP_SIZE_M = {group_m},
            SPLIT_K = {split_k},
            num_warps = {num_warps},
            num_stages = {num_stages}
        )
    except Exception:
        pass
"""
    return funcName, kernel_call_str

## Open a file generated_kernelMNK.py and generate
## 1. matmul kernels of all configs
## 2. wrapper function matmul to invoke all the generated kernels
## 3. test_gemm to invoke matmul in a loop of 10 iterations
def generate_kernel(M, N, K, configs):
    f_kernel = open(f'generated_kernel{M}{N}{K}.py', 'w')
    ## imports string
    import_str = """import torch
import triton
import triton.language as tl
"""
    f_kernel.write(import_str + "\n")

    ## test_gemm string
    test_gemm_str = """def test_gemm(M, N, K, dtype):
    a = torch.randn((M, K), device='cuda', dtype=dtype)
    b = torch.randn((K, N), device='cuda', dtype=dtype)
    # Allocates output.
    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)
    for i in range(10):
        d = matmul(a, b, c)
    return d
"""
    ## matmul string
    matmul_pre_str = """def matmul(a, b, c):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    M, K = a.shape
    K, N = b.shape
"""

    ## write matmul_kernels in generated_kernelMNK.py
    with open("matmul.kernel") as file:
        matmul_kernel_code = file.read();
    for config in configs:
        funcName, _ = construct_funcName_from_config(M, N, K, config)
        ## Copy the matmul_kernel with name replaced
        matmul_kernel_config = matmul_kernel_code.replace("matmul_kernel", funcName)
        f_kernel.write(matmul_kernel_config + "\n\n")

    ## write test_gemm
    f_kernel.write(test_gemm_str + "\n")

    ## write matmul function
    f_kernel.write(matmul_pre_str + "\n")

    for config in configs:
        block_m = config.get('BLOCK_SIZE_M')
        block_n = config.get('BLOCK_SIZE_N')
        split_k = config.get('SPLIT_K')
        grid_str = f'    grid = triton.cdiv(M, {block_m}) * triton.cdiv(N, {block_n}), {split_k}'
        funcName, kernel_call_str = construct_funcName_from_config(M, N, K, config)

        ## write the kernel
        f_kernel.write(grid_str + "\n\n")
        f_kernel.write(kernel_call_str + "\n")

    f_kernel.write("    return c\n")

    ## call test_gemm
    test_gemm_call_str = f'test_gemm({M}, {N}, {K}, torch.float16)'
    f_kernel.write(test_gemm_call_str)
    f_kernel.close()


def tune_gemm_config(M, N, K, configs):
    ## Generate kernel out of all configs
    generate_kernel(M, N, K, configs)

    ## remove any compiled kernel in the cache
    rm_cache_cmd = "rm -rf ~/.triton/cache"
    run_bash_command(rm_cache_cmd)

    ## profile generated kernels
    rocprof_cmd = f"rocprof --stats python generated_kernel{M}{N}{K}.py"
    run_bash_command(rocprof_cmd)

    ## post process results.csv to get the best config and minTime
    minTime = 1024 * 1024 * 1024
    for config in configs:
        funcName, _ = construct_funcName_from_config(M, N, K, config)
        parse_result_cmd = f'sed -n \'/{funcName}/p\' results.csv | awk -F \',\' \'{{print $NF}}\' | tail -n1'
        parsed_outputs = run_bash_command(parse_result_cmd)
        if parsed_outputs:
            min_us = int(parsed_outputs[0]) / 1000
            if min_us < minTime:
                minTime = min_us
                bestConfig = config
        else:
            min_us = -1
            print(f"invalid config: SIZE {M} {N} {K}: {config}")
    return minTime, bestConfig


def parse_args():
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,
    )

    parser.add_argument("--gemm_size_file", type=str, default="", help='yaml file to indicate matrix size')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    matrix_size_file = args.gemm_size_file
    if matrix_size_file == "" or not os.path.isfile(matrix_size_file):
        print(f"Matrix size file: {matrix_size_file} does not exist!")
        sys.exit(1)
    mnks = []
    with open(matrix_size_file) as file:
        matrix_sizes = yaml.safe_load(file)

    configs_full = get_full_tuning_space()
    for sizes in matrix_sizes:
        M = sizes['M']
        N = sizes['N']
        K = sizes['K']
        mnks.append((M, N, K))

    for (M, N, K) in mnks:
        pruned_configs = prune_configs(M, N, K, configs_full)
        print(f"Tuning GEMM (M: {M}, N: {N}, K: {K}) with {len(pruned_configs)} configs")
        minTime, bestConfig = tune_gemm_config(M, N, K, pruned_configs)
        perf_flops = lambda us: 2 * M * N * K * 1e-12 / (us)
        out_str = f'SIZE: {M} {N} {K} '
        print(f'{out_str} TFLOPS: {perf_flops(minTime)} time(us): {minTime} {bestConfig}')

if __name__ == '__main__':
    sys.exit(main())
