import argparse
import sys
import yaml
import os
import subprocess



def get_full_tuning_space():
    configs = []

    block_mn_range = [32, 64, 128]
    block_k_range = [32, 64]
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
    for config in configs:
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K =\
            config.get("BLOCK_SIZE_M"), config.get("BLOCK_SIZE_N"), config.get("BLOCK_SIZE_K")
        SPLIT_K = config.get("SPLIT_K")
        GROUP_M = config.get("GROUP_SIZE_M")
        if M <=32 and BLOCK_SIZE_M != 32:
            continue
        if N <=32 and BLOCK_SIZE_N != 32:
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


def tune_gemm_config(M, N, K):
    configs = get_full_tuning_space()
    #print(len(configs))
    pruned_configs = prune_configs(M, N, K, configs)
    #print(f"Tuning GEMM (M: {M}, N: {N}, K: {K}) with {len(pruned_configs)} configs")
    minTime = 1024 * 1024 * 1024
    index = 1
    for config in pruned_configs:
        block_m = config.get('BLOCK_SIZE_M')
        block_n = config.get('BLOCK_SIZE_N')
        block_k = config.get('BLOCK_SIZE_K')
        group_m = config.get('GROUP_SIZE_M')
        split_k = config.get('SPLIT_K')
        num_warps = config.get('num_warps')
        num_stages = config.get('num_stages')
        driver = 'rocprof_gemm.py'
        TRITON_DIR = os.getenv('TRITON_DIR')
        if TRITON_DIR is not None:
            driver = os.path.join(TRITON_DIR, 'scripts/amd/gemm', driver)
        run_cmd = f'python {driver} -m {M} -n {N} -k {K} \
                    -block_m {block_m} -block_n {block_n} -block_k {block_k} \
                    -group_m {group_m} -split_k {split_k} -num_warps {num_warps} \
                    -num_stages {num_stages}'
        prof_cmd = f'rocprof --stats {run_cmd}'
        run_bash_command(prof_cmd)

        parse_result_cmd = f'sed -n \'/matmul_kernel/p\' results.csv | awk -F \',\' \'{{print $NF}}\' | tail -n1'
        parse_outputs = run_bash_command(parse_result_cmd)
        min_us = int(parse_outputs[0]) / 1000
        if min_us < minTime:
            minTime = min_us
            bestConfig = config
        #print(f"index {index}/{len(pruned_configs)}: time: {min_us}, bestTime: {minTime}")
        index = index + 1
    return minTime, bestConfig


def main():
    mnks = []
    with open('toy-gemm-sizes.yaml') as file:
        matrix_sizes = yaml.safe_load(file)
    for sizes in matrix_sizes:
        M = sizes['M']
        N = sizes['N']
        K = sizes['K']
        mnks.append((M, N, K))

    for (M, N, K) in mnks:
        minTime, bestConfig = tune_gemm_config(M, N, K)
        perf_flops = lambda us: 2 * M * N * K * 1e-12 / (us)
        out_str = f'SIZE: {M},{N},{K} '
        print(f'{out_str} TFLOPS: {perf_flops(minTime)} time(us): {minTime} {bestConfig}')

if __name__ == '__main__':
    sys.exit(main())
