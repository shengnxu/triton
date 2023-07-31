import re
import os
import subprocess
import argparse

def run_perf_num(config_file, out_file, warmup_runs = 10):
    file = open(config_file, 'r')
    lines = file.readlines()

    for line in lines:
        print(line)
        if not line or line[0] == '#' or line[0] == ' ':
            continue
        x = line.split()
        sizes = x[2].split(',')
        params = x[5].split(',')
        M = sizes[0]
        N = sizes[1]
        K = sizes[2]
        BLOCK_M = params[0]
        BLOCK_N = params[1]
        BLOCK_K = params[2]
        SPLIT_K = params[3]
        if len(params) == 8:
            num_groups = params[4]
            mPerWave = params[5]
            kpack = params[6]
            num_warps = params[7]
            driver = 'matmul_grouped.py'
        else:
            mPerWave = params[4]
            kpack = params[5]
            num_warps = params[6]
            driver = 'matmul.py'

        run_cmd = f'python ../../../python/test/unit/language/{driver} -m {M} -n {N} -k {K} -blockM {BLOCK_M} -blockN {BLOCK_N} -blockK {BLOCK_K} -splitK {SPLIT_K} -mPerWave {mPerWave} -kpack {kpack} -num_warps {num_warps}'
        if len(params) == 8:
            run_cmd += f' -groupM {num_groups}'
        prof_cmd = f'rocprof --stats {run_cmd}'

        # # warmup runs
        # for i in range(warmup_runs):
        #     os.system(run_cmd)
            
        print(f'prof_cmd: {prof_cmd}')
        os.system(prof_cmd)
        parse_result = f'sed -n \'/matmul_kernel/p\' results.stats.csv | awk -F \',\' \'{{print $4}}\''
        os.system(parse_result + ' >> ' + out_file)

def parse_args():
    parser = argparse.ArgumentParser(description="Rerun tuning perf numbers")
    parser.add_argument('config_file', type=str, metavar='config_file', help='Tuning result file')
    parser.add_argument("--out_file", type=str, metavar='out_file', default="outputs", help='Name of the output file to store perf numbers')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    config_file = args.config_file
    out_file = args.out_file

    run_perf_num(config_file, out_file)

if __name__ == "__main__":
    main()

