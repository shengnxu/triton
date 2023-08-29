"""
Matrix Multiplication Tuning Scripts, Changed from the tutorial example "python/tutorials/03-matrix-multiplication.py"
"""

import torch

import triton
import triton.language as tl
import argparse
import sys
import yaml
import os
import gemm_kernels
import subprocess

def run_speed(M, N, K, datatype, use_rocprof, provider):
    a = torch.randn((M, K), device='cuda', dtype=datatype)
    b = torch.randn((K, N), device='cuda', dtype=datatype)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'pytorch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: gemm_kernels.matmul(a, b), quantiles=quantiles)
    return min_ms

def run_bash_command(commandstring):
    #print( commandstring )
    proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash', stdout = subprocess.PIPE)
    return proc.stdout.splitlines()


def parse_args():
    parser = argparse.ArgumentParser(
        prog="tune a specific gemm size",
        allow_abbrev=False,
    )

    parser.add_argument("-m", type=int, default=0)
    parser.add_argument("-n", type=int, default=0)
    parser.add_argument("-k", type=int, default=0)
    parser.add_argument("-dtype", type=str, default='fp16', help="Input data type, default is fp16")
    parser.add_argument("--specify_type", action='store_true', default=False, help="Whether user specify data type, default false")
    parser.add_argument("--specify_size", action='store_true', default=False, help="Whether user specify input matrix size, default false")
    parser.add_argument("--compare", action='store_true', default=False, help="Whether check result correctness")
    parser.add_argument("--gemm_size_file", type=str, default="", help='yaml file to indicate matrix size')
    parser.add_argument("--rocprof", action='store_true', default=False, help='Use rocprof to measure kernel time, default uses do_bench()!')
    parser.add_argument("-v", action='store_true', default=False, help="Print out the best tuning config")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    dtype = torch.float16
    if args.specify_type:
        if args.dtype == 'fp16':
            dtype = torch.float16
        elif args.dtype == 'fp32':
            dtype = torch.float32
        elif args.dtype == 'bf16':
            dtype = torch.bfloat16
        else:
            print(f"Unsupported datatype {args.dtype}")
            sys.exit(1)
    use_rocprof = args.rocprof
    verbose = args.v

    mnks = []
    if args.specify_size:
        M = args.m
        N = args.n
        K = args.k
        if M == 0 or N == 0 or K == 0:
            print(f"Input matrix size: (M {M}, N {N}, K {K}) contains dim size 0!")
        mnks = [(M, N, K)]
    else:
        matrix_size_file = args.gemm_size_file
        if matrix_size_file == "" or not os.path.isfile(matrix_size_file):
            print(f"Matrix size file: {matrix_size_file} does not exist!")
            sys.exit(1)

        with open(matrix_size_file) as file:
            matrix_sizes = yaml.safe_load(file)

        for sizes in matrix_sizes:
            M = sizes['M']
            N = sizes['N']
            K = sizes['K']
            mnks.append((M, N, K))


    for (m, n, k) in mnks:
        min_ms = run_speed(m, n, k, dtype, use_rocprof, 'triton')

        # function to compute flops
        perf_flops = lambda ms: 2 * m * n * k * 1e-12 / (ms * 1e-3)

        if args.compare:
            gemm_kernels.test_correctness(m, n, k, dtype)
        best_config = gemm_kernels.get_best_config(m, n, k)

        if use_rocprof:
            dtype_str = 'fp16' if (not args.specify_type) else args.dtype 
            run_cmd = f'python rocprof_gemm.py -m {M} -n {N} -k {K} -dtype {dtype_str} -v'
            prof_cmd = f'rocprof --stats {run_cmd}'
            prof_result = run_bash_command(prof_cmd)
            print(f'prof_result = {prof_result}')

            parse_result_cmd = f'sed -n \'/matmul_kernel/p\' results.stats.csv | awk -F \',\' \'{{print $4}}\''
            parse_outputs = run_bash_command(parse_result_cmd)
            print(f'parse_outputs = {parse_outputs}')
            min_ms = int(parse_outputs[0]) / 1000000

        print(f'min_ms = {min_ms}')
        out_str = f'SIZE: {m},{n},{k} '
        # print best config
        if verbose:
            out_str += f'  best_config: ({best_config}),   '
        out_str += f'TFLOPS: {perf_flops(min_ms)} time(ns): {min_ms * 1000000}'
        print(out_str)

        # print(f'SIZE: {m}, {n}, {k}, TFLOPS: {perf_flops(min_ms)}, time: {min_ms * 1000000}')

if __name__ == '__main__':
    sys.exit(main())
