import argparse
import sys
import yaml
import os
import glob
import subprocess
from datetime import datetime

import torch
import triton
import triton.language as tl

def run_bash_command(commandstring):
    proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash', stdout = subprocess.PIPE)
    return proc.stdout.splitlines()

def tune_fa_config(BS,H,N,D):
    rep = 10000
    run_bash_command("rm -rf ~/.triton/cache")
    run_bash_command(f"rocprof --stats -o fa-rocprof.csv python benchmark-fa-fwd-transV-MI300.py -bs {BS} -nheads {H} -d {D} -seqlen {N} -rep {rep}")
    parse_result_cmd = f'sed -n \'/attn_fwd/p\' fa-rocprof.stats.csv | awk -F \',\' \'{{print $4}}\' | tail -n1'
    parsed_outputs = run_bash_command(parse_result_cmd)
    return int(parsed_outputs[0]) / 1000000


def parse_args():
    parser = argparse.ArgumentParser(
        prog="tune FA fwd",
        allow_abbrev=False,
    )

    #parser.add_argument("-m", type=int, default=0)
    #parser.add_argument("-n", type=int, default=0)
    #parser.add_argument("-k", type=int, default=0)
    #parser.add_argument("--ngpus", type=int, default=1, help='number of GPUs used in the profiling step')
    parser.add_argument("--fa_size_file", type=str, default="", help='yaml file to indicate matrix size')
    #parser.add_argument("--tuning_results_file", type=str, default=get_default_tuning_result_filename(), help='yaml file to store tuning results')
    parser.add_argument("--keep", action='store_true', default=False, help='keep generated files')
    parser.add_argument("--compare", action='store_true', default=False, help="Whether check result correctness")
    parser.add_argument("--compare_wo_tuning", action='store_true', default=False, help="Whether check result correctness")
    parser.add_argument("--time_breakdown", action='store_true', default=False, help="Show detailed time breakdown of each step during the tuning")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    fa_size_file = args.fa_size_file
    #tuning_output_file = args.tuning_results_file
    keepTmp = args.keep
    #ngpus = args.ngpus

    mnks = []
    ## TODO: make it more robust to get user input
    #if fa_size_file == "" or not os.path.isfile(fa_size_file):
    #    M = args.m
    #    N = args.n
    #    K = args.k
    #    mnks = [(M, N, K)]
    #else:
    with open(fa_size_file) as file:
        fa_sizes = yaml.safe_load(file)
    for sizes in fa_sizes:
        BS = sizes['BS']
        H = sizes['H']
        N = sizes['N']
        D = sizes['D']
        mnks.append((BS, H, N, D))

    ## Check correctness from given configs
    #if args.compare_wo_tuning:
    #    for item in matrix_sizes:
    #        M = item['M']
    #        N = item['N']
    #        K = item['K']
    #        del item['M']
    #        del item['N']
    #        del item['K']
    #        test_correctness(M, N, K, item, True)
    #    return

    #configs_full = get_full_tuning_space()

    start_time = datetime.now()
    #print(f"Tuning starts at: {start_time}")

    #f_results = open(tuning_output_file, 'w')
    for (BS,H,N,D) in mnks:
        start_local_time = datetime.now()
        ## Obtain a pruned tuning space according to gemm size
        #pruned_configs = prune_configs(M, N, K, configs_full)

        #size_str = f'SIZE: {M} {N} {K}'
        #print(f"{size_str} nConfigs: {len(pruned_configs)}", end=" ", flush=True)

        minTime = tune_fa_config(BS,H,N,D)
        ## The main tuning funtion for one gemm size
        #minTime, bestConfig, compile_time, profile_time, post_time = tune_gemm_config(M, N, K, pruned_configs, ngpus = ngpus, verbose=args.time_breakdown)

        ## post processing the numbers
        perf_tflops = lambda ms: 2 * 2 * BS * H * D * N * N / ms * 1e-9
        tri_tflops = perf_tflops(minTime)
        if tri_tflops < 0.0001:
            formatted_tflops = "{:.3e}".format(tri_tflops)
        else:
            formatted_tflops = "{:.2f}".format(tri_tflops)
        print(f'D{D}-BS{BS}-H{H}-seqlen{N}: TFLOPS: {formatted_tflops} time(ms): {minTime}')

        #bestConfig_compact_str, _ = gen_kernel_and_configStr_from_config(M, N, K, bestConfig)
        #print(f'best_config: {bestConfig_compact_str}', end=" ")

        ## write best config to tuning_results.yaml
        #sizeDict = {'M': M, 'N': N, 'K': K}
        #sizeDict.update(bestConfig)
        #f_results.write("- " + str(sizeDict) + " ")
        #f_results.write(f'# TFLOPS: {formatted_tflops} time(us): {minTime:.2f}\n')

        ## remove generated files if asked to
        for f in glob.glob(f"fa-rocprof.*"):
            os.remove(f)

        ## Check correctness if asked to
        #if args.compare:
        #    print("correctness: ", end=" ")
        #    test_correctness(M, N, K, bestConfig, False)
        #else:
        #    print("")

        #end_local_time = datetime.now()
        #print(f">>> Elapsed time: {end_local_time - start_local_time} = {compile_time} (compile) + {profile_time} (profile) + {post_time} (post processing)")

    #f_results.close()

    #end_time = datetime.now()
    #tuning_time = end_time - start_time
    #print(f"Tuning ends at: {end_time}")
    #print(f"Total tuning time (h:m:s): {tuning_time}")


if __name__ == '__main__':
    sys.exit(main())

