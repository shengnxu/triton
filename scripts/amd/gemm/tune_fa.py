"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Extra Credits:
- Original flash attention paper (https://arxiv.org/abs/2205.14135)
- Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)
- Adam P. Goucher for simplified vector math

"""

import argparse
import pytest
import torch
import sys
import yaml
import csv
import re
import pandas as pd
import os

import triton
import triton.language as tl

from datetime import datetime
import subprocess

# Pick the fp8 data type

# AMD E5M2B16
# float8:tl.constexpr = torch.float8_e5m2fnuz

# AMD E4M3B8
# Note: When picking this f8 data type, scaling is required when using f8
# for the second gemm
TORCH_HAS_FP8E4 = hasattr(torch, 'float8_e4m3fnuz')
float8:tl.constexpr = None if not TORCH_HAS_FP8E4 else torch.float8_e4m3fnuz

def run_bash_command_wrapper(commandstring, capture=True):
    try:
        run_bash_command(commandstring, capture)
    except subprocess.CalledProcessError as e:
        if not capture:
            print(f"running {commandstring} one more time")
        run_bash_command(commandstring, capture)

def run_bash_command(commandstring, capture=True):
    if capture:
        proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash', stdout=subprocess.PIPE)
        return proc.stdout.splitlines()
    proc = subprocess.run(commandstring, shell=True, check=True, executable='/bin/bash')
    return None

def format_output(unformatted):
    if unformatted < 0.0001:
        formatted = "{:.3e}".format(unformatted)
    elif unformatted > 1000:
        formatted = "{:.1f}".format(unformatted)
    else:
        formatted = "{:.2f}".format(unformatted)
    return formatted

@triton.jit
def max_fn(x, y):
    return tl.math.max(x, y)

def generate_wrapper(tuning_parms):
    dri_str = """
name_to_torch_types = {
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
    'fp8': float8
}

def bench_flash_attention(BATCH, H, N_CTX, D_HEAD, causal=False, dtype='fp16'):
    if dtype == 'fp8' and not TORCH_HAS_FP8E4:
        sys.exit("fp8 is not available")
    init_dtype = torch.float16 if dtype != 'bf16' else torch.bfloat16
    q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda", requires_grad=True)
    k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=init_dtype, device="cuda", requires_grad=True)
    v = torch.randn((BATCH, H, D_HEAD, N_CTX), dtype=init_dtype, device="cuda", requires_grad=True)
    sm_scale = 1.3
    # q,k casting for partial fp8
    q = q.to(name_to_torch_types[dtype])
    k = k.to(name_to_torch_types[dtype])
    
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-2]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q, dtype=v.dtype)
    waves_per_eu = 2
    num_warps = 4
    num_stages = 1
    slice_k_tile = 32
    kpack = 1

    M = torch.empty((q.shape[0] * q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)
    """
    dri_str += '\n'
    for tp in tuning_parms:
        block_m = tp[0]
        block_n = tp[1]
        pre_load_v = tp[2]
        dri_str += f"""
    for i in range(100):
        grid = ( triton.cdiv(q.shape[2], {block_m}), q.shape[0] * q.shape[1], 1)
        _attn_fwd_BLOCKM_{block_m}_BLOCKN_{block_n}_Preloadv_{pre_load_v}[grid](
            q, k, v, sm_scale, M, o,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            q.shape[0], q.shape[1],
            N_CTX=q.shape[2],
            BLOCK_DMODEL=Lk,
            BLOCK_M = {block_m},
            BLOCK_N = {block_n},
            waves_per_eu = waves_per_eu,
            num_warps = num_warps,
            num_stages = num_stages,
            pre_load_v = {pre_load_v},
            slice_k_tile = slice_k_tile,
            kpack = kpack,
        )
        """

    return dri_str 

def generate_main(Batch, H, N_Ctx, D_Head):
    main_str = f"""
def main():
    bench_flash_attention({Batch}, {H}, {N_Ctx}, {D_Head})

if __name__ == '__main__':
    sys.exit(main())
    """

def generate_fa_kernel(Batch, H, N_Ctx, D_Head):
    # create the kernel file
    file_name = f"{Batch}_{H}_{N_Ctx}_{D_Head}.py"
    f_kernel = open("./generated_fa_kernel_"+file_name, 'w')

    # import string
    import_str = """import pytest
import torch
import sys

import triton
import triton.language as tl
# Pick the fp8 data type

# AMD E5M2B16
# float8:tl.constexpr = torch.float8_e5m2fnuz

# AMD E4M3B8
# Note: When picking this f8 data type, scaling is required when using f8
# for the second gemm
TORCH_HAS_FP8E4 = hasattr(torch, 'float8_e4m3fnuz')
float8:tl.constexpr = None if not TORCH_HAS_FP8E4 else torch.float8_e4m3fnuz
"""
    
    f_kernel.write(import_str + '\n')
    
    # generate kernels with tuning parameters
    tuning_parms = []
    block_m_range = [16, 32]
    block_n_range = [16, 32]
    pre_load_v_range = [True, False]
    with open(os.path.dirname(os.path.abspath(__file__))+"/flash_attention_fwd_kernel.py") as file:
        fa_kernel_code = file.read()

    for block_m in block_m_range:
        for block_n in block_n_range:
            for pre_load_v in pre_load_v_range:
                tuning_parms.append((block_m, block_n, pre_load_v))
                fa_kernel_str = fa_kernel_code.replace("attn_fwd_kernel", f"attn_fwd_BLOCKM_{block_m}_BLOCKN_{block_n}_Preloadv_{pre_load_v}")
                fa_kernel_str = fa_kernel_str.replace("import triton.language as tl", "")
                fa_kernel_str = fa_kernel_str.replace("import triton", "")
                f_kernel.write(fa_kernel_str + "\n")
   
    # generate the driver
    dri_str = generate_wrapper(tuning_parms)
    f_kernel.write(dri_str + "\n") 

    main_str = f"""
def main():
    bench_flash_attention({Batch}, {H}, {N_Ctx}, {D_Head})

if __name__ == '__main__':
    sys.exit(main())
    """
    f_kernel.write(main_str+'\n')
    f_kernel.close()

def tune_fa_config(Batch, H, N_Ctx, D_Head, num_threads, verbose):
    # create the kernel file
    generate_fa_kernel(Batch, H, N_Ctx, D_Head)
    run_bash_command("rm -rf ~/.triton/cache")
    start_time = datetime.now()

    file_name = f"generated_fa_kernel_{Batch}_{H}_{N_Ctx}_{D_Head}.py"
    run_bash_command(f"python {file_name} -n {num_threads}", capture=(verbose < 2)) 
    compile_end = datetime.now()
    compile_time = compile_end - start_time
    if verbose:
        print(f"compile time: {compile_time}", flush=True) 
    run_bash_command_wrapper(f"rocprof --stats python {file_name}", capture=(verbose < 2))
    df_prof = pd.read_csv(f"results.stats.csv")
    filtered_df = df_prof[df_prof['Name'].str.startswith('_attn_fwd_BLOCK')]
    # Find the row with the minimal 'AverageNs'
    min_row = filtered_df.loc[filtered_df['AverageNs'].idxmin()]

    splitted_config = min_row["Name"].split('_')
    best_config = {'Batch':Batch, 'H':H, 'N_Ctx':N_Ctx, 'D_Head':D_Head}
    best_config.update({'Block_M':splitted_config[4], 'Block_N':splitted_config[6], 'Preloadv':splitted_config[8]})
    return min_row['AverageNs'], best_config

def parse_args():
    parser = argparse.ArgumentParser(
        prog="tune a flash attention kernel",
        allow_abbrev=False,
    )

    parser.add_argument("-batch", type=int, default=16)
    parser.add_argument("-H", type=int, default=16)
    parser.add_argument("-n_ctx", type=int, default=1024)
    parser.add_argument("-d_head", type=int, default=128)
    parser.add_argument("--o", type=str, default='tuning_fa.yaml', help='yaml file to store tuning results')
    parser.add_argument("--fa_config_file", type=str, default="", help='yaml file to indicate flash attention configs')
    parser.add_argument("--keep", action='store_true', default=False, help='keep generated files')
    parser.add_argument("--verbose", action='store_true', default=False, help="enables time_breakdown and additional logging messages")
    parser.add_argument("--num_threads", type=int, default=16, help="number of threads to use for kernel compilation and post processing")
    parser.add_argument("--jobs", type=int, default=1, help="number of generated files")
    parser.add_argument("--iters", type=int, default=1000, help="number of iterations")
    parser.add_argument("--datatype", type=str, default='fp16', help="element type")
    parser.add_argument("--no_warmup", action='store_true', default=False, help="Do not call the warmup kernel")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    output = args.o
    keepTmp = args.keep
    jobs = args.jobs
    iters = args.iters
    skipWarmup = args.no_warmup

    # Get element type
    dtype = args.datatype

    fa_configs = []
    fa_config_file = args.fa_config_file
    if fa_config_file == "" or not os.path.isfile(fa_config_file):
        batch = args.batch
        h = args.H
        n_ctx = args.n_ctx
        d_head = args.d_head
        fa_configs = [(batch, h, n_ctx, d_head)]
    else:
        with open(fa_config_file) as file:
            inputs = yaml.safe_load(file)
        for item in inputs:
            fa_configs.append((item['Batch'], item['H'], item['N_Ctx'], item['D_Head']))

    f_results = open(output, 'w')
    for config in fa_configs:
        batch = config[0]
        h = config[1]
        n_ctx = config[2]
        d_head = config[3]
        minTime, bestConfig = tune_fa_config(batch, h, n_ctx, d_head, args.num_threads, args.verbose)
        minTime = format_output(minTime)
        print('best_config: ',str(bestConfig))
        f_results.write('- ' + str(bestConfig) + ' ')
        f_results.write(f'# time(us): {minTime}\n')
    f_results.close()

if __name__ == '__main__':
    sys.exit(main())
