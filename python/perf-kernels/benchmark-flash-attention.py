import subprocess
import os
import pandas as pd

from itertools import product

parameters = {
    "dtype": ["bf16"],
    "batch": [8, 16, 32, 64],
    "num_heads": [4],
    "attention_dim": [64, 128],
    "max_seq_len": [2048, 4096],
}


def profile_batch_kernels(jobs, gpuid):
    os.environ['ROCR_VISIBLE_DEVICES'] = str(gpuid)
    for jobId, config in enumerate(jobs):
        # config = phantom_configs[job]
        dtype, batch, num_heads, attention_dim, max_seq_len = config
        max_seq_len *= batch
        program = f"./perf-kernels/flash-attention.py -b {batch} -hq {num_heads} -hkv {num_heads} -nctx_kv {max_seq_len} -nctx_q {max_seq_len} -d {attention_dim} -bench_custom -dtype {dtype}"
        # print(program)
        result_dir = "perf"
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        print(f"rocprof --stats -o {result_dir}/results-{jobId}.csv " + program)
        # run_bash_command_wrapper(f"rocprof --stats -o {result_dir}/results-{jobId}.csv " + program)

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

def extract_tflops(jobs, kernel_name="_attn_fwd"):
    print("precision,batch_size,#heads,max_seq_len,attention_dim,Triton")
    for jobId, config in enumerate(jobs):
        dtype, batch, num_heads, attention_dim, max_seq_len = config
        max_seq_len *= batch
        numOps = batch * max_seq_len**2 * num_heads * attention_dim * 2 * 2
        # df = pd.read_csv(f"perf/results-{jobId}.csv")
        # durationNs = df[df.KernelName.str.contains(kernel_name)].DurationNs
        # durationNs = durationNs.iloc[0] * 1e-9
        df = pd.read_csv(f"perf/results-{jobId}.stats.csv")
        durationNs = df[df.Name.str.contains(kernel_name)].AverageNs
        durationNs = durationNs.iloc[0] * 1e-9
        print(f"{dtype},{batch},{num_heads},{max_seq_len/batch},{attention_dim},{numOps/durationNs/1e12:.2f}")


phantom_configs = list(product(*parameters.values()))
# profile_batch_kernels(phantom_configs, 6)
extract_tflops(phantom_configs)