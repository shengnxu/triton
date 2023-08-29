#! /bin/bash

## A simple script to run two flash attention backward kernels
## Adapted from https://github.com/ROCmSoftwarePlatform/triton/blob/fa_fwd_benchmark_2gpus/scripts/amd/run_2gcd.sh

TRITON_DIR=$(git rev-parse --show-toplevel)

echo $TRITON_DIR
BENCHMARK_DRIVER=${TRITON_DIR}/scripts/amd/benchmark_flash_attention_bwd.py

bs=2
nheads=48
d=64
rep=10000

# 10k reps required to amortize overhead for 1k but too long for 16k. Reduce accordingly
#for seqlen in 1024 2048 4096 8192 16384
for seqlen in 1024
do
    #rep=$(echo "163840000 / $seqlen" | bc)
    args="-bs $bs -nheads $nheads -d $d -seqlen $seqlen -rep $rep"

    start_time=$(date +%s.%3N)
    export ROCR_VISIBLE_DEVICES=14
    python ${BENCHMARK_DRIVER} $args &

    export ROCR_VISIBLE_DEVICES=15
    python ${BENCHMARK_DRIVER} $args

    wait
    end_time=$(date +%s.%3N)

    # elapsed time with millisecond resolution
    # keep three digits after floating point.
    elapsed=$(echo "scale=3; $end_time - $start_time" | bc)
    # Convert second to tflops; 2 fwd + 5 bwd GEMMs, 2 macs/op, 2 gcds, 0.5 causal
    tflops=$(echo "scale=2; 7*4*0.5*$seqlen*$seqlen*$bs*$nheads*$d*$rep/$elapsed/1000000000000" | bc)
    echo "$seqlen  $tflops  $elapsed"

done