#! /bin/bash


## A simple script to run two flash attention kernels
## with batch2-nheads48-d64 on two GPUs in parallel
## $1: mode, fwd or bwd

if [[ $# -eq 0 ]];then
    echo "Must specify mode, fwd or bwd"
    exit
fi

TRITON_DIR=$(git rev-parse --show-toplevel)

BENCHMARK_DRIVER=${TRITON_DIR}/scripts/amd/benchmark_flash_attention.py

bs=2
nheads=48
d=64
mode=$1

declare -A repA

if [[ $mode == "fwd" ]];then
    repA[1024]=16000
    repA[2048]=8000
    repA[4096]=4000
    repA[8192]=2000
    repA[16384]=1000
else
    repA[1024]=10000
    repA[2048]=10000
    repA[4096]=2500
    repA[8192]=600
    repA[16384]=100
fi

for seqlen in 1024 2048  4096 8192 16384
do
    rep=${repA[$seqlen]}
    args="-bs $bs -nheads $nheads -d $d -seqlen $seqlen -rep $rep -mode $mode"

    start_time=$(date +%s.%3N)
    export ROCR_VISIBLE_DEVICES=0
    python ${BENCHMARK_DRIVER} $args &

    export ROCR_VISIBLE_DEVICES=1
    python ${BENCHMARK_DRIVER} $args

    wait
    end_time=$(date +%s.%3N)

    # elapsed time with millisecond resolution
    # keep three digits after floating point.
    elapsed=$(echo "scale=3; $end_time - $start_time" | bc)
    # Convert second to tflops
    if [[ $mode == "fwd" ]];then
        tflops=$(echo "scale=2; 8*$seqlen*$seqlen*$bs*$nheads*$d*$rep/$elapsed/1000000000000" | bc)
    else
        tflops=$(echo "scale=2; 7*4*0.5*$seqlen*$seqlen*$bs*$nheads*$d*$rep/$elapsed/1000000000000" | bc)
    fi
    echo "$seqlen  $tflops tflops $elapsed s"

done
