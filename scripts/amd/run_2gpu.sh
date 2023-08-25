#! /bin/bash


## A simple script to run two flash attention forward kernels
## with batch2-nheads48-d64 on two GPUs in parallel

bs=2
nheads=48
d=64

for seqlen in 1024 2048  4096 8192 16384
do
    rep=$(echo "163840000 / $seqlen" | bc)
    args="-bs $bs -nheads $nheads -d $d -seqlen $seqlen -rep $rep"

    start_time=$(date +%s.%3N)
    export ROCR_VISIBLE_DEVICES=0
    python ./benchmark_flash_attention.py $args &

    export ROCR_VISIBLE_DEVICES=1
    python ./benchmark_flash_attention.py $args

    wait
    end_time=$(date +%s.%3N)

    # elapsed time with millisecond resolution
    # keep three digits after floating point.
    elapsed=$(echo "scale=3; $end_time - $start_time" | bc)
    # Convert second to tflops
    tflops=$(echo "scale=2; 8*$seqlen*$seqlen*$bs*$nheads*$d*$rep/$elapsed/1000000000000" | bc)
    echo "$seqlen  $tflops"

done
