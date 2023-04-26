#! /bin/bash

## $1: input file
## $2: M
## $3: N
## $4: K


PROF_RESULT_FILE="results.stats.csv"

M=$2
N=$3
K=$4


## Tuning space for Triton
BLOCK_RANGE=(16 32 64)
SPLIT_K_RANGE=(2 4 8 16 18 24)
NUM_WARPS_RANGE=(1 2 4 8)


echo "Tuning GEMM for M=$M, N=$N, K=$K"

minTime=""
for BLOCK_M in ${BLOCK_RANGE[@]}
do
    ## Skip BLOCK_M if it is too large for M
    if [[ $M -le 16 ]] && [[ $BLOCK_M -ne 16 ]]; then
        continue
    fi
    for BLOCK_N in ${BLOCK_RANGE[@]}
    do
        ## Skip BLOCK_N if it is too large for N
        if [[ $N -le 16 ]] && [[ $BLOCK_N -ne 16 ]]; then
            continue
        fi
        for BLOCK_K in ${BLOCK_RANGE[@]}
        do
            for SPLIT_K in ${SPLIT_K_RANGE[@]}
            do
                for NUM_WARPS in ${NUM_WARPS_RANGE[@]}
                do
                    perfConfig="$BLOCK_M,$BLOCK_N,$BLOCK_K,$SPLIT_K,$NUM_WARPS"
                    ## rule out large num_warps
                    maxNumWarps=$((BLOCK_M*BLOCK_N/256))
                    if [[ $NUM_WARPS -gt $maxNumWarps ]]; then
                        continue
                    fi

                    Msg=$(rocprof --stats python $1 -m $M -n $N -k $K \
                                  -blockM ${BLOCK_M} -blockN ${BLOCK_N} -blockK ${BLOCK_K} \
                                  -num_warps ${NUM_WARPS} -splitK ${SPLIT_K})

                    if [[ $? != 0 ]]; then
                        time="NA"
                        continue
                    else
                        time=$(sed -n '/matmul_kernel/p' ${PROF_RESULT_FILE} \
                                   | awk -F ',' '{print $4}')
                        if [[ $minTime == "" ]] || [[ $time -lt $minTime ]];then
                            minTime=$time
                            bestPerfConfig=$perfConfig
                        fi
                    fi
                    echo "Checked $perfConfig  best parameters: $bestPerfConfig --> $minTime"
                done
            done
        done
    done
done

OUTPUT=triton_perf_config.txt
echo "$M,$N,$K  best parameters: $bestPerfConfig --> $minTime" | tee -a $OUTPUT
