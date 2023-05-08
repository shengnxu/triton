#! /bin/bash

## $1: input file
## $2: M
## $3: N
## $4: K
## $5: 1: reduced tuning space

if [[ $# -lt 5 ]];then
    echo "Usage: ./prof.sh <driver program> M N K <reduceTuningSpace>"
    exit
fi

PROF_RESULT_FILE="results.stats.csv"

DRIVER=$1
M=$2
N=$3
K=$4
reduceSpace=$5

if [[ $reduceSpace -eq 0 ]];then
    ## Tuning space for Triton
    BLOCK_RANGE=(16 32 64 128)
    SPLIT_K_RANGE=(2 4 6 8 10 12 14 16 18 20 22 24)
    NUM_WARPS_RANGE=(1 2 4 8)
    ## Tuning space for rocMLIR
    KPACK_RANGE=(1 2 4 8)
    mPerWave_RANGE=(16 32 64)
else ## Reduced tuning space
    ## Tuning space for Triton
    BLOCK_RANGE=(16 32 64)
    SPLIT_K_RANGE=(2 8 12 18 24)
    NUM_WARPS_RANGE=(1 2 4 8)
    ## Tuning space for rocMLIR
    KPACK_RANGE=(4 8)
    mPerWave_RANGE=(16 32)
fi

echo "Tuning GEMM for M=$M, N=$N, K=$K"

minTime=""
##################################
## Looping BLOCK_M              ##
##################################
for BLOCK_M in ${BLOCK_RANGE[@]}
do
    ## Skip BLOCK_M if it is too large for M
    if [[ $M -le 16 ]] && [[ $BLOCK_M -ne 16 ]]; then
        continue
    fi
    ##################################
    ## Looping BLOCK_N              ##
    ##################################
    for BLOCK_N in ${BLOCK_RANGE[@]}
    do
        ## Skip BLOCK_N if it is too large for N
        if [[ $N -le 16 ]] && [[ $BLOCK_N -ne 16 ]]; then
            continue
        fi
        ##################################
        ## Looping BLOCK_K              ##
        ##################################
        for BLOCK_K in ${BLOCK_RANGE[@]}
        do
            ##################################
            ## Looping SPLIT_K              ##
            ##################################
            for SPLIT_K in ${SPLIT_K_RANGE[@]}
            do
                ## Skip SPLIT_K if K % (SPLIT_K * BLOCK_K) != 0
                leap=$((SPLIT_K * BLOCK_K))
                mod=$((K%leap))
                if [[ $mod -ne 0 ]]; then
                    continue
                fi
                ##################################
                ## Looping mPerWave            ##
                ##################################
                for mPerWave in ${mPerWave_RANGE[@]}
                do
                    ## Skip mPerWave if its larger than BLOCK_M
                    if [[ $mPerWave -gt $BLOCK_M ]]; then
                        continue
                    fi
                    ##################################
                    ## Looping num_warps            ##
                    ##################################
                    for num_warps in ${NUM_WARPS_RANGE[@]}
                    do
                        ## Skip large num_warps
                        maxNumWarps=$((BLOCK_M*BLOCK_N/16/mPerWave))
                        if [[ $num_warps -gt $maxNumWarps ]]; then
                            continue
                        fi
                        ##################################
                        ## Looping kpack                ##
                        ##################################
                        for kpack in ${KPACK_RANGE[@]}
                        do
                            perfConfig="$BLOCK_M,$BLOCK_N,$BLOCK_K,$SPLIT_K,$mPerWave,$kpack,$num_warps"
                            rm -rf ~/.triton/cache
                            Msg=$(rocprof --stats python $DRIVER -m $M -n $N -k $K \
                                          -blockM ${BLOCK_M} -blockN ${BLOCK_N} -blockK ${BLOCK_K} \
                                          -num_warps ${num_warps} -splitK ${SPLIT_K}\
                                          -kpack $kpack -mPerWave $mPerWave)

                            ## Skip if there is an error (invalid kpack)
                            if [[ $? != 0 ]]; then
                                echo "Checked $perfConfig  invalid (kpack, BLOCK_K)"
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
    done
done

echo "Best Result: $M,$N,$K  best parameters: $bestPerfConfig --> $minTime"
