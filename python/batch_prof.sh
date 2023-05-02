#! /bin/bash

##
## Usage:
##   ./batchprof.sh CONFIG_FILE OUTPUT
## This script runs the tuning script for each gemm config in CONFIG_FILE.
## Configs will be splitted on multiple GPUs on the system.
## The results are collected in OUTPUT. If OUTPUT is not provided,
## triton_prof_results.txt will be used
##
## This script is copied and modified from 20221122_Mathews_MIOpen_tuning.pdf.

set -e
set -x

## $1: config file
##     Each line in the config file must be dims for M, N, and K
##     separated by space
## $2: output file

PrWD=$(pwd)

if [[ $# -lt 1 ]];then
    echo "Usage: ./batch_prof.sh <config_file>"
    exit
fi

CONFIG_FILE=$1
OUTPUT=$2
if [[ $OUTPUT == "" ]];then
    OUTPUT="${PrWD}/triton_prof_results.txt"
fi

DRIVER="${PrWD}/test/unit/language/matmul.py"

DATESTR=$(date +'%Y%m%d_%H%M%S')
RANDSTR=$RANDOM

mkdir triton_tuning_${DATESTR}_${RANDSTR}
pushd triton_tuning_${DATESTR}_${RANDSTR}

shuf $PrWD/$CONFIG_FILE &> unique-configs-shuffled.log

NGPUS=$(rocm-smi --showgpu | grep -Eo "GPU\[[0-9]*\]" | sed -e 's/\]//g' | wc -l)

split -n l/$NGPUS unique-configs-shuffled.log configs_split
split_files=($(ls configs_split*))

for (( gpu=0; gpu<$NGPUS; gpu++ ));
do
    ## Add the profiler and driver program
    sed -i "s#^#${PrWD}/tritonProfiler.sh $DRIVER #"  ${split_files[$gpu]}
    ## Use the whole tuning space
    sed -i "s/$/ 0/"  ${split_files[$gpu]}
    sed -i "1s/^/export ROCM_VISIBLE_DEVICES=$gpu\n/" ${split_files[$gpu]}
done
wait

# launch the training tasks concurrently
for (( gpu=0; gpu<$NGPUS; gpu++ ));
do
    bash ${split_files[$gpu]} &> ${split_files[$gpu]}.log &
done
wait

# collect results
for (( gpu=0; gpu<$NGPUS; gpu++ ));
do
    sed -n "/Best Result/p" ${split_files[$gpu]}.log | tee -a $OUTPUT
done

popd

echo "Tuned gemm configs are saved in: $OUTPUT"
