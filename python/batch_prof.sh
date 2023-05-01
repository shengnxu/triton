#! /bin/bash

## $1: config file
##     Each line in the config file must be dims for N, M, and K
##     separated by space
## $2: reduced tuning space
## $3: output file

if [[ $# -lt 2 ]];then
    echo "Usage: ./batch_prof.sh <config_file> <reducedTuningSpace>"
    exit
fi

OUTPUT=$3
if [[ $OUTPUT == "" ]];then
    OUTPUT="triton_prof_results.txt"
fi


DRIVER="./test/unit/language/matmul.py"

CONFIG_FILE=$1
reduceSpace=$2

while IFS= read -r line
do
    echo "config: $line"
    gemm_config=($line)
    N=${gemm_config[0]}
    M=${gemm_config[1]}
    K=${gemm_config[2]}
    results=$(./tritonProfiler.sh $DRIVER $M $N $K $reduceSpace)
    bestPerf=${results##*$'\n'}
    echo "$bestPerf" | tee -a $OUTPUT
done < "$CONFIG_FILE"
