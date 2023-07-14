#!/bin/bash
export ROCM_VISIBLE_DEVICES=1
# set -e
set -x

if [[ $# -lt 1 ]]; then
  echo "Usage: run.sh config_file output_file iter_num"
  exit
fi

config_file=$1
out_file=$2
iter_num=$3

# default output file
if [ -z $out_file ]; then
  out_file='outputs'
fi

# default to run once
if [ -z $iter_num ]; then
  iter_num=1
fi

PROF_RESULT_FILE="results.stats.csv"

while read line; do
  echo $line >> ${out_file}
  # comment line, skip
  if [[ $line == "#"* ]]; then
    echo "Comment line, skip"
    continue
  fi

  # empty line, skip
  if [ -z $line ]; then
    echo "Empty line, skip"
    continue;
  fi

  IFS=' ' read -a line_info <<< $line
  # matrix size
  sizes=${line_info[2]}
  IFS=',' read -a dims <<< $sizes
  M=${dims[0]}
  N=${dims[1]}
  K=${dims[2]}
  
  #configs
  config=${line_info[5]}
  IFS=',' read -a params <<< $config
  BLOCK_M=${params[0]}
  BLOCK_N=${params[1]}
  BLOCK_K=${params[2]}
  SPLIT_K=${params[3]}
  num_warps=${params[4]}
  driver='verify_gemm_perf.py'

  # command to run matmul
  run_cmd="python ${driver} \
          -m ${M} -n ${N} -k ${K} -blockM ${BLOCK_M} \
          -blockN ${BLOCK_N} -blockK ${BLOCK_K} \
          -splitK ${SPLIT_K} -num_warps ${num_warps}"

  prof_cmd="rocprof --stats ${run_cmd}"
  echo ${prof_cmd}
  # warm up for once
  for ((i=0; i<1; i++)); do
    msg=$(${run_cmd})
  done

  # repeatedly run the test to get perf numbers    
  for ((i=0; i<$iter_num; i++)); do
    msg=$(${prof_cmd})

    ## skip if there is an error (invalid kpack)
    if [[ $? != 0 ]]; then
      echo "Command \'${prof_cmd}\' failed!".
      continue
    else
      t=$(sed -n '/matmul_kernel/p' ${PROF_RESULT_FILE} | awk -F ',' '{print $4}')
      echo ${t} >> ${out_file}
    fi
  done
done < $config_file
