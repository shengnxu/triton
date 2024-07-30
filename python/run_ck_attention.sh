#!/bin/bash

# Function to display help message
show_help() {
  echo "Usage: $0 -o <output_file> -gpu <gpu1> [<gpu2> ... <gpuN>]"
  echo ""
  echo "Arguments:"
  echo "  -o <output_file>   The file to which output will be logged."
  echo "  -gpu               Specify the GPUs to use, followed by GPU IDs."
  echo ""
  echo "Example:"
  echo "  $0 -o output_ck_flash_attention_3.log -gpu 0 1 2 3"
}

# Check for help option
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  show_help
  exit 0
fi

# Check if the correct number of arguments is provided
if [ "$#" -lt 4 ]; then
  show_help
  exit 1
fi

# Initialize variables
output_file=""
gpus=()

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o)
      output_file="$2"
      shift 2
      ;;
    -gpu)
      shift
      while [[ $# -gt 0 && "$1" != -* ]]; do
        gpus+=("$1")
        shift
      done
      ;;
    *)
      show_help
      exit 1
      ;;
  esac
done

# Check if output file and GPUs are provided
if [ -z "$output_file" ] || [ ${#gpus[@]} -eq 0 ]; then
  show_help
  exit 1
fi

batch_sizes=(2 4 8 16)
num_heads_q=(2 4 8 16)
num_heads_k=(2 4 8 16)
seqlen_q=(256 512 1024 2048 4096 8192 16384 32768 65536)
seqlen_k=(256 512 1024 2048 4096 8192 16384 32768 65536)
head_dim_q_k=(32 64 128)
head_dim_v=(32 64 128)
attn_mask=(0 1 2)

# Number of GPUs available
num_gpus=${#gpus[@]}

# Function to run the program with given parameters and GPU
run_program() {
  local gpu_id=$1
  shift
  ROCR_VISIBLE_DEVICES=$gpu_id /composable_kernel/build/bin/tile_example_fmha_fwd \
    -b=$1 \
    -h=$2 \
    -h_k=$3 \
    -s=$4 \
    -s_k=$5 \
    -d=$6 \
    -d_v=$7 \
    -mask=$8 \
    -kname=1 \
    -repeat=3 >> "$output_file" 2>&1
  echo -e "\n" >> "$output_file"
}

# Export the function so it can be used by parallel
export -f run_program
export output_file

# Generate the parameter combinations
combinations=()
for b in "${batch_sizes[@]}"; do
  for h in "${num_heads_q[@]}"; do
    for h_k in "${num_heads_k[@]}"; do
      for s in "${seqlen_q[@]}"; do
        for s_k in "${seqlen_k[@]}"; do
          for d in "${head_dim_q_k[@]}"; do
            for d_v in "${head_dim_v[@]}"; do
              for mask in "${attn_mask[@]}"; do
                combinations+=("$b $h $h_k $s $s_k $d $d_v $mask")
              done
            done
          done
        done
      done
    done
  done
done

# Use GNU parallel to run the jobs
printf "%s\n" "${combinations[@]}" | parallel -j "$num_gpus" run_program {=$(( {} % num_gpus ))} ${gpus[{} % num_gpus]}
