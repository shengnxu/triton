#!/bin/bash

# Function to display help message
show_help() {
  echo "Usage: $0 -o <output_file>"
  echo ""
  echo "Arguments:"
  echo "  -o <output_file>   The file to which output will be logged."
  echo ""
  echo "Example:"
  echo "  $0 -o output_ck_flash_attention_3.log"
}

# Check for help option
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
  show_help
  exit 0
fi

# Initialize variables
output_file=""

# Parse arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    -o)
      output_file="$2"
      shift 2
      ;;
    *)
      show_help
      exit 1
      ;;
  esac
done

# Check if output file is provided
if [ -z "$output_file" ]; then
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

# Function to run the program with given parameters
run_program() {
  /composable_kernel/build/bin/tile_example_fmha_fwd \
    -b=$1 \
    -h=$2 \
    -h_k=$3 \
    -s=$4 \
    -s_k=$5 \
    -d=$6 \
    -d_v=$7 \
    -mask=$8 \
    -kname=1 \
    -repeat=3
}

# Iterate over the arrays and run the program with each combination of parameters
for b in "${batch_sizes[@]}"; do
  for h in "${num_heads_q[@]}"; do
    for h_k in "${num_heads_k[@]}"; do
      for s in "${seqlen_q[@]}"; do
        for s_k in "${seqlen_k[@]}"; do
          for d in "${head_dim_q_k[@]}"; do
            for d_v in "${head_dim_v[@]}"; do
              for mask in "${attn_mask[@]}"; do
                run_program $b $h $h_k $s $s_k $d $d_v $mask >> "$output_file" 2>&1
                echo -e "\n" >> "$output_file"
              done
            done
          done
        done
      done
    done
  done
done
