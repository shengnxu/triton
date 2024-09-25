#! /bin/bash
repo_root=$(git rev-parse --show-toplevel)
git config --global --add safe.directory "$repo_root"
TRITON_HIP_USE_NEW_STREAM_PIPELINE=1 python3 ./python/perf-kernels/streamk/tune_streamk.py --gemm_size_file ./python/perf-kernels/streamk/utils/streamk_unit_test_sizes.yaml --compare_wo_tuning
