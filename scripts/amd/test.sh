#!/bin/bash

# clear
set -x

# export MLIR_ENABLE_DUMP=1
# export LLVM_IR_ENABLE_DUMP=1
# export AMDGCN_ENABLE_DUMP=1


# log dir
ROOT_DIR=$(pwd)
LOG_DIR=$ROOT_DIR/log
rm -rf $LOG_DIR
mkdir -p $LOG_DIR
chmod -R 777 $LOG_DIR

# sh scripts/amd/clean.sh

# UNIT_TEST="python/test/unit/language/test_core_amd.py"
UNIT_TEST="python/test/unit/language/test_core_amd.py::test_atomic_cas[1-None]"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_gemm[4-16-128-4-4-16-64-1]"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_dot"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_dot[32-128-64-2-False-False-none-True-float16-float16-0]"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_dot[16-16-16-4-False-False-none-False-float16-float16-0]"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_make_range[dst_layout0-src_layout0-float16-shape0]"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_empty_kernel"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_empty_kernel[float32]"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_gemm_fp816_mixed_inputs"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_gemm_fp816_mixed_inputs[32-128-64-a_type37-b_type37-out_dtype37]"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_make_range"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_program_functions"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_bin_op"
# UNIT_TEST="python/test/unit/language/test_core.py::test_bin_op[1-float32-float32-+]"
# UNIT_TEST="python/test/unit/runtime/test_cache.py::test_compile_in_subproc"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_shift_op"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_shift_op[int8-int8-<<]"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_shift_op[int32-int32->>]"
# UNIT_TEST="python/test/unit/language/test_core.py::test_bin_op"
# UNIT_TEST="python/test/unit/language/test_core.py::test_bin_op[float32-float32-+]"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_bin_op[int8-float16-%]"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_masked_load_shared_memory[dtype0]"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_masked_load_shared_memory[dtype1]"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_bin_op_constexpr"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_bin_op_constexpr[True-True-<<]" # works
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_bin_op_constexpr[False-False-<<]"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_bin_op_constexpr[False-False->>]"
# UNIT_TEST="python/test/unit/language/test_core_amd.py::test_bin_op_constexpr[False-True-<<]"

# check for backtrace
if [ "$1" == "backtrace" ]; then
	sudo apt install gdb -y

	COMMAND="-m pytest --capture=tee-sys --verbose $UNIT_TEST"
	gdb python \
		-ex "set pagination off" \
		-ex "run $COMMAND" \
		-ex "backtrace" \
		-ex "set confirm off" \
		-ex "q" \
		2>&1 | tee $LOG_DIR/backtrace.log

else
	pytest --capture=tee-sys -rfs --verbose "$UNIT_TEST" 2>&1 | tee $LOG_DIR/unit_test.log
fi

bash scripts/amd/cache_print.sh  2>&1 |tee $LOG_DIR/cache.log
