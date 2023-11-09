set -x

# clean exisitng code
# rm -rf python/triton/third_party/hip

pip3 install tree_sitter gitpython chardet tqdm

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"


python $SCRIPT_PATH/gen_hip_backend.py
# python $SCRIPT_PATH/gen_hip_backend.py --filter .py #--debug
# python $SCRIPT_PATH/gen_hip_backend.py --verbose --chmod 777  --path lib/Conversion/TritonGPUToLLVM/TritonGPUToLLVM.h #--line 9 --range 1 --debug --map $SCRIPT_PATH/triton.json  #--verbose #--debug # --visualize
# python $SCRIPT_PATH/gen_hip_backend.py --map $SCRIPT_PATH/triton.json #--line 640 --range 2 #--verbose #--debug # --visualize