set -x

cd python
pip uninstall -y triton

# bash scripts/amd/clean.sh

export TRITON_USE_ROCM=ON

# pip install -U matplotlib pandas filelock tabulate
pip install --verbose -e .
