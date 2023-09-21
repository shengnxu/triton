set -x

# bash scripts/amd/clean.sh

pip uninstall -y triton

export TRITON_USE_ROCM=ON
cd python
pip install --verbose -e .
