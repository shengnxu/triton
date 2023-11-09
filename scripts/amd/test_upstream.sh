set -x

SCRIPT_PATH="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"

git config --global --add safe.directory /dockerx/triton_rocm

# get current branch info
BRANCH_NAME=$(git branch --show-current)
BRANCH_COMMIT_HASH=$(git rev-parse HEAD)
BRANCH_COMMIT_MSG=$(git log -1 --pretty=%B)
echo $BRANCH_NAME
echo $BRANCH_COMMIT_HASH
echo $BRANCH_COMMIT_MSG

# get the last commit that AMD's fork shares with upstream triton
pip install gitpython
output=$(python3 $SCRIPT_PATH/diff_upstream.py --upstream https://github.com/openai/triton --fork https://github.com/ROCmSoftwarePlatform/triton)
upstreamcommit_hash_line=$(echo "$output" | grep 'SHARED_COMMIT_HASH=')
UPSTREAM_COMMIT_HASH=$(echo "$upstreamcommit_hash_line" | cut -d '=' -f2)
echo $UPSTREAM_COMMIT_HASH

# check out the upstream at shared last commit
UPSTREAM_REPO_DIR=/tmp/triton_upstream
rm -rf $UPSTREAM_REPO_DIR
git clone --recurse-submodules https://github.com/openai/triton $UPSTREAM_REPO_DIR
cd $UPSTREAM_REPO_DIR
git checkout $UPSTREAM_COMMIT_HASH
git apply $SCRIPT_PATH/new_backend.patch # apply patch until upstreamed

# checkout backend in upstream
cd $UPSTREAM_REPO_DIR/third_party/amd_hip_backend
git fetch --all
git checkout $BRANCH_NAME

# build upstream with this branch as a backend
cd $UPSTREAM_REPO_DIR
cd $UPSTREAM_REPO_DIR/python
pip uninstall -y triton
# pip install -U matplotlib pandas filelock tabulate

echo `pwd`
export TRITON_CODEGEN_AMD_HIP_BACKEND=1
pip install --verbose -e .


python3 $SCRIPT_PATH/check_triton.py


cd $UPSTREAM_REPO_DIR
# pytest -n 32 --capture=tee-sys -rfs --verbose "python/test/unit/language/test_core.py"
pytest -n 32 --capture=tee-sys -rfs --verbose "python/test/unit/language/test_core.py::test_empty_kernel[float32]"
