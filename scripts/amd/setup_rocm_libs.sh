#!/usr/bin/env bash

set -ex

CWD=$(dirname "$0")

echo "Building libdrm"
$CWD/drm.sh

echo "Collecting Necessary ROCM libraries"
if [[ -z "${ROCM_VERSION}" ]]; then
  ROCM_VERSION=5.4.2
fi
if [[ -z "${TRITON_ROCM_DIR}" ]]; then
  TRITON_ROCM_DIR=python/triton/third_party/rocm
fi
  
save_IFS="$IFS"
IFS=. ROCM_VERSION_ARRAY=(${ROCM_VERSION})
IFS="$save_IFS"
if [[ ${#ROCM_VERSION_ARRAY[@]} == 2 ]]; then
    ROCM_VERSION_MAJOR=${ROCM_VERSION_ARRAY[0]}
    ROCM_VERSION_MINOR=${ROCM_VERSION_ARRAY[1]}
    ROCM_VERSION_PATCH=0
elif [[ ${#ROCM_VERSION_ARRAY[@]} == 3 ]]; then
    ROCM_VERSION_MAJOR=${ROCM_VERSION_ARRAY[0]}
    ROCM_VERSION_MINOR=${ROCM_VERSION_ARRAY[1]}
    ROCM_VERSION_PATCH=${ROCM_VERSION_ARRAY[2]}
else
    echo "Unhandled ROCM_VERSION ${ROCM_VERSION}"
    exit 1
fi

ROCM_INT=$(($ROCM_VERSION_MAJOR * 10000 + $ROCM_VERSION_MINOR * 100 + $ROCM_VERSION_PATCH))
OS_NAME=`awk -F= '/^NAME/{print $2}' /etc/os-release`

if [[ $ROCM_INT -eq 50402 ]]; then
    if [[ "$OS_NAME" == *"CentOS Linux"* ]]; then
        ROCM_PKGS=(
            "https://repo.radeon.com/rocm/rhel8/5.4.2/main/hsa-rocr5.4.2-1.7.0.50402-104.el8.x86_64.rpm"
            "https://repo.radeon.com/rocm/rhel8/5.4.2/main/hip-runtime-amd-5.4.22803.50402-104.el8.x86_64.rpm"
            "https://repo.radeon.com/rocm/rhel8/5.4.2/main/hip-devel-5.4.22803.50402-104.el8.x86_64.rpm"
            "https://repo.radeon.com/rocm/rhel8/5.4.2/main/rocm-llvm5.4.2-15.0.0.22506.50402-104.el8.x86_64.rpm"
            "https://repo.radeon.com/rocm/rhel8/5.4.2/main/rocminfo5.4.2-1.0.0.50402-104.el8.x86_64.rpm"
        )

    elif [[ "$OS_NAME" == *"Ubuntu"* ]]; then
        ROCM_PKGS=(
            "https://repo.radeon.com/rocm/apt/5.4.2/pool/main/h/hsa-rocr/hsa-rocr_1.7.0.50402-104~22.04_amd64.deb"
            "https://repo.radeon.com/rocm/apt/5.4.2/pool/main/h/hip-runtime-amd5.4.2/hip-runtime-amd5.4.2_5.4.22803.50402-104~22.04_amd64.deb"
            "https://repo.radeon.com/rocm/apt/5.4.2/pool/main/h/hip-dev5.4.2/hip-dev5.4.2_5.4.22803.50402-104~22.04_amd64.deb"
            "https://repo.radeon.com/rocm/apt/5.4.2/pool/main/r/rocm-llvm5.4.2/rocm-llvm5.4.2_15.0.0.22506.50402-104~22.04_amd64.deb"
            "https://repo.radeon.com/rocm/apt/5.4.2/pool/main/r/rocminfo5.4.2/rocminfo5.4.2_1.0.0.50402-104~22.04_amd64.deb"
        )
    fi
fi

EXTRACT_DIR=vanilla_extract
mkdir -p $EXTRACT_DIR
if [[ "$OS_NAME" == *"CentOS Linux"* ]]; then
    OS_SO=(
        "/usr/lib64/libnuma.so.1"
        "/usr/lib64/libelf.so.1"
    )
    #install elf and numa packages if not already done
    yum install -y elfutils-libelf numactl-libs
    
    #download and extract ROCM packages
    pushd $EXTRACT_DIR
    for pkg  in "${ROCM_PKGS[@]}"
    do
        curl $pkg --output temp.rpm
        rpm2cpio temp.rpm | cpio -idm
        rm temp.rpm
    done
    popd
elif [[ "$OS_NAME" == *"Ubuntu"* ]]; then
    OS_SO=(
        "/usr/lib/x86_64-linux-gnu/libnuma.so.1"
        "/usr/lib/x86_64-linux-gnu/libelf.so.1"
    )
    #install elf and numa packages if not already done
    apt-get update -y
    apt-get install -y libelf1 libnuma1 curl
    apt-get clean

    #download and extract ROCM packages
    for pkg  in "${ROCM_PKGS[@]}"
    do
        curl $pkg --output temp.deb
        dpkg -x temp.deb $EXTRACT_DIR
        rm temp.deb
    done
else
    echo "ERROR: $OS_NAME is not a supported Operating System"
fi

# Required ROCm libraries from extracted packages
ROCM_FILES=(
    "lib/libhsa-runtime64.so"
    "lib/libamdhip64.so"
    "bin/rocminfo"
    "llvm/bin/ld.lld"
    "llvm/bin/lld"
    "include/hip"
)
for lib in "${ROCM_FILES[@]}"
do
    mkdir -p `dirname $TRITON_ROCM_DIR/$lib`
    cp -r $EXTRACT_DIR/opt/rocm-$ROCM_VERSION/$lib* `dirname $TRITON_ROCM_DIR/$lib`
done

for lib in "${OS_SO[@]}"
do
    cp $lib $TRITON_ROCM_DIR/lib/
done

rm -rf $EXTRACT_DIR

