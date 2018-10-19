#!/bin/bash
# ./scripts/install.sh
# installs the correct version of pytorch (accounting for available GPUs and
# CUDA versions) in the current python environment (i.e., whatever pip maps to)
# make sure this is a python3.

set -euo pipefail

case "$OSTYPE" in
  linux*)   IS_MAC="false" ;;
  darwin*)  IS_MAC="true" ;; 
  *)        echo "os type $OSTYPE unsupported" ;;
esac

if [ "$IS_MAC" = true ] ; then
    echo "OK buddy, here's CPU-only pytorch for your mac"
    pip install torch
    exit 0
fi

PYTHON_VERSION="0.0.0"

if command -v python >/dev/null 2>&1 ; then
    PYTHON_VERSION=$(python --version 2>&1 | cut -d" " -f 2)
fi

echo "found python version $PYTHON_VERSION"

PYTHON_VERSION=$(echo "$PYTHON_VERSION" | cut -d. -f1-2)
PYTHON_VERSION="${PYTHON_VERSION//.}"

HAS_GPU="true"
if ! (lspci | grep -i nvidia 1>/dev/null 2>&1 ); then
    HAS_GPU="false"
fi

echo "found GPU? $HAS_GPU"

TORCH_VERSION="0.4.0"
if [ "$HAS_GPU" = true ] ; then
    NVCC_LOCATION=""
    if [ -d /usr/local/cuda-9.1 ] ; then
        # RISE machines have non-default newer cuda available, check it here
        NVCC_LOCATION=/usr/local/cuda-9.1/bin/nvcc
    elif command -v nvcc >/dev/null 2>&1 ; then
        NVCC_LOCATION=$(command -v nvcc)
    elif [ -d /usr/local/cuda ] ; then
        NVCC_LOCATION=/usr/local/cuda/bin/nvcc
    else
        "could not find nvcc, make sure it's in your PATH"
        exit 1
    fi
    CUDA_LOCATION=$(dirname $(dirname "$NVCC_LOCATION"))
    CUDA_VERSION=$(cut -d" " -f3 "$CUDA_LOCATION/version.txt" | cut -d. -f1-2)
    echo "found cuda version $CUDA_VERSION"
    CUDA_VERSION="${CUDA_VERSION//.}"

    PYTORCH_INSTALL="http://download.pytorch.org/whl/cu${CUDA_VERSION}/torch-${TORCH_VERSION}-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}m-linux_x86_64.whl"
else
    PYTORCH_INSTALL="http://download.pytorch.org/whl/cpu/torch-${TORCH_VERSION}-cp${PYTHON_VERSION}-cp${PYTHON_VERSION}m-linux_x86_64.whl"
fi

echo "installing pytorch"

pip install --force-reinstall --no-cache-dir "$PYTORCH_INSTALL"
