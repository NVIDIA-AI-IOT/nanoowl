#!/bin/bash
set -e

echo "Starting the setup process..."

# Check for NVIDIA drivers
echo "Checking for NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null
then
    echo "WARNING: nvidia-smi command not found. Please ensure NVIDIA drivers are installed."
    exit 1
else
    echo "NVIDIA drivers found."
    nvidia-smi | grep "CUDA Version"
fi

# Check for CUDA toolkit (nvcc) and version
echo "Checking for CUDA Toolkit (nvcc)..."
if ! command -v nvcc &> /dev/null
then
    echo "WARNING: nvcc command not found. Please ensure CUDA Toolkit is installed and in your PATH."
    exit 1
else
    CUDA_VERSION_OUTPUT=$(nvcc --version)
    echo "CUDA Toolkit found:"
    echo "$CUDA_VERSION_OUTPUT"
    if ! echo "$CUDA_VERSION_OUTPUT" | grep -q "release 12.1"
    then
        echo "WARNING: CUDA version is not 12.1. This script is intended for CUDA 12.1."
        echo "Please ensure you have CUDA 12.1 installed, or modify the script if you intend to use a different version."
        # Consider exiting here if strict 12.1 compliance is required, or just warn.
        # For now, let's warn and continue, but an exit might be safer in a real scenario.
        # exit 1 
    else
        echo "CUDA version 12.1 confirmed."
    fi
fi

echo "Updating package lists..."
sudo apt-get update

echo "Installing essential packages (git, python3-pip, cmake)..."
sudo apt-get install -y git python3-pip cmake

echo "Upgrading pip and installing Python build/packaging tools..."
python3 -m pip install --upgrade pip wheel setuptools packaging

echo "Installing NVIDIA TensorRT (for CUDA 12.x via pip)..."
python3 -m pip install --upgrade tensorrt

echo "Starting torch2trt installation..."
if [ -d "torch2trt" ]; then
    echo "torch2trt directory already exists. Removing it to start fresh."
    rm -rf torch2trt
fi
echo "Cloning the torch2trt repository..."
git clone https://github.com/NVIDIA-AI-IOT/torch2trt

echo "Changing into the torch2trt directory..."
cd torch2trt

echo "Installing torch2trt with plugins..."
python3 setup.py install --plugins

echo "Changing back to the original directory..."
cd ..

echo "Setup successfully completed!"
echo "Please ensure your environment (like LD_LIBRARY_PATH) is correctly configured if needed, especially for torch2trt plugins."
