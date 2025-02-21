#!/bin/bash
set -e  # Exit on error

# Ensure CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "❌ ERROR: nvcc not found! Make sure CUDA is installed and in your PATH."
    exit 1
fi

echo "✅ Building CUDA Kernel..."
nvcc -ccbin gcc-9 -c cuda_kernel.cu -o cuda_kernel.o

echo "✅ Building Vulkan-CUDA Interop C Program..."
gcc -o cuda_vulkan_test cuda_vulkan_test.c cuda_kernel.o -lvulkan -lcuda -lcudart -fopenmp -lstdc++  # ✅ Fix: Add -lstdc++

echo "✅ Running Test..."
./cuda_vulkan_test
