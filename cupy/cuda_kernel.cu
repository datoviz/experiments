#include <cuda_runtime.h>
#include <stdio.h>

// CUDA Kernel: Multiply each element by 2
__global__ void multiply_by_two(float* data, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size)
    {
        data[idx] *= 2.0f;
    }
}

// Function callable from C
extern "C" void launch_cuda_kernel(float* data, int size)
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    multiply_by_two<<<blocksPerGrid, threadsPerBlock>>>(data, size);
    cudaDeviceSynchronize();
}
