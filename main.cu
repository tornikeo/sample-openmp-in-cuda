#include <stdio.h>
#include <cuda_runtime.h>

__global__ void helloCUDA()
{
    printf("%i %i Hello, CUDA!\n", threadIdx.x, blockIdx.x);
}

int main()
{
    helloCUDA<<<2, 2>>>();
    cudaDeviceSynchronize();
    return 0;
}