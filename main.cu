#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <iostream>
#include <cassert>
#include <vector>
#include <numeric> // std::accumulate
#include "utils.h"

#define N 8    // Number of threads (equivalent to MPI processes)
#define L 32 // Length of each array

// CUDA Kernel: Sums each L-sized array into an N-sized result
__global__ void sumKernel(int *d_input, int *d_output)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N)
    {
        int sum = 0;
        for (int i = 0; i < L; i++)
        {
            sum += d_input[idx * L + i];
        }

        d_output[idx] = sum;
    }
    // printf("%i Hello from kernel!", idx);
}

void printVector(const std::vector<int> &vec, size_t max_display = 50)
{
    std::cout << "[";
    if (vec.size() <= max_display)
    {
        for (size_t i = 0; i < vec.size(); ++i)
        {
            std::cout << vec[i];
            if (i != vec.size() - 1)
            {
                std::cout << ", ";
            }
        }
    }
    else
    {
        for (size_t i = 0; i < max_display / 2; ++i)
        {
            std::cout << vec[i] << ", ";
        }
        std::cout << "...";
        for (size_t i = vec.size() - max_display / 2; i < vec.size(); ++i)
        {
            std::cout << ", " << vec[i];
        }
    }
    std::cout << "]" << std::endl;
}

int main()
{
    std::vector<int> all_data(N * L); // Shared memory buffer for all threads
    std::vector<int> results(N, 0);   // Stores the reduced sums

    printf("Start\n");

    is_available();

// Step 1: Parallel region using OpenMP
#pragma omp parallel num_threads(N)
    {
        int thread_id = omp_get_thread_num();
        printf("Thread %i is working...\n", thread_id);
        int start_idx = thread_id * L;
        for (int i = 0; i < L; i++)
        {
            all_data[start_idx + i] = thread_id; // Fill array with thread ID
        }
    }

    printf("Numbers going into GPU:\n");
    for (size_t i = 0; i < N; i++)
    {
        for (size_t j = 0; j < L; j++)
        {
            printf("%i ", all_data[i*L + j]);
        }
        printf("\n");
    }

    // Step 2: Allocate GPU Memory
    int *d_input, *d_output;
    cudaMalloc(&d_input, N * L * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));

    // Step 3: Copy data to GPU
    cudaMemcpy(d_input, all_data.data(), N * L * sizeof(int), cudaMemcpyHostToDevice);

    // Step 4: Launch Kernel (1 block, N threads)
    printf("Launching GPU...\n");
    sumKernel<<<1, N>>>(d_input, d_output);
    cudaDeviceSynchronize();

    // Step 5: Copy results back
    cudaMemcpy(results.data(), d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Step 6: Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);
    printf("We expect the kernel to sum the values in each row...\n");

// Step 7: Each thread verifies and prints the result
#pragma omp parallel num_threads(N)
    {

#pragma omp critical
        {
            int thread_id = omp_get_thread_num();
            int expected = thread_id * L;
            if (results[thread_id] == expected)
            {
                std::cout << "Thread " << thread_id << " verified sum: "
                        << results[thread_id] << " ✅" << std::endl;
            }
            else
            {
                std::cerr << "Thread " << thread_id << " ERROR! Expected "
                        << expected << ", but got " << results[thread_id] << std::endl;
            }
        }
    }
    return 0;
}
