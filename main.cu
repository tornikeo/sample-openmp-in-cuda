#include <cuda_runtime.h>
#include <omp.h>
#include <iostream>
#include <vector>
#include <numeric>  // std::accumulate

#define N 4    // Number of threads (equivalent to MPI processes)
#define L 1024 // Length of each array

// CUDA Kernel: Sums each L-sized array into an N-sized result
__global__ void sumKernel(int *d_input, int *d_output) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int sum = 0;
        for (int i = 0; i < L; i++) {
            sum += d_input[idx * L + i];
        }
        d_output[idx] = sum;
    }
}

int main() {
    std::vector<int> all_data(N * L); // Shared memory buffer for all threads
    std::vector<int> results(N, 0);   // Stores the reduced sums

    // Step 1: Parallel region using OpenMP
    #pragma omp parallel num_threads(N)
    {
        int thread_id = omp_get_thread_num();
        int start_idx = thread_id * L;
        for (int i = 0; i < L; i++) {
            all_data[start_idx + i] = thread_id;  // Fill array with thread ID
        }
    }

    // Step 2: Allocate GPU Memory
    int *d_input, *d_output;
    cudaMalloc(&d_input, N * L * sizeof(int));
    cudaMalloc(&d_output, N * sizeof(int));

    // Step 3: Copy data to GPU
    cudaMemcpy(d_input, all_data.data(), N * L * sizeof(int), cudaMemcpyHostToDevice);

    // Step 4: Launch Kernel (1 block, N threads)
    sumKernel<<<1, N>>>(d_input, d_output);
    cudaDeviceSynchronize();
    
    // Step 5: Copy results back
    cudaMemcpy(results.data(), d_output, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Step 6: Free GPU memory
    cudaFree(d_input);
    cudaFree(d_output);

    // Step 7: Each thread verifies and prints the result
    #pragma omp parallel num_threads(N)
    {
        int thread_id = omp_get_thread_num();
        int expected = thread_id * L;

        if (results[thread_id] == expected) {
            std::cout << "Thread " << thread_id << " verified sum: " 
                      << results[thread_id] << " ✅" << std::endl;
        } else {
            std::cerr << "Thread " << thread_id << " ERROR! Expected " 
                      << expected << ", but got " << results[thread_id] << std::endl;
        }
    }

    return 0;
}
