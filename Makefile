mpi:
	nvcc -c -o cuda_kernel.o main.cu -Xcompiler -fopenmp
	mpicxx -o mpi_cuda_program cuda_kernel.o -L/usr/local/cuda/lib64 -lcudart
	mpirun -np 4 ./mpi_cuda_program
omp:
	nvcc -ccbin g++-11 -Xcompiler -fopenmp -o main.exe main.cu
	OMP_NUM_THREADS=1 ./main.exe