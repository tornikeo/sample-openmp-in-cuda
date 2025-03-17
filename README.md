# Minimal OpenMP + CUDA sample in C++

Shows a sample of using OpenMP with CUDA, with multiple CPUs batching their requests to query the GPU at the same time. In short:

1. Parallel CPU threads create jobs
2. Main thread concatenates jobs and sends it to GPU
3. GPU exection finishes
4. Main thread unconcats job results back to threads 
4. Parallel CPU threads finalize job

Refer to [starter repo](https://github.com/tornikeo/minimal-vscode-cuda-meson) on setting this up with vscode + meson.

# Compile and run this

```sh
meson setup builddir
meson compile -C builddir
meson test -C builddir --verbose
```
Should output:

```sh
# <build steps> ...

Start
Device Number: 0
  Device name: NVIDIA GeForce GTX 1050 Ti with Max-Q Design
  Memory Clock Rate (KHz): 3504000
  Memory Bus Width (bits): 128
  Peak Memory Bandwidth (GB/s): 112.128000

Thread 0 is working...
Thread 7 is working...
Thread 2 is working...
Thread 5 is working...
Thread 6 is working...
Thread 3 is working...
Thread 1 is working...
Thread 4 is working...
Numbers going into GPU:
0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 
1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 
2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 
3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 
4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 4 
5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 5 
6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 6 
7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 7 
Launching GPU...
We expect the kernel to sum the values in each row...
Thread 0 verified sum: 0 ✅
Thread 1 verified sum: 32 ✅
Thread 2 verified sum: 64 ✅
Thread 7 verified sum: 224 ✅
Thread 4 verified sum: 128 ✅
Thread 6 verified sum: 192 ✅
Thread 5 verified sum: 160 ✅
Thread 3 verified sum: 96 ✅
# ...
```

# Prerequisites

- cudatoolkit, cudatoolkit-dev (e.g from micromamba or conda)
- g++-11 (build-essential)
