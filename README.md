# Minimal vscode + meson + cuda sample

Goal: Press `F5` to create builddir, compile, and debug `main.cu`.

![alt text](assets/goal.png)

## Prerequisites

use only official sources for installing:

1. Install [conda/micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)
2. Install [meson](https://mesonbuild.com/Quick-guide.html)
3. Install [cudatoolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/#conda-installation)
4. Install [ninja](https://ninja-build.org/)
5. Install `g++-11`

## Replace contents of .vscode

1. Use `which nvcc`, `which meson`. Note down the locations.
2. Go through .vscode/*.json files and replace paths to whatever `which nvcc` command returns. (i.e., in my case `which nvcc` returns `/home/tornikeo/micromamba/envs/pb/bin/nvcc`)
3. Replace paths to all tools you intend to use, e.g. `cuda-gdb` too.

## Run

Hit F5. It should work.