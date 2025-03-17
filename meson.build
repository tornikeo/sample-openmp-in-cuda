# project('tutorial', 'c')

# gtkdep = dependency('gtk+-3.0')
# executable('hello.exe', 'hello.c', dependencies: gtkdep)

project('cuda_project', 'cuda',
  default_options: [
    'cpp_std=c++17',
    'cuda_ccbindir=/usr/bin/g++-11',
  ],
)

cuda_compiler = meson.get_compiler('cuda')
src = ['main.cu','utils.cu']

exe = executable(
  'main.bin', # Do not use exe here 
  sources: src,
  link_args: '-fopenmp',
  cuda_args: '-Xcompiler=-fopenmp',
)

test('simple_run', exe)