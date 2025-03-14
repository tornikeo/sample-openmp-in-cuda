# project('tutorial', 'c')

# gtkdep = dependency('gtk+-3.0')
# executable('hello.exe', 'hello.c', dependencies: gtkdep)

project('cuda_project', 'cuda',
  default_options: [
    'cpp_std=c++17',
    'cuda_ccbindir=/usr/bin/g++-11',
  ],
)

add_global_arguments('-Xcompiler=-fopenmp', language: 'cuda')

cuda_compiler = meson.get_compiler('cuda')

executable(
  'main.exe', 
  'main.cu',
  link_args: '-fopenmp'
)