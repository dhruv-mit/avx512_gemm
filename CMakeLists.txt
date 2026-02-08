cmake_minimum_required(VERSION 3.10)

project(MATMUL)

set(CMAKE_C_FLAGS "-O3 -march=native -Wall -mavx512f -mavx512bf16 -fopenmp")

set(CMAKE_EXPORT_COMPILE_COMMANDS True)

# include_directories("${PROJECT_SOURCE_DIR}/common")
include_directories("${PROJECT_SOURCE_DIR}/src")

add_executable(MATMUL
    avx512_test.c
    src/avx512_kernel.c
    src/avx512_matmul.c
)





