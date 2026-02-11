cmake_minimum_required(VERSION 3.10)

project(MATMUL)

find_package(Torch REQUIRED)

set(CMAKE_C_FLAGS "-O3 -march=native -Wall -mavx512f -mavx512bf16 -fopenmp -msse -msse2")
set(CMAKE_CXX_FLAGS "-O3 -march=native -Wall -mavx512f -mavx512bf16 -fopenmp")

set(CMAKE_EXPORT_COMPILE_COMMANDS True)

# include_directories("${PROJECT_SOURCE_DIR}/common")
include_directories("${PROJECT_SOURCE_DIR}/src")

add_executable(benchmark
    avx512_benchmark.cpp
    src/avx512_kernel.cpp
    src/avx512_matmul.cpp
)

target_link_libraries(benchmark "${TORCH_LIBRARIES}")

add_executable(matmul
    avx512_test.cpp
    src/avx512_kernel.cpp
    src/avx512_matmul.cpp
)



