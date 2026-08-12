cmake_minimum_required(VERSION 3.15)
project(hnsw LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# -march=native enables AVX2/FMA on the build machine. If you ship binaries to
# heterogeneous hardware, replace with -mavx2 -mfma and guard at runtime.
add_compile_options(-O3 -march=native -Wall -Wextra)

# Shared library -- this is what cgo links against.
add_library(hnsw SHARED src/hnsw_c.cpp)
target_include_directories(hnsw PUBLIC include)
find_package(Threads REQUIRED)
target_link_libraries(hnsw PUBLIC Threads::Threads)

# Benchmark / correctness harness.
add_executable(hnsw_benchmark src/benchmark.cpp)
target_include_directories(hnsw_benchmark PRIVATE include)
target_link_libraries(hnsw_benchmark PRIVATE Threads::Threads)

# Concurrency stress test.
add_executable(hnsw_stress src/stress.cpp)
target_include_directories(hnsw_stress PRIVATE include)
target_link_libraries(hnsw_stress PRIVATE Threads::Threads)

# SIFT1M / texmex-format benchmark (real data, precomputed ground truth).
add_executable(hnsw_sift_bench src/sift_bench.cpp)
target_include_directories(hnsw_sift_bench PRIVATE include)
target_link_libraries(hnsw_sift_bench PRIVATE Threads::Threads)
