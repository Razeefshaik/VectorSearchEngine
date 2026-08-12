# HNSW index (C++ core + cgo binding)

A from-scratch implementation of Hierarchical Navigable Small World graphs,
faithful to Malkov & Yashunin (2016), built as the single-node core of a
distributed vector search engine.

Read **[DOCUMENTATION.md](DOCUMENTATION.md)** for a full walkthrough of the
algorithm and every design decision.

## Build & run — full flow

Everything below runs from the repo root, in order. Steps 1–2 are required;
3–5 are the three test/benchmark executables (run any or all); 6 is optional
and needs a real SIFT1M download.

### 1. Put the MinGW toolchain on PATH

```powershell
$env:PATH = "S:\CLion 2024.3.1.1\bin\mingw\bin;" + $env:PATH   # runtime DLLs (libstdc++-6.dll etc.) must be on PATH
```

### 2. Configure + build with CMake/Ninja

```powershell
& "S:\CLion 2024.3.1.1\bin\cmake\win\x64\bin\cmake.exe" -B build -G Ninja `
  -DCMAKE_MAKE_PROGRAM="S:/CLion 2024.3.1.1/bin/ninja/win/x64/ninja.exe" `
  -DCMAKE_CXX_COMPILER="S:/CLion 2024.3.1.1/bin/mingw/bin/g++.exe"

& "S:\CLion 2024.3.1.1\bin\cmake\win\x64\bin\cmake.exe" --build build -j
```

This produces `libhnsw.dll` (the cgo-facing shared lib) plus three
executables: `hnsw_benchmark.exe`, `hnsw_stress.exe`, `hnsw_sift_bench.exe`.

### 3. General recall/latency benchmark + correctness checks

```powershell
.\build\hnsw_benchmark.exe 20000 128   # args: N vectors, dim (defaults 20000 128)
```

Sweeps `efSearch`, prints a recall@10 vs. latency/QPS table, then runs a
soft-delete check and a save/load roundtrip check.

### 4. Concurrency stress test

```powershell
.\build\hnsw_stress.exe                # concurrent insert/query, no args
```

Runs multiple writer/reader threads against the same index and checks for
corruption.

### 5. SIFT1M-format benchmark, using synthetic data (no download needed)

`hnsw_sift_bench.exe` reads the real texmex `.fvecs`/`.ivecs` format. To
exercise it without the 168 MB SIFT1M download, first generate a small
synthetic dataset in that format:

```powershell
g++ -std=c++17 -O2 make_test_fvecs.cpp -o make_test_fvecs.exe
.\make_test_fvecs.exe                  # writes test_base.fvecs, test_query.fvecs, test_gt.ivecs
.\build\hnsw_sift_bench.exe test_base.fvecs test_query.fvecs test_gt.ivecs
```

`hnsw_sift_bench` args: `<base.fvecs> <query.fvecs> <groundtruth.ivecs> [N] [M] [efConstruction]`
(`N`=0 means "use all vectors in the file").

### 6. (Optional) Real SIFT1M benchmark

Needs `wget`/`curl` and outbound FTP access:

```bash
./download_sift1m.sh                   # fetches into data/sift/
```

```powershell
.\build\hnsw_sift_bench.exe data\sift\sift_base.fvecs data\sift\sift_query.fvecs data\sift\sift_groundtruth.ivecs
```

Without cmake:

```bash
g++ -std=c++17 -O3 -march=native -Iinclude src/benchmark.cpp -o bench -pthread
```

## Features

- Algorithms 1–5 from the paper, including the neighbour-selection heuristic
- Flat contiguous storage, float32, AVX2 distance kernels
- Cosine (via pre-normalized inner product) and squared L2
- Thread-safe concurrent insert and query
- Versioned binary snapshots (`save` / `load`)
- Soft deletes with tombstones
- `extern "C"` ABI + Go binding for the cgo boundary

## Layout

```
include/hnsw.hpp     the algorithm (header-only)
include/hnsw_c.h     flat C ABI for cgo
src/hnsw_c.cpp       C ABI implementation
src/benchmark.cpp    recall@k vs brute force, latency percentiles
src/stress.cpp       concurrency test (run under -fsanitize=thread)
go/hnsw.go           cgo binding
```

## Verified

| test | result |
|---|---|
| recall@10, 20k × 16d, ef=100 | 1.000 |
| recall@10, 20k × 32d, ef=200 | 0.996 |
| save/load roundtrip | identical results across 200 queries |
| soft delete | filtered from results, graph connectivity preserved |
| 4 writers + 4 readers concurrent | 20k inserts, 72k queries, no corruption |

Random Gaussian data is near-worst-case for ANN. Benchmark on SIFT1M or GloVe
for numbers you can quote — see DOCUMENTATION.md §13.
