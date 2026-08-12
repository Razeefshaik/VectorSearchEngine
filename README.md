# HNSW index (C++ core + cgo binding)

A from-scratch implementation of Hierarchical Navigable Small World graphs,
faithful to Malkov & Yashunin (2016), built as the single-node core of a
distributed vector search engine.

Read **[DOCUMENTATION.md](DOCUMENTATION.md)** for a full walkthrough of the
algorithm and every design decision.

## Build

```bash
cmake -B build && cmake --build build -j
./build/hnsw_benchmark 20000 128     # recall + latency sweep
./build/hnsw_stress                  # concurrent insert/query
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
