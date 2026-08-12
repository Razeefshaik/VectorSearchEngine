// sift_bench.cpp -- benchmark against the standard SIFT1M / GIST1M / etc.
// texmex-corpus format (.fvecs base + query, .ivecs ground truth).
//
// Usage:
//   sift_bench <base.fvecs> <query.fvecs> <groundtruth.ivecs> [N] [M] [efConstruction]
//
//   N              -- number of base vectors to index (default: all in file)
//   M              -- graph out-degree (default 16)
//   efConstruction -- build beam width (default 200)
//
// Unlike benchmark.cpp, ground truth here is NOT computed by brute force --
// it ships with the dataset, precomputed once by the dataset authors. This is
// both faster and the standard way these datasets are used, so your numbers
// are directly comparable to published results.

#include "hnsw.hpp"

#include <chrono>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace hnsw;
using Clock = std::chrono::steady_clock;

// ---------------------------------------------------------------------------
// .fvecs / .ivecs reader
//
// Format (little-endian, used unchanged since the original texmex corpus):
//   repeated records, each:
//     int32   d          -- dimension of this vector
//     d * T   values      -- T = float32 for .fvecs, int32 for .ivecs
//
// Every record in a given file has the same d in practice, so we read it once
// from the first record and use it to compute how many records the file holds
// from its size -- no separate index or header.
// ---------------------------------------------------------------------------

template <typename T>
struct VecFile {
    std::vector<T> data;   // n * d, contiguous, row-major
    size_t n = 0, d = 0;
};

template <typename T>
VecFile<T> readVecs(const std::string& path, size_t limit = 0) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);

    int32_t d0;
    f.read(reinterpret_cast<char*>(&d0), sizeof(int32_t));
    if (!f || d0 <= 0) throw std::runtime_error(path + ": failed to read dimension header");

    f.seekg(0, std::ios::end);
    std::streamoff fileSize = f.tellg();
    f.seekg(0, std::ios::beg);

    size_t recordSize = sizeof(int32_t) + static_cast<size_t>(d0) * sizeof(T);
    if (fileSize % static_cast<std::streamoff>(recordSize) != 0)
        throw std::runtime_error(path + ": file size not a multiple of record size -- "
                                        "wrong dtype (fvecs vs ivecs) or corrupt file?");

    size_t total = static_cast<size_t>(fileSize) / recordSize;
    size_t n = (limit == 0) ? total : std::min(limit, total);

    VecFile<T> out;
    out.d = static_cast<size_t>(d0);
    out.n = n;
    out.data.resize(n * out.d);

    for (size_t i = 0; i < n; ++i) {
        int32_t d;
        f.read(reinterpret_cast<char*>(&d), sizeof(int32_t));
        if (static_cast<size_t>(d) != out.d)
            throw std::runtime_error(path + ": inconsistent dimension at record " + std::to_string(i));
        f.read(reinterpret_cast<char*>(out.data.data() + i * out.d), out.d * sizeof(T));
    }
    if (!f && !f.eof()) throw std::runtime_error(path + ": read error");
    return out;
}

static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t i = static_cast<size_t>(p / 100.0 * (v.size() - 1));
    return v[i];
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr,
            "usage: %s <base.fvecs> <query.fvecs> <groundtruth.ivecs> "
            "[N] [M] [efConstruction]\n", argv[0]);
        return 1;
    }
    const std::string basePath = argv[1], queryPath = argv[2], gtPath = argv[3];
    const size_t N       = (argc > 4) ? std::stoul(argv[4]) : 0;   // 0 = all
    const size_t M       = (argc > 5) ? std::stoul(argv[5]) : 16;
    const size_t EF_CONS = (argc > 6) ? std::stoul(argv[6]) : 200;
    const size_t K       = 10;   // standard recall@10

    std::cout << "loading base vectors from " << basePath << " ...\n";
    auto base = readVecs<float>(basePath, N);
    std::cout << "  " << base.n << " vectors, dim=" << base.d << "\n";

    std::cout << "loading queries from " << queryPath << " ...\n";
    auto queries = readVecs<float>(queryPath);
    std::cout << "  " << queries.n << " queries, dim=" << queries.d << "\n";

    std::cout << "loading ground truth from " << gtPath << " ...\n";
    auto gt = readVecs<int32_t>(gtPath);
    std::cout << "  " << gt.n << " rows, " << gt.d << " neighbours each\n\n";

    if (base.d != queries.d)
        throw std::runtime_error("base and query dimension mismatch");
    if (gt.n < queries.n)
        throw std::runtime_error("ground truth has fewer rows than queries");
    if (gt.d < K)
        throw std::runtime_error("ground truth has fewer than K neighbours per row");

    // If we indexed only a subset (N < full base), ground truth computed
    // against the FULL base is no longer valid -- neighbours may have been
    // excluded. Only meaningful to compare recall when N == 0 (full base).
    bool subsetWarning = (N != 0 && N < 1000000);   // heuristic; texmex bases are ~1M

    std::cout << "HNSW build: dim=" << base.d << " M=" << M
              << " efConstruction=" << EF_CONS << " space=L2\n";

    Index index(Space::L2, base.d, base.n, M, EF_CONS);

    auto t0 = Clock::now();
    for (size_t i = 0; i < base.n; ++i) {
        index.addPoint(base.data.data() + i * base.d, static_cast<label_t>(i));
        if ((i + 1) % 100000 == 0)
            std::cout << "  inserted " << (i + 1) << " / " << base.n << "\n";
    }
    auto t1 = Clock::now();
    double buildSec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\nbuild: " << std::fixed << std::setprecision(1) << buildSec << " s  ("
              << static_cast<size_t>(base.n / buildSec) << " inserts/s)\n"
              << "graph height: " << index.maxLevel() + 1 << " layers\n"
              << "memory: " << index.memoryBytes() / (1024.0 * 1024.0) << " MB  ("
              << std::setprecision(1) << index.memoryBytes() / static_cast<double>(base.n)
              << " bytes/vector)\n";
    if (subsetWarning)
        std::cout << "NOTE: indexed a subset (N=" << base.n << ") but ground truth was computed "
                     "against the full base -- recall numbers below are not meaningful. "
                     "Run with N=0 (all vectors) for a real comparison.\n";
    std::cout << "\n";

    const size_t NQUERY = queries.n;   // typically 10,000 for SIFT1M

    std::cout << "efSearch |  recall@" << K << "  |  p50 (ms)  p95 (ms)  p99 (ms)  |   QPS\n"
              << "---------+------------+---------------------------------+---------\n";

    bool hitTarget = false;
    for (size_t ef : {10, 20, 40, 60, 100, 150, 200, 400}) {
        if (ef < K) continue;
        size_t hits = 0;
        std::vector<double> latencies;
        latencies.reserve(NQUERY);

        for (size_t q = 0; q < NQUERY; ++q) {
            auto s0 = Clock::now();
            auto res = index.search(queries.data.data() + q * queries.d, K, ef);
            auto s1 = Clock::now();
            latencies.push_back(std::chrono::duration<double, std::milli>(s1 - s0).count());

            // ground-truth row q, first K entries, sorted nearest-first
            const int32_t* truth = gt.data.data() + q * gt.d;
            for (const auto& r : res)
                for (size_t t = 0; t < K; ++t)
                    if (static_cast<int32_t>(r.label) == truth[t]) { ++hits; break; }
        }

        double recall = static_cast<double>(hits) / (NQUERY * K);
        double p50 = percentile(latencies, 50), p95 = percentile(latencies, 95),
               p99 = percentile(latencies, 99);
        double mean = 0; for (double l : latencies) mean += l; mean /= latencies.size();

        std::printf("%8zu |   %6.3f   |  %7.3f   %7.3f   %7.3f  | %7.0f\n",
                    ef, recall, p50, p95, p99, 1000.0 / mean);

        if (recall >= 0.95 && p99 < 50.0 && !hitTarget) hitTarget = true;
    }

    std::cout << "\n" << (hitTarget ? "PASS" : "no efSearch value hit both targets")
              << ": 95% recall@10 AND p99 < 50ms\n";
    return 0;
}
