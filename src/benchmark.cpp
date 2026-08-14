





#include "hnsw.hpp"

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

using namespace hnsw;
using Clock = std::chrono::steady_clock;





static std::vector<float> makeRandomVectors(size_t n, size_t dim, uint64_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> out(n * dim);
    for (auto& v : out) v = dist(rng);
    return out;
}


static std::vector<label_t> bruteForce(const std::vector<float>& base, size_t n, size_t dim,
                                       const float* query, size_t k, Space space) {
    std::vector<std::pair<float, label_t>> scored(n);

    std::vector<float> qn(query, query + dim);
    if (space == Space::Cosine) {
        float norm = std::sqrt(dotProduct(qn.data(), qn.data(), dim));
        if (norm > 0) for (auto& v : qn) v /= norm;
    }

    for (size_t i = 0; i < n; ++i) {
        const float* p = base.data() + i * dim;
        float d;
        if (space == Space::Cosine) {
            std::vector<float> pn(p, p + dim);
            float norm = std::sqrt(dotProduct(pn.data(), pn.data(), dim));
            if (norm > 0) for (auto& v : pn) v /= norm;
            d = cosineDistance(qn.data(), pn.data(), dim);
        } else {
            d = l2SquaredDistance(qn.data(), p, dim);
        }
        scored[i] = {d, static_cast<label_t>(i)};
    }
    std::partial_sort(scored.begin(), scored.begin() + k, scored.end());
    std::vector<label_t> out(k);
    for (size_t i = 0; i < k; ++i) out[i] = scored[i].second;
    return out;
}

static double percentile(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t i = static_cast<size_t>(p / 100.0 * (v.size() - 1));
    return v[i];
}



int main(int argc, char** argv) {
    const size_t N        = (argc > 1) ? std::stoul(argv[1]) : 20000;
    const size_t DIM      = (argc > 2) ? std::stoul(argv[2]) : 128;
    const size_t NQUERY   = 200;
    const size_t K        = 10;
    const size_t M        = 16;
    const size_t EF_CONS  = 200;
    const Space  SPACE    = Space::Cosine;

    std::cout << "HNSW benchmark\n"
              << "  vectors=" << N << "  dim=" << DIM
              << "  M=" << M << "  efConstruction=" << EF_CONS
              << "  space=" << (SPACE == Space::Cosine ? "cosine" : "l2") << "\n\n";

    auto base    = makeRandomVectors(N, DIM, 42);
    auto queries = makeRandomVectors(NQUERY, DIM, 1337);

    
    Index index(SPACE, DIM, N, M, EF_CONS);

    auto t0 = Clock::now();
    for (size_t i = 0; i < N; ++i)
        index.addPoint(base.data() + i * DIM, static_cast<label_t>(i));
    auto t1 = Clock::now();

    double buildSec = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "build: " << std::fixed << std::setprecision(2) << buildSec << " s  ("
              << static_cast<size_t>(N / buildSec) << " inserts/s)\n"
              << "graph height: " << index.maxLevel() + 1 << " layers\n"
              << "memory: " << index.memoryBytes() / (1024.0 * 1024.0) << " MB  ("
              << index.memoryBytes() / static_cast<double>(N) << " bytes/vector)\n\n";

    
    std::cout << "computing exact ground truth for " << NQUERY << " queries...\n";
    std::vector<std::vector<label_t>> truth(NQUERY);
    for (size_t q = 0; q < NQUERY; ++q)
        truth[q] = bruteForce(base, N, DIM, queries.data() + q * DIM, K, SPACE);
    std::cout << "\n";

    
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
            auto res = index.search(queries.data() + q * DIM, K, ef);
            auto s1 = Clock::now();
            latencies.push_back(std::chrono::duration<double, std::milli>(s1 - s0).count());

            for (const auto& r : res)
                for (label_t t : truth[q])
                    if (r.label == t) { ++hits; break; }
        }

        double recall = static_cast<double>(hits) / (NQUERY * K);
        double p50 = percentile(latencies, 50), p95 = percentile(latencies, 95),
               p99 = percentile(latencies, 99);
        double mean = 0; for (double l : latencies) mean += l; mean /= latencies.size();

        std::printf("%8zu |   %6.3f   |  %7.3f   %7.3f   %7.3f  | %7.0f\n",
                    ef, recall, p50, p95, p99, 1000.0 / mean);

        if (recall >= 0.95 && !hitTarget) hitTarget = true;
    }
    std::cout << "\n" << (hitTarget ? "PASS" : "FAIL")
              << ": 95% recall@10 target " << (hitTarget ? "reached" : "NOT reached")
              << "\n\n";

    
    auto before = index.search(queries.data(), K, 100);
    label_t victim = before.front().label;
    index.markDeleted(victim);
    auto after = index.search(queries.data(), K, 100);
    bool victimGone = true;
    for (const auto& r : after) if (r.label == victim) victimGone = false;
    std::cout << "soft delete: label " << victim << " -> "
              << (victimGone ? "PASS (filtered from results)" : "FAIL (still returned)")
              << "  active=" << index.activeSize() << "/" << index.size() << "\n";
    index.unmarkDeleted(victim);

    
    const std::string path = (std::filesystem::temp_directory_path() / "hnsw_index.bin").string();
    index.save(path);
    auto reloaded = Index::load(path);

    size_t mismatches = 0;
    for (size_t q = 0; q < NQUERY; ++q) {
        auto a = index.search(queries.data() + q * DIM, K, 100);
        auto b = reloaded->search(queries.data() + q * DIM, K, 100);
        if (a.size() != b.size()) { ++mismatches; continue; }
        for (size_t i = 0; i < a.size(); ++i)
            if (a[i].label != b[i].label) { ++mismatches; break; }
    }
    std::cout << "save/load: " << (mismatches == 0 ? "PASS" : "FAIL")
              << " (" << mismatches << " query mismatches, "
              << reloaded->size() << " vectors restored)\n";

    return 0;
}
