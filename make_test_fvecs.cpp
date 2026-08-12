// make_test_fvecs.cpp -- writes a tiny synthetic dataset in real texmex
// fvecs/ivecs format, with exact brute-force ground truth, purely to verify
// sift_bench.cpp's file reader before running it on the real 1M-vector file.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

static void writeFvecs(const std::string& path, const std::vector<float>& data, size_t n, size_t d) {
    std::ofstream f(path, std::ios::binary);
    for (size_t i = 0; i < n; ++i) {
        int32_t dd = static_cast<int32_t>(d);
        f.write(reinterpret_cast<const char*>(&dd), 4);
        f.write(reinterpret_cast<const char*>(data.data() + i * d), d * sizeof(float));
    }
}
static void writeIvecs(const std::string& path, const std::vector<int32_t>& data, size_t n, size_t d) {
    std::ofstream f(path, std::ios::binary);
    for (size_t i = 0; i < n; ++i) {
        int32_t dd = static_cast<int32_t>(d);
        f.write(reinterpret_cast<const char*>(&dd), 4);
        f.write(reinterpret_cast<const char*>(data.data() + i * d), d * sizeof(int32_t));
    }
}

int main() {
    const size_t N = 5000, DIM = 32, NQ = 200, K = 100;
    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.f, 1.f);

    std::vector<float> base(N * DIM), queries(NQ * DIM);
    for (auto& v : base) v = dist(rng);
    for (auto& v : queries) v = dist(rng);

    // exact brute-force L2 ground truth, top-K per query
    std::vector<int32_t> gt(NQ * K);
    for (size_t q = 0; q < NQ; ++q) {
        std::vector<std::pair<float, int32_t>> scored(N);
        const float* qv = queries.data() + q * DIM;
        for (size_t i = 0; i < N; ++i) {
            const float* bv = base.data() + i * DIM;
            float d = 0;
            for (size_t j = 0; j < DIM; ++j) { float diff = qv[j] - bv[j]; d += diff * diff; }
            scored[i] = {d, static_cast<int32_t>(i)};
        }
        std::partial_sort(scored.begin(), scored.begin() + K, scored.end());
        for (size_t k = 0; k < K; ++k) gt[q * K + k] = scored[k].second;
    }

    writeFvecs("test_base.fvecs", base, N, DIM);
    writeFvecs("test_query.fvecs", queries, NQ, DIM);
    writeIvecs("test_gt.ivecs", gt, NQ, K);
    std::cout << "wrote test_base.fvecs (" << N << "x" << DIM << "), "
              << "test_query.fvecs (" << NQ << "x" << DIM << "), "
              << "test_gt.ivecs (" << NQ << "x" << K << ")\n";
    return 0;
}
