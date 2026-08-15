// shard_sim.cpp -- validates the cluster design BEFORE we build it.
//
// Three questions this answers empirically, using the real HNSW core:
//
//   1. Does fnv1a(clientId, label) mod numShards distribute vectors evenly?
//      (If not, one shard runs out of capacity while others sit idle.)
//
//   2. Does scatter-gather (ask every shard for k, merge by distance) return
//      the SAME results as one monolithic index holding all the vectors?
//      This is the correctness claim the whole coordinator design rests on.
//
//   3. What happens if you use the "obvious" optimization of asking each shard
//      for k/numShards instead of k? (Spoiler: it's wrong. This quantifies
//      how wrong.)
//
// Build:
//   g++ -std=c++17 -O2 -march=native -Iinclude src/shard_sim.cpp -o shard_sim -pthread

#include "hnsw.hpp"

#include <algorithm>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <vector>

using namespace hnsw;

// ---------------------------------------------------------------------------
// Routing hash. This MUST match the Go implementation byte for byte, or a
// vector written via Go lands on a different shard than a C++ reader expects.
//
// Mirrors: h := fnv.New64a(); binary.Write(h, binary.LittleEndian, clientID);
//          binary.Write(h, binary.LittleEndian, label); h.Sum64()
// ---------------------------------------------------------------------------
static uint64_t fnv1a64(const uint8_t* data, size_t n) {
    uint64_t h = 14695981039346656037ULL;      // FNV offset basis
    for (size_t i = 0; i < n; ++i) {
        h ^= static_cast<uint64_t>(data[i]);
        h *= 1099511628211ULL;                 // FNV prime
    }
    return h;
}

static uint64_t hashKey(uint64_t clientId, uint64_t label) {
    uint8_t buf[16];
    for (int i = 0; i < 8; ++i) buf[i]     = static_cast<uint8_t>(clientId >> (8 * i)); // LE
    for (int i = 0; i < 8; ++i) buf[8 + i] = static_cast<uint8_t>(label    >> (8 * i)); // LE
    return fnv1a64(buf, 16);
}

static size_t shardFor(uint64_t clientId, uint64_t label, size_t numShards) {
    return static_cast<size_t>(hashKey(clientId, label) % numShards);
}

// ---------------------------------------------------------------------------

struct Scored {
    float    distance;
    uint64_t clientId;
    uint64_t label;
    bool operator<(const Scored& o) const { return distance < o.distance; }
};

static std::vector<Scored> bruteForce(const std::vector<float>& base,
                                      const std::vector<uint64_t>& clientIds,
                                      size_t n, size_t dim,
                                      const float* query, size_t k) {
    // Normalize query and base rows for cosine, same as the index does.
    std::vector<float> qn(query, query + dim);
    float qnorm = std::sqrt(dotProduct(qn.data(), qn.data(), dim));
    if (qnorm > 0) for (auto& v : qn) v /= qnorm;

    std::vector<Scored> all(n);
    for (size_t i = 0; i < n; ++i) {
        std::vector<float> bn(base.data() + i * dim, base.data() + (i + 1) * dim);
        float bnorm = std::sqrt(dotProduct(bn.data(), bn.data(), dim));
        if (bnorm > 0) for (auto& v : bn) v /= bnorm;
        all[i] = Scored{cosineDistance(qn.data(), bn.data(), dim),
                        clientIds[i], static_cast<uint64_t>(i)};
    }
    std::partial_sort(all.begin(), all.begin() + k, all.end());
    all.resize(k);
    return all;
}

static double overlapFraction(const std::vector<Scored>& a, const std::vector<Scored>& b) {
    std::set<std::pair<uint64_t, uint64_t>> sb;
    for (const auto& x : b) sb.insert({x.clientId, x.label});
    size_t hits = 0;
    for (const auto& x : a) if (sb.count({x.clientId, x.label})) ++hits;
    return a.empty() ? 0.0 : static_cast<double>(hits) / a.size();
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    const size_t N         = (argc > 1) ? std::stoul(argv[1]) : 20000;
    const size_t DIM       = (argc > 2) ? std::stoul(argv[2]) : 64;
    const size_t NUMSHARDS = (argc > 3) ? std::stoul(argv[3]) : 4;
    const size_t NQUERY    = 200;
    const size_t K         = 10;
    const size_t EF        = 100;
    const size_t M         = 16;
    const size_t EFC       = 200;
    const size_t NCLIENTS  = 5;   // several clients, to exercise the composite key

    std::cout << "cluster design simulation\n"
              << "  vectors=" << N << " dim=" << DIM << " shards=" << NUMSHARDS
              << " clients=" << NCLIENTS << " k=" << K << " ef=" << EF << "\n\n";

    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> base(N * DIM);
    for (auto& v : base) v = nd(rng);

    std::vector<float> queries(NQUERY * DIM);
    for (auto& v : queries) v = nd(rng);

    // Assign each vector to a client. label == index, so (clientId, label)
    // pairs are unique even though labels repeat across clients.
    std::vector<uint64_t> clientIds(N);
    for (size_t i = 0; i < N; ++i) clientIds[i] = (i % NCLIENTS) + 1;

    // ---- Q1: hash distribution ------------------------------------------
    std::vector<size_t> counts(NUMSHARDS, 0);
    std::vector<std::vector<size_t>> shardMembers(NUMSHARDS);
    for (size_t i = 0; i < N; ++i) {
        size_t s = shardFor(clientIds[i], i, NUMSHARDS);
        counts[s]++;
        shardMembers[s].push_back(i);
    }

    std::cout << "--- Q1: shard balance ---\n";
    size_t mn = N, mx = 0;
    for (size_t s = 0; s < NUMSHARDS; ++s) {
        double pct = 100.0 * counts[s] / N;
        std::printf("  shard %zu: %6zu vectors (%5.2f%%)\n", s, counts[s], pct);
        mn = std::min(mn, counts[s]);
        mx = std::max(mx, counts[s]);
    }
    double skew = 100.0 * (static_cast<double>(mx) - mn) / (static_cast<double>(N) / NUMSHARDS);
    std::printf("  spread: min=%zu max=%zu, skew=%.2f%% of even split\n\n", mn, mx, skew);

    // ---- build the monolithic index --------------------------------------
    std::cout << "building monolithic index (" << N << " vectors)...\n";
    Index mono(Space::Cosine, DIM, N, M, EFC);
    for (size_t i = 0; i < N; ++i)
        mono.addPoint(base.data() + i * DIM, Label{clientIds[i], static_cast<uint64_t>(i)});

    // ---- build the sharded indexes ---------------------------------------
    std::cout << "building " << NUMSHARDS << " shard indexes...\n\n";
    std::vector<std::unique_ptr<Index>> shards;
    for (size_t s = 0; s < NUMSHARDS; ++s) {
        // Real deployment would size this with headroom; exact fit here.
        shards.push_back(std::make_unique<Index>(
            Space::Cosine, DIM, std::max<size_t>(counts[s], 1), M, EFC));
        for (size_t i : shardMembers[s])
            shards[s]->addPoint(base.data() + i * DIM,
                                Label{clientIds[i], static_cast<uint64_t>(i)});
    }

    // ---- Q2 + Q3: query all three ways -----------------------------------
    double sumRecallMono = 0, sumRecallScatter = 0, sumRecallNaive = 0;
    double sumRecallMonoFair = 0;
    double sumOverlapMonoScatter = 0;
    size_t exactMatches = 0;

    for (size_t q = 0; q < NQUERY; ++q) {
        const float* qv = queries.data() + q * DIM;
        auto truth = bruteForce(base, clientIds, N, DIM, qv, K);

        // (a) monolithic at the same ef as each shard
        auto monoRes = mono.search(qv, K, EF);
        std::vector<Scored> monoScored;
        for (const auto& r : monoRes)
            monoScored.push_back(Scored{r.distance, r.label.clientId, r.label.label});

        // (b) scatter-gather, CORRECT: ask every shard for the full k
        std::vector<Scored> pool;
        for (size_t s = 0; s < NUMSHARDS; ++s) {
            auto res = shards[s]->search(qv, K, EF);
            for (const auto& r : res)
                pool.push_back(Scored{r.distance, r.label.clientId, r.label.label});
        }
        std::sort(pool.begin(), pool.end());
        if (pool.size() > K) pool.resize(K);

        // (c) scatter-gather, NAIVE: ask each shard for only k/numShards
        size_t perShard = std::max<size_t>(1, K / NUMSHARDS);
        std::vector<Scored> naivePool;
        for (size_t s = 0; s < NUMSHARDS; ++s) {
            auto res = shards[s]->search(qv, perShard, EF);
            for (const auto& r : res)
                naivePool.push_back(Scored{r.distance, r.label.clientId, r.label.label});
        }
        std::sort(naivePool.begin(), naivePool.end());
        if (naivePool.size() > K) naivePool.resize(K);

        // (a2) monolithic at ef * numShards -- same total candidate budget
        //      as the scatter-gather does across all shards. This is the
        //      apples-to-apples comparison.
        auto monoFairRes = mono.search(qv, K, EF * NUMSHARDS);
        std::vector<Scored> monoFairScored;
        for (const auto& r : monoFairRes)
            monoFairScored.push_back(Scored{r.distance, r.label.clientId, r.label.label});
        sumRecallMonoFair += overlapFraction(monoFairScored, truth);

        sumRecallMono    += overlapFraction(monoScored, truth);
        sumRecallScatter += overlapFraction(pool, truth);
        sumRecallNaive   += overlapFraction(naivePool, truth);
        sumOverlapMonoScatter += overlapFraction(pool, monoScored);

        bool identical = pool.size() == monoScored.size();
        for (size_t i = 0; identical && i < pool.size(); ++i)
            if (pool[i].clientId != monoScored[i].clientId ||
                pool[i].label    != monoScored[i].label) identical = false;
        if (identical) ++exactMatches;
    }

    std::cout << "--- Q2: does scatter-gather match a single index? ---\n";
    std::printf("  recall@%zu vs exact ground truth:\n", K);
    std::printf("    monolithic, ef=%zu          : %.4f\n", EF, sumRecallMono / NQUERY);
    std::printf("    monolithic, ef=%zu (equal work): %.4f\n", EF * NUMSHARDS,
                sumRecallMonoFair / NQUERY);
    std::printf("    scatter-gather, ef=%zu x %zu shards: %.4f\n", EF, NUMSHARDS,
                sumRecallScatter / NQUERY);
    std::printf("  result overlap, scatter vs monolithic: %.4f\n",
                sumOverlapMonoScatter / NQUERY);
    std::printf("  queries where the two agreed exactly (order included): %zu / %zu\n\n",
                exactMatches, NQUERY);

    std::cout << "--- Q3: the naive k/numShards optimization ---\n";
    std::printf("  asked each shard for %zu instead of %zu\n", std::max<size_t>(1, K / NUMSHARDS), K);
    std::printf("    recall@%zu: %.4f  (vs %.4f correct)\n",
                K, sumRecallNaive / NQUERY, sumRecallScatter / NQUERY);
    double lost = (sumRecallScatter - sumRecallNaive) / NQUERY;
    std::printf("    recall lost: %.4f (%.1f%% of the correct result set)\n\n",
                lost, 100.0 * lost / (sumRecallScatter / NQUERY));

    return 0;
}
