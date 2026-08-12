// stress.cpp -- concurrent insert + query, to be run under -fsanitize=thread.
//
//   g++ -std=c++17 -O1 -g -fsanitize=thread -Iinclude src/stress.cpp -o stress -pthread
//
// This is the test that catches the data races cgo will otherwise expose in
// production, where Go serves many requests per second against one index.

#include "hnsw.hpp"
#include <atomic>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace hnsw;

int main() {
    const size_t N = 20000, DIM = 64, WRITERS = 4, READERS = 4;

    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> base(N * DIM);
    for (auto& v : base) v = dist(rng);

    Index index(Space::Cosine, DIM, N, 16, 100);
    std::atomic<size_t> next{0};
    std::atomic<bool> done{false};
    std::atomic<size_t> queriesRun{0};

    std::vector<std::thread> threads;
    for (size_t w = 0; w < WRITERS; ++w) {
        threads.emplace_back([&] {
            size_t i;
            while ((i = next.fetch_add(1)) < N)
                index.addPoint(base.data() + i * DIM, static_cast<label_t>(i));
        });
    }
    for (size_t r = 0; r < READERS; ++r) {
        threads.emplace_back([&] {
            std::mt19937 local(r + 1);
            std::normal_distribution<float> d(0.f, 1.f);
            std::vector<float> q(DIM);
            while (!done.load()) {
                for (auto& v : q) v = d(local);
                auto res = index.search(q.data(), 10, 50);
                (void)res;
                queriesRun.fetch_add(1);
            }
        });
    }

    for (size_t w = 0; w < WRITERS; ++w) threads[w].join();
    done.store(true);
    for (size_t r = WRITERS; r < threads.size(); ++r) threads[r].join();

    std::cout << "inserted " << index.size() << " vectors across " << WRITERS
              << " writers\nserved " << queriesRun.load()
              << " concurrent queries across " << READERS << " readers\n";

    auto res = index.search(base.data(), 10, 100);
    std::cout << "sanity query returned " << res.size()
              << " results, nearest label=" << (res.empty() ? 0 : res[0].label)
              << " (expected 0)\n";
    return 0;
}
