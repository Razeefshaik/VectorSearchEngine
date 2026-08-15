// snapshot_race.cpp -- does Index::save() need writes to be quiesced?
//
// The shard server wants a background snapshot ticker running while gRPC
// writes continue. That is only safe if save() is synchronized against
// addPoint(). This checks, under ThreadSanitizer.
//
//   g++ -std=c++17 -O1 -g -fsanitize=thread -Iinclude src/snapshot_race.cpp \
//       -o snapshot_race -pthread

#include "hnsw.hpp"
#include <atomic>
#include <iostream>
#include <random>
#include <thread>
#include <vector>

using namespace hnsw;

int main() {
    const size_t N = 4000, DIM = 32;

    std::mt19937 rng(7);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> base(N * DIM);
    for (auto& v : base) v = dist(rng);

    Index index(Space::Cosine, DIM, N, 16, 100);
    std::atomic<size_t> next{0};
    std::atomic<bool> done{false};
    std::atomic<size_t> snapshots{0};

    // Writer: inserts continuously.
    std::thread writer([&] {
        size_t i;
        while ((i = next.fetch_add(1)) < N)
            index.addPoint(base.data() + i * DIM, Label{1, static_cast<uint64_t>(i)});
        done.store(true);
    });

    // Snapshotter: saves repeatedly WHILE the writer is inserting.
    std::thread snapshotter([&] {
        while (!done.load()) {
            index.save("/tmp/race_snapshot.bin");
            snapshots.fetch_add(1);
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    });

    writer.join();
    snapshotter.join();

    std::cout << "inserted " << index.size() << " vectors while taking "
              << snapshots.load() << " concurrent snapshots\n"
              << "(if ThreadSanitizer printed WARNINGs above, save() is NOT\n"
              << " safe against concurrent writes and must be serialized)\n";
    return 0;
}
