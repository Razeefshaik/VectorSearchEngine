# HNSW, explained through the code

This document walks through every part of the implementation in `include/hnsw.hpp`.
Read it top to bottom the first time; after that use it as a reference.

The goal is that by the end you could re-derive the algorithm from scratch and
defend every design choice in an interview.

---

## 0. The intuition first

Forget graphs for a second. Think about a **skip list**.

A skip list is a sorted linked list with express lanes. The bottom lane has every
element. The next lane up has half of them. The next has a quarter. To find
something, you start in the top lane, sprint right until you'd overshoot, drop
down a lane, sprint again, drop down, and so on. You cover huge distance cheaply
at the top and refine at the bottom. That's `O(log n)` instead of `O(n)`.

**HNSW is a skip list for vectors.** The problem is that vectors have no natural
sort order — "next" is meaningless in 768 dimensions. So instead of a sorted
list at each layer, HNSW builds a *graph* at each layer, where each node is
connected to some of its nearest neighbours. "Sprint right until you'd
overshoot" becomes "hop to whichever neighbour is closer to my query, repeat
until no neighbour is closer."

Two ideas, layered on each other:

1. **The navigable small world graph** (the "NSW" part): a graph where greedy
   hopping actually reaches the true nearest neighbour, because the graph
   contains both short local links *and* a few long-range shortcuts.
2. **The hierarchy** (the "H" part): stack several such graphs. The top ones are
   sparse and let you cross the whole space in a few hops. The bottom one has
   everything and gives you precision.

Everything else in this document is detail on how to build those two things
correctly.

```
     layer 2      (A)- - - - - - - - - - - - -(F)          sparse, long hops
                   |                            |
     layer 1      (A)- - - -(C)- - - - - -(E)-(F)          medium
                   |         |             |    |
     layer 0      (A)-(B)-(C)-(D)-(E)-(F)-(G)-(H)          everything, short hops
                                  ^
                              query lands here
```

A search enters at `A` on layer 2, hops greedily, drops to layer 1, hops again,
drops to layer 0, and does a careful beam search there.

---

## 1. Quick start

```bash
cmake -B build && cmake --build build -j
./build/hnsw_benchmark 20000 128      # N vectors, D dimensions
./build/hnsw_stress                   # concurrent insert + query
```

Or without cmake:

```bash
g++ -std=c++17 -O3 -march=native -Iinclude src/benchmark.cpp -o bench -pthread
./bench 20000 128
```

Minimal usage:

```cpp
#include "hnsw.hpp"
using namespace hnsw;

Index index(Space::Cosine, /*dim=*/128, /*maxElements=*/1000000,
            /*M=*/16, /*efConstruction=*/200);

index.addPoint(vec.data(), /*label=*/42);

auto results = index.search(query.data(), /*k=*/10, /*ef=*/100);
for (auto& r : results)
    std::cout << r.label << "  " << r.distance << "\n";

index.save("index.bin");
auto restored = Index::load("index.bin");
```

---

## 2. File map

| File | What it is |
|---|---|
| `include/hnsw.hpp` | The whole algorithm. Header-only, ~600 lines. |
| `include/hnsw_c.h` | Flat C ABI. The only thing Go ever sees. |
| `src/hnsw_c.cpp` | C ABI implementation, translates exceptions to error codes. |
| `src/benchmark.cpp` | Recall@k vs brute force, latency percentiles, efSearch sweep. |
| `src/stress.cpp` | Concurrent writers + readers, for running under ThreadSanitizer. |
| `go/hnsw.go` | cgo binding. |

---

## 3. Data layout — the part that decides whether you hit 50 ms

Before any algorithm, the storage. This is where most from-scratch
implementations quietly lose.

### 3.1 Vectors live in one flat arena

```cpp
std::vector<float> data_;   // maxElements * dim, contiguous

inline const float* getData(idx_t id) const {
    return data_.data() + static_cast<size_t>(id) * dim_;
}
```

One allocation. Vector `id` starts at offset `id * dim`. Getting a vector is
pointer arithmetic — no hash lookup, no indirection.

**Why this matters.** The obvious alternative is
`unordered_map<int, vector<double>>`. That costs you, per distance computation:
a hash of the key, a bucket probe, a pointer dereference to the `vector`
control block, and a second dereference to its heap buffer — which lives
somewhere unrelated in memory. A search at `ef=100` does thousands of distance
computations. Every one is a cache miss chain. The flat arena means consecutive
IDs are often on the same cache line, and the hardware prefetcher can actually
help you.

**`float`, not `double`.** Embeddings come out of models as float32.
Storing float64 doubles your memory for zero recall benefit. At 1M × 768:

| | float32 | float64 |
|---|---|---|
| vectors | 2.9 GB | 5.9 GB |

That difference decides whether your 4-shard plan needs 4 machines or 8.

### 3.2 Adjacency lists are fixed-stride arrays

```cpp
std::vector<uint32_t> linksL0_;                    // maxElements * (2M + 1)
std::vector<std::vector<uint32_t>> linksUpper_;    // per node, level * (M + 1)
```

Each node's neighbour list at a given level is a block of `uint32_t`. **Slot 0
holds the count**, slots 1..count hold neighbour IDs:

```
node 5, layer 0:  [ 3 | 17 | 402 | 91 | ? | ? | ... ]
                    ^    ^-------------^
                  count      neighbours
```

```cpp
inline uint32_t* linkBlock(idx_t id, int level) {
    return level == 0
        ? linksL0_.data() + static_cast<size_t>(id) * strideL0_
        : linksUpper_[id].data() + static_cast<size_t>(level - 1) * strideUpper_;
}
```

Layer 0 is separated out because it's the hot path — every query ends there, and
every node exists there. It gets one big contiguous array. Upper layers are
sparse (most nodes only exist on layer 0), so allocating `maxElements × maxLevel`
would waste enormous memory; those get a per-node vector, allocated only if the
node actually has upper levels.

### 3.3 `Mmax0 = 2M`

Layer 0 allows twice the out-degree of upper layers:

```cpp
maxM_(M), maxM0_(M * 2)
```

This is straight from the paper. Layer 0 is where precision happens and where
all nodes live, so it can afford — and benefits from — a denser graph. Upper
layers just need to be navigable, not precise.

### 3.4 No payloads in the index

The index maps `vector -> uint64 label`. It does not store your document text.

That's deliberate, and it's the right call for your architecture: the C++ core
should be a pure numeric index. Document text, metadata, and filters live in the
Go layer's metadata store. Reasons:

- Strings across the cgo boundary are expensive and allocation-heavy.
- Your snapshot format stays a fixed-size binary blob you can `memcpy`.
- Metadata changes (retitling a doc) shouldn't dirty the index.
- Sharding is cleaner — the index shards on vector, metadata replicates
  separately.

---

## 4. Distance

```cpp
inline float dotProduct(const float* a, const float* b, size_t n) {
#if defined(__AVX__)
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(a + i),
                                               _mm256_loadu_ps(b + i)));
    }
    // ... horizontal sum, then scalar tail
```

Eight floats per instruction instead of one. On a 128-dim vector that's 16
iterations instead of 128. The `#if defined(__AVX__)` guard with a scalar
fallback means it still compiles anywhere.

### The normalization trick

Cosine similarity is:

```
cos(a, b) = (a · b) / (‖a‖ · ‖b‖)
```

Your original recomputed `‖a‖` and `‖b‖` on **every single comparison**. But `a`
never changes between comparisons — it's the same stored vector every time. So:
normalize once, at insert:

```cpp
float* dst = mutableData(id);
std::memcpy(dst, vec, dim_ * sizeof(float));
if (space_ == Space::Cosine) normalizeInPlace(dst);
```

Now `‖a‖ = ‖b‖ = 1` and cosine similarity collapses to a bare dot product:

```cpp
inline float cosineDistance(const float* a, const float* b, size_t n) {
    return 1.0f - dotProduct(a, b, n);
}
```

Two of the three loops in the hot function disappear. That alone is roughly a 3×
speedup, before SIMD.

**The query must be normalized too** — easy to forget, and it silently ruins your
distances:

```cpp
if (space_ == Space::Cosine) {
    qbuf.assign(query, query + dim_);
    normalizeInPlace(qbuf.data());
    q = qbuf.data();
}
```

### Distance, not similarity

Everything internally uses **distance**, where smaller is better. Cosine distance
is `1 - cos`, and L2 is squared Euclidean (no `sqrt` — it's monotonic, so it
doesn't change the ordering, and skipping it saves an instruction per
comparison).

Using distance rather than similarity means one consistent comparison direction
everywhere, which removes a whole category of sign bugs.

---

## 5. Level assignment

```cpp
int randomLevel() {
    double u = uniform(0, 1);
    return static_cast<int>(-std::log(u) * levelMult_);
}
```

with

```cpp
levelMult_(1.0 / std::log(static_cast<double>(M)))
```

This is `floor(-ln(U) · mL)` where `mL = 1/ln(M)`, exactly as the paper
specifies. It produces an exponential distribution where the probability of
reaching level `l` decays like `M^-l`.

**Why not a coin flip?** A coin flip (`while (rand() > 0.5) level++`) gives decay
of `2^-l`. Compare expected node counts at 1M vectors:

| level | coin flip (p=0.5) | paper (M=16) |
|---|---|---|
| 0 | 1,000,000 | 1,000,000 |
| 1 | 500,000 | 62,500 |
| 2 | 250,000 | 3,906 |
| 3 | 125,000 | 244 |
| 4 | 62,500 | 15 |

With the coin flip, layer 1 has half a million nodes. The "express lane" isn't
express — traversing it costs almost as much as layer 0, and you've built ~20
layers of that. The whole point of the hierarchy evaporates.

The decay must be tied to `M`, because `M` is what determines how much ground a
single hop covers. That's the entire content of `mL = 1/ln(M)`.

---

## 6. Search

Two different search procedures, used at different layers.

### 6.1 Greedy descent (upper layers, ef = 1)

```cpp
idx_t greedyDescend(const float* q, idx_t qid, idx_t entry,
                    dist_t& curDist, int level) const {
    idx_t cur = entry;
    curDist = distance(q, qid, cur);
    bool changed = true;
    while (changed) {
        changed = false;
        std::lock_guard<std::mutex> g(linkLocks_[cur]);
        const uint32_t* blk = linkBlock(cur, level);
        uint32_t n = blk[0];
        for (uint32_t i = 0; i < n; ++i) {
            idx_t cand = blk[i + 1];
            dist_t d = distance(q, qid, cand);
            if (d < curDist) { curDist = d; cur = cand; changed = true; }
        }
    }
    return cur;
}
```

Plain hill-climbing. Look at all my neighbours; if any is closer to the query
than I am, move there; repeat until stuck. When stuck, we're at a **local**
minimum for this layer — and that's fine, because we immediately drop a layer,
where the graph is denser and we can refine.

This is `searchLayer` with `ef = 1`, written out separately because it's the hot
path during descent and avoids all the priority-queue machinery.

### 6.2 Beam search — Algorithm 2 (layer 0)

This is the heart of HNSW.

```cpp
MaxHeap searchLayer(const float* q, idx_t qid, idx_t entry, size_t ef,
                    int level, bool filterDeleted) const {
    auto visited = visitedPool_->acquire();
    visited->reset();

    MaxHeap results;   // best ef found so far  -- top() is the WORST
    MinHeap frontier;  // still to expand       -- top() is the BEST

    dist_t d0 = distance(q, qid, entry);
    frontier.emplace(d0, entry);
    if (!filterDeleted || !deleted_[entry]) results.emplace(d0, entry);
    visited->testAndSet(entry);

    dist_t worst = results.empty() ? FLT_MAX : results.top().first;

    while (!frontier.empty()) {
        Cand best = frontier.top();
        if (best.first > worst && results.size() >= ef) break;   // <-- stop rule
        frontier.pop();

        for (each neighbour cand of best) {
            if (visited->testAndSet(cand)) continue;
            dist_t d = distance(q, qid, cand);
            if (results.size() < ef || d < worst) {
                frontier.emplace(d, cand);
                if (!filterDeleted || !deleted_[cand]) {
                    results.emplace(d, cand);
                    if (results.size() > ef) results.pop();
                }
                worst = results.top().first;
            }
        }
    }
    return results;
}
```

**Two heaps, opposite orientations.** This confuses everyone the first time:

- `frontier` is a **min-heap**: `top()` is the closest unexplored candidate.
  We always expand the most promising node next.
- `results` is a **max-heap**: `top()` is the *worst* of our current best `ef`.
  We keep it that way so that discarding the worst is `O(log ef)` — just `pop()`.

```cpp
struct FartherFirst { bool operator()(const Cand& a, const Cand& b) const { return a.first < b.first; } };
struct CloserFirst  { bool operator()(const Cand& a, const Cand& b) const { return a.first > b.first; } };
```

(`std::priority_queue` is a max-heap under the given comparator, so inverting the
comparator gives a min-heap. Hence the reversed-looking operators.)

**The stop rule** is the clever bit:

```cpp
if (best.first > worst && results.size() >= ef) break;
```

Read it as: *the most promising thing left to explore is already farther away
than the worst result I'm holding, and I'm holding a full set.* Since we expand
in order of promise, nothing after this can be better either. Stop.

This is what makes the search adaptive. On an easy query it terminates after a
handful of expansions; on a hard one it keeps digging. You are never paying a
fixed cost.

**The visited set** stops us re-expanding nodes in the cycles that graphs are
full of. It's version-stamped rather than cleared:

```cpp
void reset() {
    ++cur_;
    if (cur_ == 0) { std::fill(marks_.begin(), marks_.end(), 0); cur_ = 1; }
}
inline bool testAndSet(size_t i) {
    if (marks_[i] == cur_) return true;
    marks_[i] = cur_;
    return false;
}
```

Resetting is `++cur_` — `O(1)`, not `O(n)`. Only every 65,536th search pays for
an actual clear. And the lists are pooled, so a query never allocates.

### 6.3 Putting it together

```cpp
std::vector<SearchResult> search(const float* query, size_t k, size_t ef) const {
    idx_t cur = entryPoint_;
    dist_t curDist = distance(q, kNone, cur);

    for (int l = curMaxLevel; l > 0; --l)          // greedy down to layer 1
        cur = greedyDescend(q, kNone, cur, curDist, l);

    auto top = searchLayer(q, kNone, cur, std::max(ef, k), 0, true);
    while (top.size() > k) top.pop();
    // ... unpack heap backwards, since pop() gives farthest first
}
```

**One entry point.** Not a scan for all top-layer nodes — a single stored
`entryPoint_`, the node with the highest level. That's the difference between
`O(log n)` and `O(n)` per query, and between an index that builds in minutes and
one that never finishes.

---

## 7. Neighbour selection — Algorithm 4

If you only take one thing from this document, take this one.

When we insert a node and find its `efConstruction` nearest candidates, which
`M` do we actually link to?

**The naive answer** — take the `M` closest — produces a bad graph. Here's why:

```
   query point X sits inside a tight cluster:

        o o o o
       o o X o o          all M nearest neighbours are inside the cluster
        o o o o

                              . . . . . . . .  (a different cluster, far away)
```

X links to `M` nodes that are all in the same tight blob. So does every other
node in the blob. The blob becomes an internally dense, externally isolated
island. Greedy search that enters the wrong blob **can never leave it** — every
neighbour of every node is inside. Recall collapses, and worse, it collapses
non-uniformly: some queries get perfect results, others get garbage.

**The heuristic** fixes this with one rule: *keep candidate `c` only if `c` is
closer to me than it is to anything I've already selected.*

```cpp
for (const auto& c : cands) {              // sorted nearest-first
    if (kept.size() >= M) break;
    bool keep = true;
    for (idx_t r : kept) {
        if (rawDistance(getData(c.second), getData(r)) < c.first) {
            keep = false;                  // c is closer to an existing pick
            break;                         // than to me -- r already covers it
        }
    }
    if (keep) kept.push_back(c.second);
    else      discarded.push_back(c);
}
```

Think of it as **coverage rather than proximity**. If I've already linked to
`r`, and candidate `c` is right next to `r`, then linking to `c` buys me almost
nothing — I can reach `c` by going through `r` in one extra hop. That link slot
is better spent on something in a *direction* I don't yet cover.

The result is that each node keeps a spread of links pointing in different
directions, including some longer ones. Those long links are precisely the
"express lanes" that make the graph navigable. **Empirically this is worth
10–20 points of recall at the same `M`** versus naive top-M — and the gap widens
as the data gets more clustered, which real embeddings very much are.

The backfill at the end:

```cpp
for (const auto& d : discarded) {
    if (kept.size() >= M) break;
    kept.push_back(d.second);
}
```

If the heuristic was so aggressive that we ended up with fewer than `M` links, we
top up from the rejects. Better a mediocre link than an under-connected node.
(This is `keepPrunedConnections` in the reference implementation.)

### The same heuristic on pruning

When we link `X -> Y`, we also link `Y -> X`. But `Y` may already be at
capacity. The lazy fix is to evict `Y`'s farthest neighbour. **Don't.** Re-run
the heuristic over the full set:

```cpp
MaxHeap pool;
pool.emplace(rawDistance(getData(n), getData(id)), id);   // the newcomer
for (uint32_t i = 0; i < deg; ++i) {                      // plus incumbents
    idx_t e = blk[i + 1];
    pool.emplace(rawDistance(getData(n), getData(e)), e);
}
setLinks(n, level, selectNeighborsHeuristic(std::move(pool), maxDeg));
```

Evicting the farthest neighbour systematically destroys long-range links —
exactly the ones you need. Re-running the heuristic protects them, because a
distant neighbour in a direction nothing else covers survives the rule.

---

## 8. Insert — Algorithm 1

```cpp
idx_t addPoint(const float* vec, label_t label) {
    idx_t id = count_.fetch_add(1);

    float* dst = mutableData(id);
    std::memcpy(dst, vec, dim_ * sizeof(float));
    if (space_ == Space::Cosine) normalizeInPlace(dst);

    int level = randomLevel();
    levels_[id] = level;

    // read the entry point / max level as they are RIGHT NOW
    int   curMaxLevel = maxLevel_;
    idx_t curEntry    = entryPoint_;

    if (curEntry == kNone) { entryPoint_ = id; maxLevel_ = level; return id; }

    const float* q = getData(id);          // the NORMALIZED copy

    // Phase 1: greedy descent from the old entry point down to level+1
    if (level < curMaxLevel) {
        dist_t curDist = distance(q, id, curEntry);
        for (int l = curMaxLevel; l > level; --l)
            cur = greedyDescend(q, id, cur, curDist, l);
    }

    // Phase 2: beam search + connect, from min(level, maxLevel) down to 0
    for (int l = std::min(level, curMaxLevel); l >= 0; --l) {
        auto top = searchLayer(q, id, cur, efConstruction_, l, false);
        cur = connectNewElement(id, std::move(top), l);
    }

    if (level > curMaxLevel) {             // promote entry point LAST
        entryPoint_ = id;
        maxLevel_ = level;
    }
    return id;
}
```

### The ordering rule that your original version broke

Notice that `maxLevel_` and `entryPoint_` are updated at the **end**, after all
searching is done. This is not stylistic. Your original did:

```cpp
highest_layer = max(highest_layer, max_layer);   // BEFORE
candidates_by_layer = search(embedding, 50);     // searches from highest_layer
```

When a new node drew a level above everything seen so far, the search then
started at a layer where **no existing node lived**. The candidate set came back
empty, the empty-set `continue` propagated it all the way down, and the node was
inserted with zero connections.

Then it got worse. Every subsequent insert seeded its search from that isolated
node — the only node at the top layer — whose neighbour lists were all empty. So
every insert saw exactly one candidate. The graph degenerated into a star
centred on that one node, capped at `M` spokes.

The correct order is: **search with the old entry point, connect, then promote.**
The new node can't help you navigate to itself.

### Two phases, and why they're different

- **Phase 1 (`curMaxLevel` down to `level+1`)** — the new node doesn't exist at
  these layers, so there's nothing to connect. We're purely navigating: get from
  the global entry point into the right *neighbourhood* as cheaply as possible.
  `ef = 1` greedy is enough.

- **Phase 2 (`min(level, maxLevel)` down to `0`)** — the node exists at these
  layers, so we need actual neighbours. Full beam search at `efConstruction`,
  then the heuristic, then bidirectional linking.

The entry point for each layer is the closest neighbour found on the layer above:

```cpp
idx_t nextEntry = selected[0];   // selected is sorted nearest-first
```

Each layer hands the next one a warm start.

### `efConstruction` vs `efSearch`

Two separate knobs, and conflating them is a common mistake.

- `efConstruction` — beam width **while building**. Higher means each node finds
  better neighbours, so the graph itself is better. Pay this cost once.
- `efSearch` — beam width **while querying**. Higher means better recall,
  slower queries. Pay this cost on every request.

A graph built with low `efConstruction` has a hard recall ceiling that no amount
of `efSearch` can fix. Build quality is not recoverable at query time.

---

## 9. Deletes

True deletion from a graph is nasty: remove a node and you may sever the only
path between two regions. So this uses **soft deletes**:

```cpp
bool markDeleted(label_t label) {
    deleted_[it->second] = 1;
    ++numDeleted_;
}
```

The node stays in the graph as a connector — traversal still passes through it —
but it's filtered from results:

```cpp
if (!filterDeleted || !deleted_[cand]) {
    results.emplace(d, cand);
```

Note `filterDeleted` is `false` during construction. New nodes may legitimately
link to deleted ones, because their value as connectors is unchanged.

The cost is that deleted vectors still consume memory and still cost distance
computations. Once the tombstone ratio gets high (say >20%), you rebuild the
index from live vectors. **In your architecture that's a compaction job in the
Go layer**, analogous to LSM compaction: build a fresh index from the WAL +
snapshot filtered by tombstones, then atomically swap it in.

---

## 10. Persistence

```cpp
void save(const std::string& path) const {
    w(&magic, 4); w(&version, 4); w(&sp, 4);
    w(&dim, 8); w(&maxEl, 8); w(&m, 8); w(&efc, 8); w(&cnt, 8); w(&ndel, 8);
    w(&ml, 4); w(&ep, 4);

    w(data_.data(), cnt * dim_ * sizeof(float));
    w(levels_.data(), cnt * sizeof(int32_t));
    // ...
}
```

Because the layout is flat, the snapshot is essentially a `memcpy` of a few
arrays. Notice:

- A **magic number** so you fail loudly on a wrong or corrupt file.
- A **version field** so a format change is a clear error, not silent corruption.
- Only `count_` elements are written, not the full preallocated capacity — an
  index at 10% capacity produces a 10% snapshot.

`labelToId_` is *not* serialized; it's rebuilt on load. Never persist a hash map
— its layout is implementation-defined and you'd be pinning yourself to one
standard library version.

This is the piece your snapshot layer calls. The WAL handles writes since the
last snapshot; recovery is `load(snapshot)` then replay the WAL tail.

---

## 11. Concurrency

The Go layer will hit one index from many goroutines. The locking, from
coarsest to finest:

```cpp
mutable std::vector<std::mutex>  linkLocks_;   // one per node
mutable std::shared_mutex        entryLock_;   // entry point / max level
mutable std::mutex               labelLock_;   // label -> id map
```

**Per-node link locks.** Two inserts touching different regions of the graph
never contend. This is what makes concurrent build actually scale.

**Shared mutex on the entry point.** Reads take a shared lock (all queries
proceed in parallel). Only the rare insert that raises `maxLevel_` takes the
exclusive lock — with `mL = 1/ln(16)`, that's roughly one insert in 16.

**Capacity is preallocated.** This is the quiet but critical one:

```cpp
data_.assign(maxElements_ * dim_, 0.0f);
```

If the arena could reallocate, a concurrent reader holding `const float*` from
`getData()` would be pointing into freed memory. Preallocating removes the
hazard entirely rather than trying to synchronize around it. The price is that
capacity is fixed at construction — which is fine, because your shard sizes are
a deployment decision anyway.

**Copy links under the lock, compute distances outside it:**

```cpp
uint32_t buf[kMaxDegree];
std::memcpy(buf, blk + 1, n * sizeof(uint32_t));
g.unlock();
for (uint32_t i = 0; i < n; ++i) { /* distance computations */ }
```

The neighbour list is tiny (≤32 entries); copying it is nanoseconds. Holding a
lock across `n` SIMD distance computations would serialize the whole search.

Verify with ThreadSanitizer, not by reasoning:

```bash
g++ -std=c++17 -O1 -g -fsanitize=thread -Iinclude src/stress.cpp -o stress -pthread
./stress
```

---

## 12. The cgo boundary

cgo cannot see C++ classes, templates, or exceptions. So `hnsw_c.h` exposes a
flat C ABI over opaque pointers:

```c
typedef struct HnswIndex HnswIndex;

HnswIndex* hnsw_new(int space, size_t dim, size_t max_elements,
                    size_t M, size_t ef_construction, uint64_t seed);
int hnsw_search(HnswIndex* idx, const float* query, size_t k, size_t ef,
                uint64_t* out_labels, float* out_distances);
```

Three rules this obeys:

**1. No exception may cross the boundary.** An exception unwinding into Go's
stack is undefined behaviour and produces crashes that are genuinely miserable to
debug. Every entry point is wrapped:

```cpp
try {
    cast(idx)->addPoint(vec, label);
    return HNSW_OK;
} catch (const std::exception& e) {
    setError(e.what());
    return HNSW_ERR_GENERIC;
} catch (...) { ... }
```

**2. Error detail via thread-local string.** The return code says *that* it
failed; `hnsw_last_error()` says why. Thread-local so concurrent goroutines don't
clobber each other's messages.

**3. Caller allocates output buffers.** `hnsw_search` writes into arrays Go
already owns. If C++ allocated and returned a buffer, Go would have to call back
into C to free it — one forgotten free per query and you leak until OOM.

On the Go side:

```go
labels := make([]uint64, k)
dists  := make([]float32, k)

n := C.hnsw_search(i.ptr,
    (*C.float)(unsafe.Pointer(&query[0])),
    C.size_t(k), C.size_t(ef),
    (*C.uint64_t)(unsafe.Pointer(&labels[0])),
    (*C.float)(unsafe.Pointer(&dists[0])))
runtime.KeepAlive(query)
```

`&query[0]` is legal to pass to C because `[]float32` contains no Go pointers
(cgo pointer passing rule 1). `runtime.KeepAlive` stops the GC collecting the
slice while C still holds the pointer.

**Watch the call overhead.** A cgo call costs roughly 50–100 ns. Irrelevant next
to a 1 ms search; very relevant if you ever loop `hnsw_add` per vector from Go.
For bulk loading, add a batch entry point that takes a pointer to `n` contiguous
vectors and does the loop in C++.

---

## 13. Tuning

| Knob | Effect | Sensible range |
|---|---|---|
| `M` | out-degree; ↑ recall, ↑ memory, ↑ build time | 12–48 (16 default, 32–48 for high dim) |
| `efConstruction` | build beam width; ↑ graph quality, ↑ build time only | 100–500 |
| `efSearch` | query beam width; ↑ recall, ↑ latency | 50–500, tune per SLA |

Method: fix `M` and `efConstruction`, sweep `efSearch`, plot recall against p99
latency. That curve **is** your result — it's the graph that goes in the README
and the thing an interviewer will ask about. `benchmark.cpp` produces it.

If the curve plateaus below your recall target no matter how high `efSearch`
goes, the *graph* is the limit: raise `M`, then `efConstruction`. If recall is
fine but latency is too high, lower `efSearch` or shard further.

### Memory for 1M vectors

```
vectors:   1M × dim × 4 bytes
layer 0:   1M × (2M + 1) × 4 bytes         = 132 MB at M=16
upper:     ~1/15 of nodes, small
```

| dim | vectors | links (M=16) | total |
|---|---|---|---|
| 128 | 0.51 GB | 0.13 GB | ~0.65 GB |
| 384 | 1.54 GB | 0.13 GB | ~1.7 GB |
| 768 | 3.07 GB | 0.13 GB | ~3.2 GB |
| 1536 | 6.14 GB | 0.13 GB | ~6.3 GB |

`memoryBytes()` reports the real number. Note this confirms your earlier sizing:
at 768+ dims, 1M vectors on one node is uncomfortable, and 4-way sharding is
justified on memory grounds alone.

### A benchmarking trap

Uniformly random vectors are **near-worst-case** for any ANN algorithm. In high
dimensions random points are all roughly equidistant, so there is barely a
"nearest" neighbour to find. On this implementation:

| data | recall@10 at ef=200 |
|---|---|
| random Gaussian, 128d | 0.83 |
| random Gaussian, 32d | 0.996 |
| random Gaussian, 16d | 1.000 |

Nothing is wrong at 128d — that's the curse of dimensionality. Real embeddings
have far lower *intrinsic* dimensionality (they lie on a low-dimensional
manifold inside the ambient space) and behave much more like the 32d row.

**Benchmark on real data**: SIFT1M (128d, L2) or GloVe (100d, cosine) from
[ann-benchmarks](https://github.com/erikbern/ann-benchmarks), which ship with
precomputed ground truth. Quoting recall numbers from random data will
understate your system and invite an awkward question.

---

## 14. What changed from your original

| | Original | Now |
|---|---|---|
| Entry point | scan all nodes, `O(N)` per query | single stored entry point, `O(log N)` |
| Index build | `O(N²)` — won't finish at 1M | `O(N log N)` |
| Layer search | one-hop expansion + sort | best-first beam search with visited set |
| Neighbour selection | top-M by similarity | Algorithm 4 heuristic |
| Pruning | evict farthest | re-run heuristic |
| Level decay | `p=0.5` coin flip | `-ln(U)/ln(M)` |
| Entry-point update | before search (isolates nodes) | after connect |
| Vector storage | `unordered_map<int, vector<double>>` | flat `float` arena |
| Adjacency | nested `unordered_map` | fixed-stride `uint32_t` arrays |
| Layer-0 degree | `M` | `2M` |
| Distance | recomputed magnitudes, scalar | pre-normalized, AVX2 |
| Threading | none | per-node locks + shared entry lock |
| Persistence | none | versioned binary snapshot |
| Deletes | none | tombstones |
| Go integration | none | `extern "C"` ABI + cgo binding |

---

## 15. Where this fits in the project

The C++ core is done. What's left, roughly in order:

1. **Benchmark on SIFT1M.** Produces your recall/latency curve. Confirms the
   95% recall@10 and p99 < 50 ms targets are reachable single-node before you
   add any distribution.
2. **WAL.** Append `(label, vector)` before `addPoint` returns. Recovery is
   snapshot + WAL tail replay.
3. **Snapshot scheduling.** Periodic `save()`, truncate WAL on success. Take it
   on a copy or during a brief write-quiesce so you don't snapshot a torn graph.
4. **Go shard server.** gRPC over the cgo binding. One index per shard.
5. **Coordinator.** Scatter-gather across shard primaries, merge the `k`-lists
   by distance. Merging is trivial *because* every shard returns distances in a
   consistent space — one more reason distances beat similarities.
6. **Replication.** Async replay of the WAL to replicas.

One thing to decide early: **`maxElements` is fixed per index**. When a shard
fills, you either over-provision at creation or implement index rotation
(build a second index, search both, merge, retire the old one). Over-provisioning
is fine for the project; note rotation as future work.

---

## References

- Malkov & Yashunin, *Efficient and robust approximate nearest neighbor search
  using Hierarchical Navigable Small World graphs*, arXiv:1603.09320. Algorithms
  1–5 are two pages and map one-to-one onto this code.
- `hnswlib` — the authors' reference C++ implementation. Worth diffing against
  once you've understood this one.
