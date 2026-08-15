

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <random>
#include <shared_mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__AVX__)
#include <immintrin.h>
#endif

namespace hnsw {

using idx_t = uint32_t;
using dist_t = float;

// A label is scoped to the client that chose it: two different clients can
// legitimately pick the same numeric label for two unrelated vectors, so
// identity is the (clientId, label) pair, not label alone. Uniqueness is
// guaranteed by operator== (exact struct equality), not by LabelHash -- the
// hash is free to collide internally, since std::unordered_map already
// resolves same-bucket collisions via operator==.
struct Label {
    uint64_t clientId;
    uint64_t label;

    bool operator==(const Label& other) const {
        return clientId == other.clientId && label == other.label;
    }
    bool operator!=(const Label& other) const { return !(*this == other); }
};

struct LabelHash {
    size_t operator()(const Label& l) const {
        size_t h1 = std::hash<uint64_t>{}(l.clientId);
        size_t h2 = std::hash<uint64_t>{}(l.label);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
    }
};


enum class Space : int { Cosine = 0, L2 = 1 };





inline float dotProduct(const float* a, const float* b, size_t n) {
#if defined(__AVX__)
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(a + i),
                                               _mm256_loadu_ps(b + i)));
    }
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc),
                           _mm256_extractf128_ps(acc, 1));
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float res = _mm_cvtss_f32(lo);
    for (; i < n; ++i) res += a[i] * b[i];
    return res;
#else
    float res = 0.0f;
    for (size_t i = 0; i < n; ++i) res += a[i] * b[i];
    return res;
#endif
}

inline float l2SquaredDistance(const float* a, const float* b, size_t n) {
#if defined(__AVX__)
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 d = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
        acc = _mm256_add_ps(acc, _mm256_mul_ps(d, d));
    }
    __m128 lo = _mm_add_ps(_mm256_castps256_ps128(acc),
                           _mm256_extractf128_ps(acc, 1));
    lo = _mm_hadd_ps(lo, lo);
    lo = _mm_hadd_ps(lo, lo);
    float res = _mm_cvtss_f32(lo);
    for (; i < n; ++i) { float d = a[i] - b[i]; res += d * d; }
    return res;
#else
    float res = 0.0f;
    for (size_t i = 0; i < n; ++i) { float d = a[i] - b[i]; res += d * d; }
    return res;
#endif
}


inline float cosineDistance(const float* a, const float* b, size_t n) {
    return 1.0f - dotProduct(a, b, n);
}





class VisitedList {
public:
    explicit VisitedList(size_t n) : marks_(n, 0), cur_(0) {}

    void reset() {
        ++cur_;
        if (cur_ == 0) {                       
            std::fill(marks_.begin(), marks_.end(), 0);
            cur_ = 1;
        }
    }
    inline bool testAndSet(size_t i) {
        if (marks_[i] == cur_) return true;    
        marks_[i] = cur_;
        return false;
    }

private:
    std::vector<uint16_t> marks_;
    uint16_t cur_;
};

class VisitedListPool {
public:
    VisitedListPool(size_t initial, size_t n) : n_(n) {
        for (size_t i = 0; i < initial; ++i)
            pool_.push_back(std::make_unique<VisitedList>(n_));
    }

    std::unique_ptr<VisitedList> acquire() {
        std::lock_guard<std::mutex> g(m_);
        if (pool_.empty()) return std::make_unique<VisitedList>(n_);
        auto v = std::move(pool_.back());
        pool_.pop_back();
        v->reset();
        return v;
    }
    void release(std::unique_ptr<VisitedList> v) {
        std::lock_guard<std::mutex> g(m_);
        pool_.push_back(std::move(v));
    }

private:
    std::vector<std::unique_ptr<VisitedList>> pool_;
    std::mutex m_;
    size_t n_;
};





class Index {
public:
    using Cand = std::pair<dist_t, idx_t>;

    struct SearchResult {
        Label  label;
        dist_t distance;
    };

    
    
    Index(Space space,
          size_t dim,
          size_t maxElements,
          size_t M = 16,
          size_t efConstruction = 200,
          uint64_t seed = 100)
        : space_(space),
          dim_(dim),
          maxElements_(maxElements),
          M_(M),
          maxM_(M),
          maxM0_(M * 2),
          efConstruction_(std::max<size_t>(efConstruction, M)),
          levelMult_(1.0 / std::log(static_cast<double>(M))),
          rng_(seed) {
        if (dim == 0) throw std::invalid_argument("dim must be > 0");
        if (maxElements == 0) throw std::invalid_argument("maxElements must be > 0");
        if (M == 0 || M * 2 > kMaxDegree)
            throw std::invalid_argument("M must be in 1..kMaxDegree/2");

        data_.assign(maxElements_ * dim_, 0.0f);
        levels_.assign(maxElements_, 0);
        labels_.assign(maxElements_, Label{0, 0});
        deleted_.assign(maxElements_, 0);

        strideL0_ = maxM0_ + 1;              
        strideUpper_ = maxM_ + 1;
        linksL0_.assign(maxElements_ * strideL0_, 0);
        linksUpper_.resize(maxElements_);

        linkLocks_ = std::vector<std::mutex>(maxElements_);
        visitedPool_ = std::make_unique<VisitedListPool>(1, maxElements_);
    }

    

    
    idx_t addPoint(const float* vec, Label label) {
        {
            std::lock_guard<std::mutex> g(labelLock_);
            if (labelToId_.count(label))
                throw std::runtime_error("duplicate label; use updatePoint or a new label");
        }

        idx_t id = static_cast<idx_t>(count_.fetch_add(1));
        if (id >= maxElements_) {
            count_.fetch_sub(1);
            throw std::runtime_error("index is full: raise maxElements");
        }

        
        float* dst = mutableData(id);
        std::memcpy(dst, vec, dim_ * sizeof(float));
        if (space_ == Space::Cosine) normalizeInPlace(dst);

        labels_[id] = label;
        deleted_[id] = 0;
        {
            std::lock_guard<std::mutex> g(labelLock_);
            labelToId_[label] = id;
        }

        int level = randomLevel();
        levels_[id] = level;
        if (level > 0) linksUpper_[id].assign(static_cast<size_t>(level) * strideUpper_, 0);

        
        
        std::unique_lock<std::shared_mutex> entryWrite(entryLock_, std::defer_lock);
        std::shared_lock<std::shared_mutex> entryRead(entryLock_, std::defer_lock);
        int  curMaxLevel;
        idx_t curEntry;
        bool becomesEntry;
        {
            std::shared_lock<std::shared_mutex> probe(entryLock_);
            curMaxLevel = maxLevel_;
            curEntry = entryPoint_;
            becomesEntry = (entryPoint_ == kNone) || (level > curMaxLevel);
        }
        if (becomesEntry) { entryWrite.lock(); curMaxLevel = maxLevel_; curEntry = entryPoint_; }
        else              { entryRead.lock();  curMaxLevel = maxLevel_; curEntry = entryPoint_; }

        if (curEntry == kNone) {                       
            entryPoint_ = id;
            maxLevel_ = level;
            return id;
        }

        const float* q = getData(id);   

        
        
        
        idx_t cur = curEntry;
        if (level < curMaxLevel) {
            dist_t curDist = distance(q, id, cur);
            for (int l = curMaxLevel; l > level; --l)
                cur = greedyDescend(q, id, cur, curDist, l);
        }

        
        for (int l = std::min(level, curMaxLevel); l >= 0; --l) {
            auto top = searchLayer(q, id, cur, efConstruction_, l, false);
            cur = connectNewElement(id, std::move(top), l);
        }

        if (level > curMaxLevel) {                     
            entryPoint_ = id;
            maxLevel_ = level;
        }
        return id;
    }

    
    
    bool markDeleted(Label label) {
        std::lock_guard<std::mutex> g(labelLock_);
        auto it = labelToId_.find(label);
        if (it == labelToId_.end()) return false;
        if (deleted_[it->second]) return false;
        deleted_[it->second] = 1;
        ++numDeleted_;
        return true;
    }

    bool unmarkDeleted(Label label) {
        std::lock_guard<std::mutex> g(labelLock_);
        auto it = labelToId_.find(label);
        if (it == labelToId_.end()) return false;
        if (!deleted_[it->second]) return false;
        deleted_[it->second] = 0;
        --numDeleted_;
        return true;
    }

    

    std::vector<SearchResult> search(const float* query, size_t k, size_t ef) const {
        std::vector<SearchResult> out;
        if (count_.load() == 0) return out;

        
        std::vector<float> qbuf;
        const float* q = query;
        if (space_ == Space::Cosine) {
            qbuf.assign(query, query + dim_);
            normalizeInPlace(qbuf.data());
            q = qbuf.data();
        }

        idx_t cur;
        int curMaxLevel;
        {
            std::shared_lock<std::shared_mutex> g(entryLock_);
            cur = entryPoint_;
            curMaxLevel = maxLevel_;
        }
        if (cur == kNone) return out;

        dist_t curDist = distance(q, kNone, cur);
        for (int l = curMaxLevel; l > 0; --l)
            cur = greedyDescend(q, kNone, cur, curDist, l);

        auto top = searchLayer(q, kNone, cur, std::max(ef, k), 0, true);
        while (top.size() > k) top.pop();

        out.resize(top.size());
        for (size_t i = out.size(); i-- > 0;) {        
            out[i] = SearchResult{labels_[top.top().second], top.top().first};
            top.pop();
        }
        return out;
    }

    

    size_t size() const        { return count_.load(); }
    size_t activeSize() const  { return count_.load() - numDeleted_; }
    size_t dim() const         { return dim_; }
    size_t capacity() const    { return maxElements_; }
    int    maxLevel() const    { return maxLevel_; }
    size_t M() const           { return M_; }
    size_t efConstruction() const { return efConstruction_; }
    Space  space() const       { return space_; }

    size_t memoryBytes() const {
        size_t b = data_.size() * sizeof(float) + linksL0_.size() * sizeof(uint32_t);
        for (const auto& v : linksUpper_) b += v.size() * sizeof(uint32_t);
        return b;
    }

    

    void save(const std::string& path) const {
        std::ofstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + path + " for writing");

        auto w = [&](const void* p, size_t n) { f.write(static_cast<const char*>(p), n); };
        uint32_t magic = kMagic, version = kVersion;
        uint32_t sp = static_cast<uint32_t>(space_);
        uint64_t dim = dim_, maxEl = maxElements_, m = M_, efc = efConstruction_;
        uint64_t cnt = count_.load(), ndel = numDeleted_;
        int32_t  ml = maxLevel_;
        uint32_t ep = entryPoint_;

        w(&magic, 4); w(&version, 4); w(&sp, 4);
        w(&dim, 8); w(&maxEl, 8); w(&m, 8); w(&efc, 8); w(&cnt, 8); w(&ndel, 8);
        w(&ml, 4); w(&ep, 4);

        w(data_.data(), cnt * dim_ * sizeof(float));
        w(levels_.data(), cnt * sizeof(int32_t));
        w(labels_.data(), cnt * sizeof(Label));
        w(deleted_.data(), cnt * sizeof(uint8_t));
        w(linksL0_.data(), cnt * strideL0_ * sizeof(uint32_t));
        for (uint64_t i = 0; i < cnt; ++i) {
            uint32_t n = static_cast<uint32_t>(linksUpper_[i].size());
            w(&n, 4);
            if (n) w(linksUpper_[i].data(), n * sizeof(uint32_t));
        }
        if (!f) throw std::runtime_error("write failed for " + path);
    }

    static std::unique_ptr<Index> load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("cannot open " + path + " for reading");

        auto r = [&](void* p, size_t n) { f.read(static_cast<char*>(p), n); };
        uint32_t magic, version, sp;
        uint64_t dim, maxEl, m, efc, cnt, ndel;
        int32_t  ml;
        uint32_t ep;

        r(&magic, 4); r(&version, 4);
        if (magic != kMagic) throw std::runtime_error("not an hnsw index file");
        if (version != kVersion)
            throw std::runtime_error(
                "index snapshot version " + std::to_string(version) +
                " is not supported (expected " + std::to_string(kVersion) +
                "); this is a breaking format change (Label is now a "
                "16-byte {clientId, label} pair, not an 8-byte uint64) -- "
                "old snapshots cannot be loaded and must be rebuilt");
        r(&sp, 4);
        r(&dim, 8); r(&maxEl, 8); r(&m, 8); r(&efc, 8); r(&cnt, 8); r(&ndel, 8);
        r(&ml, 4); r(&ep, 4);

        auto idx = std::make_unique<Index>(static_cast<Space>(sp), dim, maxEl, m, efc);
        r(idx->data_.data(), cnt * dim * sizeof(float));
        r(idx->levels_.data(), cnt * sizeof(int32_t));
        r(idx->labels_.data(), cnt * sizeof(Label));
        r(idx->deleted_.data(), cnt * sizeof(uint8_t));
        r(idx->linksL0_.data(), cnt * idx->strideL0_ * sizeof(uint32_t));
        for (uint64_t i = 0; i < cnt; ++i) {
            uint32_t n; r(&n, 4);
            idx->linksUpper_[i].resize(n);
            if (n) r(idx->linksUpper_[i].data(), n * sizeof(uint32_t));
        }
        if (!f) throw std::runtime_error("truncated index file " + path);

        idx->count_.store(cnt);
        idx->numDeleted_ = ndel;
        idx->maxLevel_ = ml;
        idx->entryPoint_ = ep;
        for (uint64_t i = 0; i < cnt; ++i) idx->labelToId_[idx->labels_[i]] = static_cast<idx_t>(i);
        return idx;
    }

private:
    static constexpr size_t   kMaxDegree = 256;   
    static constexpr idx_t    kNone = std::numeric_limits<idx_t>::max();
    static constexpr uint32_t kMagic = 0x484E5357;
    static constexpr uint32_t kVersion = 2;   // v2: Label is {clientId, label}, not bare uint64

    
    struct FartherFirst {
        bool operator()(const Cand& a, const Cand& b) const { return a.first < b.first; }
    };
    
    struct CloserFirst {
        bool operator()(const Cand& a, const Cand& b) const { return a.first > b.first; }
    };
    using MaxHeap = std::priority_queue<Cand, std::vector<Cand>, FartherFirst>;
    using MinHeap = std::priority_queue<Cand, std::vector<Cand>, CloserFirst>;

    

    inline const float* getData(idx_t id) const { return data_.data() + static_cast<size_t>(id) * dim_; }
    inline float*   mutableData(idx_t id)       { return data_.data() + static_cast<size_t>(id) * dim_; }

    
    inline uint32_t* linkBlock(idx_t id, int level) {
        return level == 0
            ? linksL0_.data() + static_cast<size_t>(id) * strideL0_
            : linksUpper_[id].data() + static_cast<size_t>(level - 1) * strideUpper_;
    }
    inline const uint32_t* linkBlock(idx_t id, int level) const {
        return level == 0
            ? linksL0_.data() + static_cast<size_t>(id) * strideL0_
            : linksUpper_[id].data() + static_cast<size_t>(level - 1) * strideUpper_;
    }
    inline uint32_t linkCount(idx_t id, int level) const { return linkBlock(id, level)[0]; }
    inline const uint32_t* linkNeighbors(idx_t id, int level) const { return linkBlock(id, level) + 1; }

    void setLinks(idx_t id, int level, const std::vector<idx_t>& ns) {
        uint32_t* blk = linkBlock(id, level);
        size_t cap = (level == 0 ? maxM0_ : maxM_);
        size_t n = std::min(ns.size(), cap);
        blk[0] = static_cast<uint32_t>(n);
        for (size_t i = 0; i < n; ++i) blk[i + 1] = ns[i];
    }

    void normalizeInPlace(float* v) const {
        float norm = std::sqrt(dotProduct(v, v, dim_));
        if (norm <= 1e-30f) return;                  
        float inv = 1.0f / norm;
        for (size_t i = 0; i < dim_; ++i) v[i] *= inv;
    }

    inline dist_t rawDistance(const float* a, const float* b) const {
        return space_ == Space::Cosine ? cosineDistance(a, b, dim_)
                                       : l2SquaredDistance(a, b, dim_);
    }
    
    
    inline dist_t distance(const float* q, idx_t , idx_t target) const {
        return rawDistance(q, getData(target));
    }

    
    
    
    idx_t greedyDescend(const float* q, idx_t qid, idx_t entry, dist_t& curDist, int level) const {
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

    
    
    
    MaxHeap searchLayer(const float* q, idx_t qid, idx_t entry, size_t ef,
                        int level, bool filterDeleted) const {
        auto visited = visitedPool_->acquire();
        visited->reset();

        MaxHeap results;   
        MinHeap frontier;  

        dist_t d0 = distance(q, qid, entry);
        frontier.emplace(d0, entry);
        if (!filterDeleted || !deleted_[entry]) results.emplace(d0, entry);
        visited->testAndSet(entry);

        dist_t worst = results.empty() ? std::numeric_limits<dist_t>::max()
                                       : results.top().first;

        while (!frontier.empty()) {
            Cand best = frontier.top();
            
            if (best.first > worst && results.size() >= ef) break;
            frontier.pop();

            std::unique_lock<std::mutex> g(linkLocks_[best.second]);
            const uint32_t* blk = linkBlock(best.second, level);
            uint32_t n = blk[0];
            
            uint32_t buf[kMaxDegree];
            n = std::min<uint32_t>(n, kMaxDegree);
            std::memcpy(buf, blk + 1, n * sizeof(uint32_t));
            g.unlock();

            for (uint32_t i = 0; i < n; ++i) {
                idx_t cand = buf[i];
                if (visited->testAndSet(cand)) continue;

                dist_t d = distance(q, qid, cand);
                if (results.size() < ef || d < worst) {
                    frontier.emplace(d, cand);
                    if (!filterDeleted || !deleted_[cand]) {
                        results.emplace(d, cand);
                        if (results.size() > ef) results.pop();
                    }
                    if (!results.empty()) worst = results.top().first;
                }
            }
        }
        visitedPool_->release(std::move(visited));
        return results;
    }

    
    
    
    
    std::vector<idx_t> selectNeighborsHeuristic(MaxHeap top, size_t M) const {
        std::vector<Cand> cands;
        cands.reserve(top.size());
        while (!top.empty()) { cands.push_back(top.top()); top.pop(); }
        std::sort(cands.begin(), cands.end());       

        std::vector<idx_t> kept;
        if (cands.size() <= M) {
            kept.reserve(cands.size());
            for (const auto& c : cands) kept.push_back(c.second);
            return kept;
        }

        std::vector<Cand> discarded;
        kept.reserve(M);
        for (const auto& c : cands) {
            if (kept.size() >= M) break;
            bool keep = true;
            for (idx_t r : kept) {
                if (rawDistance(getData(c.second), getData(r)) < c.first) { keep = false; break; }
            }
            if (keep) kept.push_back(c.second);
            else      discarded.push_back(c);
        }
        
        for (const auto& d : discarded) {
            if (kept.size() >= M) break;
            kept.push_back(d.second);
        }
        return kept;
    }

    
    
    
    
    idx_t connectNewElement(idx_t id, MaxHeap top, int level) {
        size_t maxDeg = (level == 0) ? maxM0_ : maxM_;
        std::vector<idx_t> selected = selectNeighborsHeuristic(std::move(top), M_);
        if (selected.empty()) return id;

        idx_t nextEntry = selected[0];
        {
            std::lock_guard<std::mutex> g(linkLocks_[id]);
            setLinks(id, level, selected);
        }

        for (idx_t n : selected) {
            std::lock_guard<std::mutex> g(linkLocks_[n]);
            uint32_t* blk = linkBlock(n, level);
            uint32_t deg = blk[0];

            if (deg < maxDeg) {                    
                blk[deg + 1] = id;
                blk[0] = deg + 1;
                continue;
            }
            
            
            MaxHeap pool;
            pool.emplace(rawDistance(getData(n), getData(id)), id);
            for (uint32_t i = 0; i < deg; ++i) {
                idx_t e = blk[i + 1];
                pool.emplace(rawDistance(getData(n), getData(e)), e);
            }
            setLinks(n, level, selectNeighborsHeuristic(std::move(pool), maxDeg));
        }
        return nextEntry;
    }

    
    
    
    int randomLevel() {
        std::uniform_real_distribution<double> dis(0.0, 1.0);
        double u;
        {
            std::lock_guard<std::mutex> g(rngLock_);
            u = dis(rng_);
        }
        if (u <= 0.0) u = std::numeric_limits<double>::min();
        return static_cast<int>(-std::log(u) * levelMult_);
    }

    
    Space  space_;
    size_t dim_;
    size_t maxElements_;
    size_t M_, maxM_, maxM0_, efConstruction_;
    size_t strideL0_ = 0, strideUpper_ = 0;
    double levelMult_;

    std::vector<float>    data_;        
    std::vector<int32_t>  levels_;
    std::vector<Label>    labels_;
    std::vector<uint8_t>  deleted_;
    std::vector<uint32_t> linksL0_;     
    std::vector<std::vector<uint32_t>> linksUpper_;

    std::atomic<size_t> count_{0};
    size_t numDeleted_ = 0;
    idx_t  entryPoint_ = kNone;
    int    maxLevel_ = 0;

    std::unordered_map<Label, idx_t, LabelHash> labelToId_;

    mutable std::vector<std::mutex>  linkLocks_;
    mutable std::shared_mutex        entryLock_;
    mutable std::mutex               labelLock_;
    std::mutex                       rngLock_;
    std::mt19937                     rng_;
    std::unique_ptr<VisitedListPool> visitedPool_;
};

} 
