// hnsw_c.cpp -- implementation of the C ABI.
//
// The one rule here: no exception may cross the extern "C" boundary. An
// exception unwinding into Go's stack is undefined behaviour and will crash the
// process in ways that are extremely unpleasant to debug. Every entry point is
// wrapped.

#include "hnsw_c.h"
#include "hnsw.hpp"

#include <string>

namespace {

thread_local std::string g_lastError;

void setError(const char* msg) { g_lastError = msg; }

inline hnsw::Index* cast(HnswIndex* p) { return reinterpret_cast<hnsw::Index*>(p); }

} // namespace

extern "C" {

HnswIndex* hnsw_new(int space, size_t dim, size_t max_elements,
                    size_t M, size_t ef_construction, uint64_t seed) {
    try {
        auto* idx = new hnsw::Index(
            space == HNSW_SPACE_L2 ? hnsw::Space::L2 : hnsw::Space::Cosine,
            dim, max_elements, M, ef_construction, seed);
        return reinterpret_cast<HnswIndex*>(idx);
    } catch (const std::exception& e) {
        setError(e.what());
        return nullptr;
    } catch (...) {
        setError("unknown error in hnsw_new");
        return nullptr;
    }
}

void hnsw_free(HnswIndex* idx) {
    delete cast(idx);
}

int hnsw_add(HnswIndex* idx, const float* vec, uint64_t label) {
    if (!idx || !vec) { setError("null argument"); return HNSW_ERR_NULL; }
    try {
        cast(idx)->addPoint(vec, label);
        return HNSW_OK;
    } catch (const std::exception& e) {
        setError(e.what());
        return std::string(e.what()).find("full") != std::string::npos
                   ? HNSW_ERR_FULL : HNSW_ERR_GENERIC;
    } catch (...) {
        setError("unknown error in hnsw_add");
        return HNSW_ERR_GENERIC;
    }
}

int hnsw_search(HnswIndex* idx, const float* query, size_t k, size_t ef,
                uint64_t* out_labels, float* out_distances) {
    if (!idx || !query || !out_labels || !out_distances) {
        setError("null argument");
        return HNSW_ERR_NULL;
    }
    try {
        auto res = cast(idx)->search(query, k, ef);
        for (size_t i = 0; i < res.size(); ++i) {
            out_labels[i] = res[i].label;
            out_distances[i] = res[i].distance;
        }
        return static_cast<int>(res.size());
    } catch (const std::exception& e) {
        setError(e.what());
        return HNSW_ERR_GENERIC;
    } catch (...) {
        setError("unknown error in hnsw_search");
        return HNSW_ERR_GENERIC;
    }
}

int hnsw_mark_deleted(HnswIndex* idx, uint64_t label) {
    if (!idx) { setError("null index"); return HNSW_ERR_NULL; }
    try {
        return cast(idx)->markDeleted(label) ? HNSW_OK : HNSW_ERR_NOT_FOUND;
    } catch (...) {
        setError("unknown error in hnsw_mark_deleted");
        return HNSW_ERR_GENERIC;
    }
}

int hnsw_unmark_deleted(HnswIndex* idx, uint64_t label) {
    if (!idx) { setError("null index"); return HNSW_ERR_NULL; }
    try {
        return cast(idx)->unmarkDeleted(label) ? HNSW_OK : HNSW_ERR_NOT_FOUND;
    } catch (...) {
        setError("unknown error in hnsw_unmark_deleted");
        return HNSW_ERR_GENERIC;
    }
}

size_t hnsw_size(HnswIndex* idx)          { return idx ? cast(idx)->size() : 0; }
size_t hnsw_active_size(HnswIndex* idx)   { return idx ? cast(idx)->activeSize() : 0; }
size_t hnsw_dim(HnswIndex* idx)           { return idx ? cast(idx)->dim() : 0; }
size_t hnsw_capacity(HnswIndex* idx)      { return idx ? cast(idx)->capacity() : 0; }
size_t hnsw_memory_bytes(HnswIndex* idx)  { return idx ? cast(idx)->memoryBytes() : 0; }

int hnsw_save(HnswIndex* idx, const char* path) {
    if (!idx || !path) { setError("null argument"); return HNSW_ERR_NULL; }
    try {
        cast(idx)->save(path);
        return HNSW_OK;
    } catch (const std::exception& e) {
        setError(e.what());
        return HNSW_ERR_GENERIC;
    } catch (...) {
        setError("unknown error in hnsw_save");
        return HNSW_ERR_GENERIC;
    }
}

HnswIndex* hnsw_load(const char* path) {
    if (!path) { setError("null path"); return nullptr; }
    try {
        auto idx = hnsw::Index::load(path);
        return reinterpret_cast<HnswIndex*>(idx.release());
    } catch (const std::exception& e) {
        setError(e.what());
        return nullptr;
    } catch (...) {
        setError("unknown error in hnsw_load");
        return nullptr;
    }
}

const char* hnsw_last_error(void) {
    return g_lastError.empty() ? "" : g_lastError.c_str();
}

} // extern "C"
