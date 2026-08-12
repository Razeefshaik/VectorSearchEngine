// hnsw_c.h -- flat C ABI over the C++ index, for cgo.
//
// cgo cannot call C++ classes, templates, or anything that throws. Every
// function here is extern "C", takes/returns POD, and converts exceptions into
// negative return codes. This file is the ONLY thing Go ever sees.

#ifndef HNSW_C_H
#define HNSW_C_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct HnswIndex HnswIndex;

#define HNSW_OK             0
#define HNSW_ERR_GENERIC   -1
#define HNSW_ERR_NULL      -2
#define HNSW_ERR_FULL      -3
#define HNSW_ERR_NOT_FOUND -4

#define HNSW_SPACE_COSINE 0
#define HNSW_SPACE_L2     1

// Returns NULL on failure; call hnsw_last_error() for the reason.
HnswIndex* hnsw_new(int space, size_t dim, size_t max_elements,
                    size_t M, size_t ef_construction, uint64_t seed);

void hnsw_free(HnswIndex* idx);

// Adds one vector of length dim. Returns HNSW_OK or a negative code.
int hnsw_add(HnswIndex* idx, const float* vec, uint64_t label);

// Writes up to k results into out_labels / out_distances (caller-allocated,
// each of length >= k). Returns the number written, or a negative code.
int hnsw_search(HnswIndex* idx, const float* query, size_t k, size_t ef,
                uint64_t* out_labels, float* out_distances);

int hnsw_mark_deleted(HnswIndex* idx, uint64_t label);
int hnsw_unmark_deleted(HnswIndex* idx, uint64_t label);

size_t hnsw_size(HnswIndex* idx);
size_t hnsw_active_size(HnswIndex* idx);
size_t hnsw_dim(HnswIndex* idx);
size_t hnsw_capacity(HnswIndex* idx);
size_t hnsw_memory_bytes(HnswIndex* idx);

int        hnsw_save(HnswIndex* idx, const char* path);
HnswIndex* hnsw_load(const char* path);

// Thread-local; valid until the next failing call on the same thread.
const char* hnsw_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // HNSW_C_H
