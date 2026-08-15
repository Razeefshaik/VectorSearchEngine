






package hnsw

/*
#cgo CXXFLAGS: -std=c++17 -O3 -march=native
#cgo CFLAGS: -I${SRCDIR}/../../include
#cgo LDFLAGS: -L${SRCDIR}/../../build -Wl,-rpath,${SRCDIR}/../../build -lhnsw -lstdc++ -lm
#include <stdlib.h>
#include "hnsw_c.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

type Space int

const (
	Cosine Space = C.HNSW_SPACE_COSINE
	L2     Space = C.HNSW_SPACE_L2
)






var (
	ErrDuplicateLabel = errors.New("hnsw: label already exists")
	ErrNotFound       = errors.New("hnsw: label not found")
	ErrFull           = errors.New("hnsw: index is full")
)


type Index struct {
	ptr *C.HnswIndex
	dim int
}

// Key identifies a vector. Two different clients can pick the same Label
// value for unrelated vectors without colliding -- identity is the whole
// (ClientID, Label) pair, compared exactly, not a hash of it.
type Key struct {
	ClientID uint64
	Label    uint64
}

type Result struct {
	Key      Key
	Distance float32
}

func lastError() error {
	msg := C.GoString(C.hnsw_last_error())
	if msg == "" {
		return errors.New("hnsw: unknown error")
	}
	return fmt.Errorf("hnsw: %s", msg)
}



func New(space Space, dim, maxElements, m, efConstruction int, seed uint64) (*Index, error) {
	ptr := C.hnsw_new(C.int(space), C.size_t(dim), C.size_t(maxElements),
		C.size_t(m), C.size_t(efConstruction), C.uint64_t(seed))
	if ptr == nil {
		return nil, lastError()
	}
	idx := &Index{ptr: ptr, dim: dim}
	runtime.SetFinalizer(idx, (*Index).Close)
	return idx, nil
}

func (i *Index) Close() {
	if i.ptr != nil {
		C.hnsw_free(i.ptr)
		i.ptr = nil
		runtime.SetFinalizer(i, nil)
	}
}


func (i *Index) Add(vec []float32, key Key) error {
	if len(vec) != i.dim {
		return fmt.Errorf("hnsw: expected %d dims, got %d", i.dim, len(vec))
	}


	rc := C.hnsw_add(i.ptr, (*C.float)(unsafe.Pointer(&vec[0])),
		C.uint64_t(key.ClientID), C.uint64_t(key.Label))
	runtime.KeepAlive(vec)
	switch rc {
	case C.HNSW_OK:
		return nil
	case C.HNSW_ERR_DUPLICATE:
		return ErrDuplicateLabel
	case C.HNSW_ERR_FULL:
		return ErrFull
	default:
		return lastError()
	}
}



func (i *Index) Search(query []float32, k, ef int) ([]Result, error) {
	if len(query) != i.dim {
		return nil, fmt.Errorf("hnsw: expected %d dims, got %d", i.dim, len(query))
	}
	if ef < k {
		ef = k
	}
	clientIDs := make([]uint64, k)
	labels := make([]uint64, k)
	dists := make([]float32, k)

	n := C.hnsw_search(i.ptr,
		(*C.float)(unsafe.Pointer(&query[0])),
		C.size_t(k), C.size_t(ef),
		(*C.uint64_t)(unsafe.Pointer(&clientIDs[0])),
		(*C.uint64_t)(unsafe.Pointer(&labels[0])),
		(*C.float)(unsafe.Pointer(&dists[0])))
	runtime.KeepAlive(query)

	if n < 0 {
		return nil, lastError()
	}
	out := make([]Result, n)
	for j := 0; j < int(n); j++ {
		out[j] = Result{Key: Key{ClientID: clientIDs[j], Label: labels[j]}, Distance: dists[j]}
	}
	return out, nil
}

func (i *Index) MarkDeleted(key Key) error {
	switch rc := C.hnsw_mark_deleted(i.ptr, C.uint64_t(key.ClientID), C.uint64_t(key.Label)); rc {
	case C.HNSW_OK:
		return nil
	case C.HNSW_ERR_NOT_FOUND:
		return ErrNotFound
	default:
		return lastError()
	}
}




func (i *Index) UnmarkDeleted(key Key) error {
	switch rc := C.hnsw_unmark_deleted(i.ptr, C.uint64_t(key.ClientID), C.uint64_t(key.Label)); rc {
	case C.HNSW_OK:
		return nil
	case C.HNSW_ERR_NOT_FOUND:
		return ErrNotFound
	default:
		return lastError()
	}
}

func (i *Index) Save(path string) error {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	if rc := C.hnsw_save(i.ptr, cpath); rc != C.HNSW_OK {
		return lastError()
	}
	return nil
}

func Load(path string) (*Index, error) {
	cpath := C.CString(path)
	defer C.free(unsafe.Pointer(cpath))
	ptr := C.hnsw_load(cpath)
	if ptr == nil {
		return nil, lastError()
	}
	idx := &Index{ptr: ptr, dim: int(C.hnsw_dim(ptr))}
	runtime.SetFinalizer(idx, (*Index).Close)
	return idx, nil
}

func (i *Index) Len() int         { return int(C.hnsw_size(i.ptr)) }
func (i *Index) ActiveLen() int   { return int(C.hnsw_active_size(i.ptr)) }
func (i *Index) Dim() int         { return i.dim }
func (i *Index) MemoryBytes() int { return int(C.hnsw_memory_bytes(i.ptr)) }
