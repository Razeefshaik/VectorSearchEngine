package durable

import (
	"path/filepath"
	"testing"

	"hnswdb/hnsw"
)

func testConfig(dir string) Config {
	return Config{
		SnapshotPath:   filepath.Join(dir, "index.snapshot"),
		WALPath:        filepath.Join(dir, "index.wal"),
		Space:          hnsw.Cosine,
		Dim:            8,
		MaxElements:    1000,
		M:              16,
		EfConstruction: 100,
		Seed:           42,
	}
}

func testVec(seed float32) []float32 {
	v := make([]float32, 8)
	for i := range v {
		v[i] = seed + float32(i)*0.01
	}
	return v
}

func TestCrashRecovery(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(dir)

	idx, err := Open(cfg)
	if err != nil {
		t.Fatalf("Open (fresh): %v", err)
	}

	const n = 50
	for i := 0; i < n; i++ {
		if err := idx.Add(testVec(float32(i)), hnsw.Key{ClientID: 1, Label: uint64(i)}); err != nil {
			t.Fatalf("Add(%d): %v", i, err)
		}
	}
	if idx.Len() != n {
		t.Fatalf("Len() = %d, want %d", idx.Len(), n)
	}

	// Simulate an unclean crash: drop the reference without calling
	// idx.Close(), so Open() below must recover purely from what's on disk.
	// The old WAL file handle is still technically open at the OS level
	// until GC finalizes it; that's fine for the recovery test itself (POSIX
	// and Windows both allow a second handle to read/append a file that's
	// still open elsewhere), but Windows won't let TempDir's cleanup delete
	// the directory while any handle to a file in it is open. Keep a
	// reference so it can be force-closed after the recovery assertions,
	// once it's no longer part of what's being tested.
	crashed := idx
	defer crashed.w.Close()
	idx = nil

	recovered, err := Open(cfg)
	if err != nil {
		t.Fatalf("Open (recovery): %v", err)
	}
	defer recovered.Close()

	if recovered.Len() != n {
		t.Fatalf("after recovery, Len() = %d, want %d", recovered.Len(), n)
	}

	// testVec(10) is exactly the vector stored under label 10, so it's an
	// exact (distance 0) match -- but the test vectors are closely spaced
	// (testVec(i) and testVec(i+1) differ by a tiny additive offset per
	// dimension), and HNSW is an approximate index, so top-1 alone isn't
	// guaranteed to land on the exact match. Search a small top-k instead
	// and require label 10 to appear in it, which is what recovery actually
	// promises: the vector survived, findable, not necessarily ranked #1
	// among near-duplicates.
	res, err := recovered.Search(testVec(10), 3, 50)
	if err != nil {
		t.Fatalf("Search after recovery: %v", err)
	}
	found := false
	for _, r := range res {
		if r.Key.Label == 10 {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("Search after recovery = %+v, want label 10 among top results", res)
	}
}

func TestSnapshotThenCrashRecovers(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(dir)

	idx, err := Open(cfg)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	for i := 0; i < 20; i++ {
		if err := idx.Add(testVec(float32(i)), hnsw.Key{ClientID: 1, Label: uint64(i)}); err != nil {
			t.Fatalf("Add(%d): %v", i, err)
		}
	}
	if err := idx.Snapshot(); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	for i := 20; i < 30; i++ {
		if err := idx.Add(testVec(float32(i)), hnsw.Key{ClientID: 1, Label: uint64(i)}); err != nil {
			t.Fatalf("Add(%d): %v", i, err)
		}
	}
	crashed := idx
	defer crashed.w.Close() // release the handle so TempDir cleanup can succeed on Windows
	idx = nil

	recovered, err := Open(cfg)
	if err != nil {
		t.Fatalf("Open (recovery): %v", err)
	}
	defer recovered.Close()

	if recovered.Len() != 30 {
		t.Fatalf("after snapshot+WAL recovery, Len() = %d, want 30", recovered.Len())
	}
}

func TestDeleteSurvivesRecovery(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(dir)

	idx, err := Open(cfg)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	for i := 0; i < 5; i++ {
		if err := idx.Add(testVec(float32(i)), hnsw.Key{ClientID: 1, Label: uint64(i)}); err != nil {
			t.Fatalf("Add(%d): %v", i, err)
		}
	}
	if err := idx.MarkDeleted(hnsw.Key{ClientID: 1, Label: 2}); err != nil {
		t.Fatalf("MarkDeleted: %v", err)
	}
	crashed := idx
	defer crashed.w.Close() // release the handle so TempDir cleanup can succeed on Windows
	idx = nil

	recovered, err := Open(cfg)
	if err != nil {
		t.Fatalf("Open (recovery): %v", err)
	}
	defer recovered.Close()

	if recovered.ActiveLen() != 4 {
		t.Fatalf("ActiveLen() after recovery = %d, want 4", recovered.ActiveLen())
	}
	if recovered.Len() != 5 {
		t.Fatalf("Len() after recovery = %d, want 5 (soft delete keeps the node)", recovered.Len())
	}
}

func TestReopenTwiceIsIdempotent(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(dir)

	idx, err := Open(cfg)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	for i := 0; i < 10; i++ {
		if err := idx.Add(testVec(float32(i)), hnsw.Key{ClientID: 1, Label: uint64(i)}); err != nil {
			t.Fatalf("Add(%d): %v", i, err)
		}
	}
	if err := idx.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	r1, err := Open(cfg)
	if err != nil {
		t.Fatalf("first reopen: %v", err)
	}
	if r1.Len() != 10 {
		t.Fatalf("first reopen Len() = %d, want 10", r1.Len())
	}
	if err := r1.Snapshot(); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	if err := r1.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	r2, err := Open(cfg)
	if err != nil {
		t.Fatalf("second reopen: %v", err)
	}
	defer r2.Close()
	if r2.Len() != 10 {
		t.Fatalf("second reopen Len() = %d, want 10 (snapshot + near-empty WAL replay)", r2.Len())
	}
}
