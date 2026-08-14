







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
		if err := idx.Add(testVec(float32(i)), uint64(i)); err != nil {
			t.Fatalf("Add(%d): %v", i, err)
		}
	}
	if idx.Len() != n {
		t.Fatalf("Len() = %d, want %d", idx.Len(), n)
	}

	
	
	
	idx = nil

	recovered, err := Open(cfg)
	if err != nil {
		t.Fatalf("Open (recovery): %v", err)
	}
	defer recovered.Close()

	if recovered.Len() != n {
		t.Fatalf("after recovery, Len() = %d, want %d", recovered.Len(), n)
	}

	res, err := recovered.Search(testVec(10), 1, 50)
	if err != nil {
		t.Fatalf("Search after recovery: %v", err)
	}
	if len(res) != 1 || res[0].Label != 10 {
		t.Fatalf("Search after recovery = %+v, want nearest label 10", res)
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
		if err := idx.Add(testVec(float32(i)), uint64(i)); err != nil {
			t.Fatalf("Add(%d): %v", i, err)
		}
	}
	if err := idx.Snapshot(); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	for i := 20; i < 30; i++ { 
		if err := idx.Add(testVec(float32(i)), uint64(i)); err != nil {
			t.Fatalf("Add(%d): %v", i, err)
		}
	}
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
		if err := idx.Add(testVec(float32(i)), uint64(i)); err != nil {
			t.Fatalf("Add(%d): %v", i, err)
		}
	}
	if err := idx.MarkDeleted(2); err != nil {
		t.Fatalf("MarkDeleted: %v", err)
	}
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
		if err := idx.Add(testVec(float32(i)), uint64(i)); err != nil {
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
