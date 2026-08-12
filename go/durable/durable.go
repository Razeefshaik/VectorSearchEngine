// Package durable combines the cgo-wrapped HNSW index with a write-ahead log
// so that writes survive a process crash. This is the piece that sits
// between the gRPC shard server and the raw hnsw.Index: everything the shard
// server calls should go through here, not through hnsw.Index directly,
// or writes silently lose their durability guarantee.
package durable

import (
	"errors"
	"fmt"
	"os"

	"hnswdb/hnsw"
	"hnswdb/wal"
)

// Config describes how to open or create a durable index. SnapshotPath and
// WALPath live next to each other on disk; both are required.
type Config struct {
	SnapshotPath   string
	WALPath        string
	Space          hnsw.Space
	Dim            int
	MaxElements    int
	M              int
	EfConstruction int
	Seed           uint64
}

type Index struct {
	idx *hnsw.Index
	w   *wal.WAL
	cfg Config
}

// Open recovers state from cfg.SnapshotPath (if it exists) plus
// cfg.WALPath (if it exists), replays any WAL records not yet captured by
// the snapshot, then reopens the WAL for new appends -- dropping any torn
// tail record left by a crash mid-write. If neither file exists, this
// creates a fresh empty index and a fresh empty WAL.
//
// Call this once at process startup. Do not call Replay yourself elsewhere;
// mixing manual replay with a live Index will re-apply records the running
// index has already seen.
func Open(cfg Config) (*Index, error) {
	idx, err := loadOrCreateIndex(cfg)
	if err != nil {
		return nil, err
	}

	applied, lastGood, err := wal.Replay(cfg.WALPath, func(r wal.Record) error {
		return apply(idx, r)
	})
	if err != nil {
		return nil, fmt.Errorf("durable: replaying WAL: %w", err)
	}
	_ = applied // number of records recovered; log this at the call site if useful

	w, err := reopenWAL(cfg.WALPath, lastGood)
	if err != nil {
		return nil, err
	}

	return &Index{idx: idx, w: w, cfg: cfg}, nil
}

func loadOrCreateIndex(cfg Config) (*hnsw.Index, error) {
	if _, err := os.Stat(cfg.SnapshotPath); err == nil {
		idx, err := hnsw.Load(cfg.SnapshotPath)
		if err != nil {
			return nil, fmt.Errorf("durable: loading snapshot %s: %w", cfg.SnapshotPath, err)
		}
		return idx, nil
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("durable: checking snapshot %s: %w", cfg.SnapshotPath, err)
	}

	idx, err := hnsw.New(cfg.Space, cfg.Dim, cfg.MaxElements, cfg.M, cfg.EfConstruction, cfg.Seed)
	if err != nil {
		return nil, fmt.Errorf("durable: creating fresh index: %w", err)
	}
	return idx, nil
}

func reopenWAL(path string, lastGoodOffset int64) (*wal.WAL, error) {
	if _, err := os.Stat(path); err == nil {
		w, err := wal.OpenForAppend(path, lastGoodOffset) // truncates any torn tail
		if err != nil {
			return nil, fmt.Errorf("durable: reopening WAL %s: %w", path, err)
		}
		return w, nil
	} else if !os.IsNotExist(err) {
		return nil, fmt.Errorf("durable: checking WAL %s: %w", path, err)
	}

	w, err := wal.Create(path)
	if err != nil {
		return nil, fmt.Errorf("durable: creating WAL %s: %w", path, err)
	}
	return w, nil
}

// apply applies one WAL record to idx during replay, treating "this was
// already applied" errors as success rather than failure. That's what makes
// replay idempotent: a record can safely be re-applied if a crash happened
// between fsyncing it and a snapshot that made it redundant -- see
// (*Index).Snapshot for exactly when that window exists.
func apply(idx *hnsw.Index, r wal.Record) error {
	switch r.Op {
	case wal.OpAdd:
		err := idx.Add(r.Vector, r.Label)
		if errors.Is(err, hnsw.ErrDuplicateLabel) {
			return nil // already present -- captured by a snapshot after this record
		}
		return err
	case wal.OpMarkDeleted:
		err := idx.MarkDeleted(r.Label)
		if errors.Is(err, hnsw.ErrNotFound) {
			return nil // already deleted, or the add that would precede it
			// was itself already captured -- either way, nothing to do
		}
		return err
	case wal.OpUnmarkDeleted:
		err := idx.UnmarkDeleted(r.Label)
		if errors.Is(err, hnsw.ErrNotFound) {
			return nil
		}
		return err
	default:
		return fmt.Errorf("durable: unknown WAL op %d", r.Op)
	}
}

// Add durably inserts a vector. The WAL record is written and fsynced BEFORE
// the vector is applied to the in-memory index, and only then does this
// return. A crash at any point after Add returns nil will recover this
// vector on the next Open.
func (d *Index) Add(vec []float32, label uint64) error {
	if err := d.w.Append(wal.Record{Op: wal.OpAdd, Label: label, Vector: vec}); err != nil {
		return fmt.Errorf("durable: wal append: %w", err)
	}
	if err := d.idx.Add(vec, label); err != nil {
		// Logged durably but the in-memory apply failed -- e.g. a duplicate
		// label from a bad caller. The record sits harmlessly in the WAL; a
		// future replay's idempotency handling will just no-op it.
		return fmt.Errorf("durable: index add (already logged to WAL): %w", err)
	}
	return nil
}

func (d *Index) MarkDeleted(label uint64) error {
	if err := d.w.Append(wal.Record{Op: wal.OpMarkDeleted, Label: label}); err != nil {
		return fmt.Errorf("durable: wal append: %w", err)
	}
	return d.idx.MarkDeleted(label)
}

func (d *Index) UnmarkDeleted(label uint64) error {
	if err := d.w.Append(wal.Record{Op: wal.OpUnmarkDeleted, Label: label}); err != nil {
		return fmt.Errorf("durable: wal append: %w", err)
	}
	return d.idx.UnmarkDeleted(label)
}

// Search does not touch the WAL -- reads have nothing to make durable.
func (d *Index) Search(query []float32, k, ef int) ([]hnsw.Result, error) {
	return d.idx.Search(query, k, ef)
}

// Snapshot durably captures the current index state and compacts the WAL:
// once this returns, every record the WAL held has been folded into the
// snapshot file, so the WAL is rotated to empty.
//
// Order matters. We (1) save the snapshot to a temp file, (2) atomically
// rename it into place, (3) create a new empty WAL, (4) close the old WAL,
// (5) atomically rename the new WAL into place. If the process crashes
// between steps 2 and 5, the OLD WAL (with records already reflected in the
// new snapshot) is still what gets replayed on restart -- and that's fine:
// apply()'s idempotency handling means re-applying already-captured records
// is a harmless no-op, not a correctness bug.
func (d *Index) Snapshot() error {
	tmpSnapshot := d.cfg.SnapshotPath + ".tmp"
	if err := d.idx.Save(tmpSnapshot); err != nil {
		return fmt.Errorf("durable: saving snapshot: %w", err)
	}
	if err := os.Rename(tmpSnapshot, d.cfg.SnapshotPath); err != nil {
		return fmt.Errorf("durable: renaming snapshot into place: %w", err)
	}

	tmpWALPath := d.cfg.WALPath + ".new"
	newWAL, err := wal.Create(tmpWALPath)
	if err != nil {
		return fmt.Errorf("durable: creating new WAL: %w", err)
	}
	if err := d.w.Close(); err != nil {
		newWAL.Close()
		os.Remove(tmpWALPath)
		return fmt.Errorf("durable: closing old WAL: %w", err)
	}
	if err := os.Rename(tmpWALPath, d.cfg.WALPath); err != nil {
		return fmt.Errorf("durable: rotating WAL into place: %w", err)
	}
	d.w = newWAL
	return nil
}

func (d *Index) Close() error {
	return d.w.Close()
}

func (d *Index) Len() int       { return d.idx.Len() }
func (d *Index) ActiveLen() int { return d.idx.ActiveLen() }
