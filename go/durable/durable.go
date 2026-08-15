package durable

import (
	"errors"
	"fmt"
	"os"
	"time"

	"hnswdb/hnsw"
	"hnswdb/wal"
)

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
	_ = applied

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
		w, err := wal.OpenForAppend(path, lastGoodOffset)
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

func apply(idx *hnsw.Index, r wal.Record) error {
	key := hnsw.Key{ClientID: r.ClientID, Label: r.Label}
	switch r.Op {
	case wal.OpAdd:
		err := idx.Add(r.Vector, key)
		if errors.Is(err, hnsw.ErrDuplicateLabel) {
			return nil
		}
		return err
	case wal.OpMarkDeleted:
		err := idx.MarkDeleted(key)
		if errors.Is(err, hnsw.ErrNotFound) {
			return nil
		}
		return err
	case wal.OpUnmarkDeleted:
		err := idx.UnmarkDeleted(key)
		if errors.Is(err, hnsw.ErrNotFound) {
			return nil
		}
		return err
	default:
		return fmt.Errorf("durable: unknown WAL op %d", r.Op)
	}
}

func (d *Index) Add(vec []float32, key hnsw.Key) error {
	if err := d.w.Append(wal.Record{Op: wal.OpAdd, ClientID: key.ClientID, Label: key.Label, Vector: vec}); err != nil {
		return fmt.Errorf("durable: wal append: %w", err)
	}
	if err := d.idx.Add(vec, key); err != nil {
		return fmt.Errorf("durable: index add (already logged to WAL): %w", err)
	}
	return nil
}

func (d *Index) MarkDeleted(key hnsw.Key) error {
	if err := d.w.Append(wal.Record{Op: wal.OpMarkDeleted, ClientID: key.ClientID, Label: key.Label}); err != nil {
		return fmt.Errorf("durable: wal append: %w", err)
	}
	return d.idx.MarkDeleted(key)
}

func (d *Index) UnmarkDeleted(key hnsw.Key) error {
	if err := d.w.Append(wal.Record{Op: wal.OpUnmarkDeleted, ClientID: key.ClientID, Label: key.Label}); err != nil {
		return fmt.Errorf("durable: wal append: %w", err)
	}
	return d.idx.UnmarkDeleted(key)
}

func (d *Index) Search(query []float32, k, ef int) ([]hnsw.Result, error) {
	return d.idx.Search(query, k, ef)
}

func (d *Index) Snapshot() error {
	tmpSnapshot := d.cfg.SnapshotPath + ".tmp"
	if err := d.idx.Save(tmpSnapshot); err != nil {
		return fmt.Errorf("durable: saving snapshot: %w", err)
	}
	if err := renameWithRetry(tmpSnapshot, d.cfg.SnapshotPath); err != nil {
		return fmt.Errorf("durable: renaming snapshot into place: %w", err)
	}

	tmpWALPath := d.cfg.WALPath + ".new"
	newWAL, err := wal.Create(tmpWALPath)
	if err != nil {
		return fmt.Errorf("durable: creating new WAL: %w", err)
	}
	// Unlike POSIX rename, Windows refuses to rename a file as either the
	// source or the destination while any handle to it is still open. Close
	// both the old WAL and the freshly created one before renaming, then
	// reopen the rotated file for append.
	if err := d.w.Close(); err != nil {
		newWAL.Close()
		os.Remove(tmpWALPath)
		return fmt.Errorf("durable: closing old WAL: %w", err)
	}
	if err := newWAL.Close(); err != nil {
		os.Remove(tmpWALPath)
		return fmt.Errorf("durable: closing new WAL: %w", err)
	}
	if err := renameWithRetry(tmpWALPath, d.cfg.WALPath); err != nil {
		return fmt.Errorf("durable: rotating WAL into place: %w", err)
	}
	w, err := wal.OpenForAppend(d.cfg.WALPath, -1)
	if err != nil {
		return fmt.Errorf("durable: reopening rotated WAL: %w", err)
	}
	d.w = w
	return nil
}

// renameWithRetry retries os.Rename briefly on failure. Even after a handle
// is closed, Windows can report ERROR_SHARING_VIOLATION for a short window
// afterward (commonly caused by antivirus/indexer scanning touching the file
// the instant it closes) where POSIX rename would have succeeded
// unconditionally. The retries are a no-op on platforms/situations where the
// first attempt already succeeds.
func renameWithRetry(oldpath, newpath string) error {
	var err error
	for i := 0; i < 10; i++ {
		if err = os.Rename(oldpath, newpath); err == nil {
			return nil
		}
		time.Sleep(20 * time.Millisecond)
	}
	return err
}

func (d *Index) Close() error {
	return d.w.Close()
}

func (d *Index) Len() int       { return d.idx.Len() }
func (d *Index) ActiveLen() int { return d.idx.ActiveLen() }
