package durable

import (
	"errors"
	"fmt"
	"os"

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
	switch r.Op {
	case wal.OpAdd:
		err := idx.Add(r.Vector, r.Label)
		if errors.Is(err, hnsw.ErrDuplicateLabel) {
			return nil
		}
		return err
	case wal.OpMarkDeleted:
		err := idx.MarkDeleted(r.Label)
		if errors.Is(err, hnsw.ErrNotFound) {
			return nil
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

func (d *Index) Add(vec []float32, label uint64) error {
	if err := d.w.Append(wal.Record{Op: wal.OpAdd, Label: label, Vector: vec}); err != nil {
		return fmt.Errorf("durable: wal append: %w", err)
	}
	if err := d.idx.Add(vec, label); err != nil {
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

func (d *Index) Search(query []float32, k, ef int) ([]hnsw.Result, error) {
	return d.idx.Search(query, k, ef)
}

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
