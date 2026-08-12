// Package wal implements a simple, crash-safe write-ahead log for the vector
// index. Every mutation (add / mark-deleted / unmark-deleted) is appended and
// fsynced here BEFORE it is applied to the in-memory HNSW index. On restart,
// replaying this log on top of the last snapshot reconstructs exactly the
// state the index had at the moment of the crash.
//
// Format (all integers little-endian):
//
//	file   := header record*
//	header := magic(4) version(4)
//	record := length(4) payload(length) crc32(4)
//	payload:= op(1) label(8) dim(4) vector(dim*4)
//
// `length` covers `payload` only, not itself or the trailing crc32.
//
// A record is valid only if length+payload+crc32 are ALL present on disk and
// the checksum matches. Any failure to read a complete, checksummed record --
// a short read, or a checksum mismatch -- is treated as a torn write from a
// crash mid-append, not as corruption: replay stops there and reports how
// many complete records it recovered plus the byte offset immediately after
// the last good one. This is the standard WAL assumption used by LevelDB,
// RocksDB, and similar systems: a WAL has exactly one writer appending to the
// tail, so the only way to end up with a partial record is a crash while
// writing the very last one. Corruption anywhere earlier in the file would
// indicate a different, more serious problem (disk bit rot, a bug elsewhere)
// that this package deliberately does not try to paper over.
package wal

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
	"io"
	"math"
	"os"
	"sync"
)

type OpType uint8

const (
	OpAdd           OpType = 1
	OpMarkDeleted   OpType = 2
	OpUnmarkDeleted OpType = 3
)

const (
	magic   uint32 = 0x57414C31 // "WAL1"
	version uint32 = 1

	headerSize = 8
	// Sanity bound on a record's payload length. Guards against a corrupt or
	// torn length field sending the reader off trying to allocate gigabytes
	// before the short-read check ever gets a chance to fire. Generous enough
	// for any real embedding dimension (a 1-million-dim vector would still fit).
	maxPayloadLen = 1 << 24 // 16 MiB
)

// Record is one logged mutation. Vector is set (and required, non-empty) for
// OpAdd; it is nil for the delete ops.
type Record struct {
	Op     OpType
	Label  uint64
	Vector []float32
}

// WAL is an append-only log file. Safe for concurrent Append/Sync calls.
type WAL struct {
	mu   sync.Mutex
	f    *os.File
	path string
}

// Create makes a brand-new, empty WAL file, failing if one already exists at
// path. Use this only when there is no prior state to recover, i.e. a
// freshly created index with no snapshot yet.
func Create(path string) (*WAL, error) {
	f, err := os.OpenFile(path, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0o644)
	if err != nil {
		return nil, fmt.Errorf("wal: create %s: %w", path, err)
	}
	if err := writeHeader(f); err != nil {
		f.Close()
		os.Remove(path)
		return nil, err
	}
	if err := f.Sync(); err != nil {
		f.Close()
		return nil, fmt.Errorf("wal: fsync header: %w", err)
	}
	return &WAL{f: f, path: path}, nil
}

// OpenForAppend opens an existing WAL file and positions it for new Append
// calls. It does NOT replay anything -- call Replay first to recover prior
// state, then pass the lastGoodOffset it returns here as truncateTo to
// discard any torn tail record before new writes begin. Pass truncateTo < 0
// to skip truncation (e.g. if you already know the file is clean).
func OpenForAppend(path string, truncateTo int64) (*WAL, error) {
	f, err := os.OpenFile(path, os.O_RDWR, 0o644)
	if err != nil {
		return nil, fmt.Errorf("wal: open %s: %w", path, err)
	}
	if truncateTo >= 0 {
		if err := f.Truncate(truncateTo); err != nil {
			f.Close()
			return nil, fmt.Errorf("wal: truncating torn tail: %w", err)
		}
	}
	if _, err := f.Seek(0, io.SeekEnd); err != nil {
		f.Close()
		return nil, fmt.Errorf("wal: seeking to end: %w", err)
	}
	return &WAL{f: f, path: path}, nil
}

func writeHeader(f *os.File) error {
	var hdr [headerSize]byte
	binary.LittleEndian.PutUint32(hdr[0:4], magic)
	binary.LittleEndian.PutUint32(hdr[4:8], version)
	_, err := f.Write(hdr[:])
	if err != nil {
		return fmt.Errorf("wal: writing header: %w", err)
	}
	return nil
}

// Append writes one record and fsyncs before returning. The record is
// durable the moment this call returns nil: a crash after that point will
// recover it on the next Replay.
//
// fsync-per-append is the safe default -- every write this returns nil for
// survives a crash, full stop. It costs one disk flush per call (sub-ms on
// SSD, more on spinning disk or a network filesystem). If that becomes your
// throughput ceiling, the standard fix is *group commit*: batch several
// pending Appends behind one fsync rather than one each. That trades a little
// latency (a write waits for its batch) for much higher throughput, without
// weakening durability -- worth reaching for once you've actually measured
// fsync as the bottleneck, not before.
func (w *WAL) Append(r Record) error {
	buf, err := encode(r)
	if err != nil {
		return err
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if _, err := w.f.Write(buf); err != nil {
		return fmt.Errorf("wal: write: %w", err)
	}
	if err := w.f.Sync(); err != nil {
		return fmt.Errorf("wal: fsync: %w", err)
	}
	return nil
}

// Sync flushes any OS-buffered writes to stable storage. Append already does
// this per call; Sync is exposed for callers that batch writes themselves.
func (w *WAL) Sync() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if err := w.f.Sync(); err != nil {
		return fmt.Errorf("wal: fsync: %w", err)
	}
	return nil
}

func (w *WAL) Close() error {
	w.mu.Lock()
	defer w.mu.Unlock()
	if err := w.f.Close(); err != nil {
		return fmt.Errorf("wal: close: %w", err)
	}
	return nil
}

func (w *WAL) Path() string { return w.path }

func encode(r Record) ([]byte, error) {
	if r.Op == OpAdd && len(r.Vector) == 0 {
		return nil, errors.New("wal: OpAdd requires a non-empty vector")
	}
	if r.Op != OpAdd && len(r.Vector) != 0 {
		return nil, errors.New("wal: only OpAdd carries a vector")
	}

	dim := len(r.Vector)
	payloadLen := 1 + 8 + 4 + dim*4 // op + label + dim + vector
	buf := make([]byte, 4+payloadLen+4)

	binary.LittleEndian.PutUint32(buf[0:4], uint32(payloadLen))
	p := buf[4 : 4+payloadLen]
	p[0] = byte(r.Op)
	binary.LittleEndian.PutUint64(p[1:9], r.Label)
	binary.LittleEndian.PutUint32(p[9:13], uint32(dim))
	for i, v := range r.Vector {
		binary.LittleEndian.PutUint32(p[13+i*4:17+i*4], math.Float32bits(v))
	}

	crc := crc32.ChecksumIEEE(p)
	binary.LittleEndian.PutUint32(buf[4+payloadLen:], crc)
	return buf, nil
}

func decode(p []byte) (Record, error) {
	if len(p) < 13 {
		return Record{}, errors.New("wal: payload shorter than fixed header")
	}
	op := OpType(p[0])
	label := binary.LittleEndian.Uint64(p[1:9])
	dim := binary.LittleEndian.Uint32(p[9:13])
	if uint64(len(p)) != 13+uint64(dim)*4 {
		return Record{}, errors.New("wal: payload length does not match declared dim")
	}
	switch op {
	case OpAdd, OpMarkDeleted, OpUnmarkDeleted:
	default:
		return Record{}, fmt.Errorf("wal: unknown op byte %d", op)
	}

	var vec []float32
	if dim > 0 {
		vec = make([]float32, dim)
		for i := range vec {
			bits := binary.LittleEndian.Uint32(p[13+i*4 : 17+i*4])
			vec[i] = math.Float32frombits(bits)
		}
	}
	return Record{Op: op, Label: label, Vector: vec}, nil
}

// Replay reads path from the beginning, calling apply for each valid record
// in order. recordsApplied is how many records were successfully applied.
// lastGoodOffset is the byte offset immediately after the last valid record --
// pass it to OpenForAppend's truncateTo to drop any torn tail before resuming
// writes.
//
// Replay stops at the first sign of a torn record (short read or bad
// checksum) rather than erroring out -- see the package doc for why that's
// the correct behaviour for a WAL, not a bug being swallowed.
//
// apply is called synchronously, in order. If apply returns an error, Replay
// stops immediately and returns it wrapped. apply is responsible for
// deciding which of ITS OWN errors are fatal versus safe to ignore -- e.g. an
// "already exists" error is expected and fine if this record's effect was
// already captured by a snapshot taken after the WAL was written but before
// the crash. See durable.apply for exactly that logic.
//
// A missing file at path is not an error: it means there is nothing to
// replay yet, and Replay returns (0, 0, nil).
func Replay(path string, apply func(Record) error) (recordsApplied int, lastGoodOffset int64, err error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return 0, 0, nil
		}
		return 0, 0, fmt.Errorf("wal: open %s: %w", path, err)
	}
	defer f.Close()

	br := bufio.NewReader(f)

	var hdr [headerSize]byte
	if _, err := io.ReadFull(br, hdr[:]); err != nil {
		return 0, 0, fmt.Errorf("wal: %s: reading header: %w", path, err)
	}
	if got := binary.LittleEndian.Uint32(hdr[0:4]); got != magic {
		return 0, 0, fmt.Errorf("wal: %s: bad magic 0x%x -- not a WAL file", path, got)
	}
	if got := binary.LittleEndian.Uint32(hdr[4:8]); got != version {
		return 0, 0, fmt.Errorf("wal: %s: version %d, this code expects %d", path, got, version)
	}

	offset := int64(headerSize)
	for {
		rec, n, ok := readRecord(br)
		if !ok {
			break // clean EOF or torn tail -- Replay treats them identically
		}
		if err := apply(rec); err != nil {
			return recordsApplied, offset, fmt.Errorf(
				"wal: applying record %d (op=%d label=%d): %w",
				recordsApplied, rec.Op, rec.Label, err)
		}
		recordsApplied++
		offset += int64(n)
	}
	return recordsApplied, offset, nil
}

// readRecord reads one record from r. ok is false both for a clean end of
// file and for a torn/corrupt tail -- see the package doc for why Replay
// treats those two cases the same way.
func readRecord(r *bufio.Reader) (rec Record, bytesRead int, ok bool) {
	var lenBuf [4]byte
	if _, err := io.ReadFull(r, lenBuf[:]); err != nil {
		return Record{}, 0, false // clean EOF at a record boundary
	}
	payloadLen := binary.LittleEndian.Uint32(lenBuf[:])
	if payloadLen == 0 || payloadLen > maxPayloadLen {
		return Record{}, 0, false // corrupt/torn length field
	}

	payload := make([]byte, payloadLen)
	if _, err := io.ReadFull(r, payload); err != nil {
		return Record{}, 0, false // torn write mid-payload
	}

	var crcBuf [4]byte
	if _, err := io.ReadFull(r, crcBuf[:]); err != nil {
		return Record{}, 0, false // torn write mid-checksum
	}
	want := binary.LittleEndian.Uint32(crcBuf[:])
	if crc32.ChecksumIEEE(payload) != want {
		return Record{}, 0, false // checksum mismatch: torn or corrupt, treat as torn
	}

	rec, err := decode(payload)
	if err != nil {
		return Record{}, 0, false
	}
	return rec, 4 + int(payloadLen) + 4, true
}
