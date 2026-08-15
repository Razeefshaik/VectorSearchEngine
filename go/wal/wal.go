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
	magic   uint32 = 0x57414C31
	version uint32 = 2 // v2: payload carries clientId(8) alongside label(8), see Record

	headerSize = 8

	maxPayloadLen = 1 << 24
)

// ClientID and Label together identify the target vector -- two different
// clients may legitimately reuse the same Label value, so both fields are
// required to name a record's target unambiguously.
type Record struct {
	Op       OpType
	ClientID uint64
	Label    uint64
	Vector   []float32
}

type WAL struct {
	mu   sync.Mutex
	f    *os.File
	path string
}

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
	payloadLen := 1 + 8 + 8 + 4 + dim*4 // op + clientId + label + dim + vector
	buf := make([]byte, 4+payloadLen+4)

	binary.LittleEndian.PutUint32(buf[0:4], uint32(payloadLen))
	p := buf[4 : 4+payloadLen]
	p[0] = byte(r.Op)
	binary.LittleEndian.PutUint64(p[1:9], r.ClientID)
	binary.LittleEndian.PutUint64(p[9:17], r.Label)
	binary.LittleEndian.PutUint32(p[17:21], uint32(dim))
	for i, v := range r.Vector {
		binary.LittleEndian.PutUint32(p[21+i*4:25+i*4], math.Float32bits(v))
	}

	crc := crc32.ChecksumIEEE(p)
	binary.LittleEndian.PutUint32(buf[4+payloadLen:], crc)
	return buf, nil
}

func decode(p []byte) (Record, error) {
	if len(p) < 21 {
		return Record{}, errors.New("wal: payload shorter than fixed header")
	}
	op := OpType(p[0])
	clientID := binary.LittleEndian.Uint64(p[1:9])
	label := binary.LittleEndian.Uint64(p[9:17])
	dim := binary.LittleEndian.Uint32(p[17:21])
	if uint64(len(p)) != 21+uint64(dim)*4 {
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
			bits := binary.LittleEndian.Uint32(p[21+i*4 : 25+i*4])
			vec[i] = math.Float32frombits(bits)
		}
	}
	return Record{Op: op, ClientID: clientID, Label: label, Vector: vec}, nil
}

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
		return 0, 0, fmt.Errorf(
			"wal: %s: version %d, this code expects %d (v2 added a clientId "+
				"field to every record; old WAL files cannot be replayed and "+
				"must be discarded)", path, got, version)
	}

	offset := int64(headerSize)
	for {
		rec, n, ok := readRecord(br)
		if !ok {
			break
		}
		if err := apply(rec); err != nil {
			return recordsApplied, offset, fmt.Errorf(
				"wal: applying record %d (op=%d clientId=%d label=%d): %w",
				recordsApplied, rec.Op, rec.ClientID, rec.Label, err)
		}
		recordsApplied++
		offset += int64(n)
	}
	return recordsApplied, offset, nil
}

func readRecord(r *bufio.Reader) (rec Record, bytesRead int, ok bool) {
	var lenBuf [4]byte
	if _, err := io.ReadFull(r, lenBuf[:]); err != nil {
		return Record{}, 0, false
	}
	payloadLen := binary.LittleEndian.Uint32(lenBuf[:])
	if payloadLen == 0 || payloadLen > maxPayloadLen {
		return Record{}, 0, false
	}

	payload := make([]byte, payloadLen)
	if _, err := io.ReadFull(r, payload); err != nil {
		return Record{}, 0, false
	}

	var crcBuf [4]byte
	if _, err := io.ReadFull(r, crcBuf[:]); err != nil {
		return Record{}, 0, false
	}
	want := binary.LittleEndian.Uint32(crcBuf[:])
	if crc32.ChecksumIEEE(payload) != want {
		return Record{}, 0, false
	}

	rec, err := decode(payload)
	if err != nil {
		return Record{}, 0, false
	}
	return rec, 4 + int(payloadLen) + 4, true
}
