package wal

import (
	"errors"
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestCreateAppendReplay(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.wal")

	w, err := Create(path)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	records := []Record{
		{Op: OpAdd, Label: 1, Vector: []float32{1, 2, 3}},
		{Op: OpAdd, Label: 2, Vector: []float32{4, 5, 6}},
		{Op: OpMarkDeleted, Label: 1},
		{Op: OpAdd, Label: 3, Vector: []float32{7, 8, 9}},
	}
	for _, r := range records {
		if err := w.Append(r); err != nil {
			t.Fatalf("Append(%+v): %v", r, err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	var got []Record
	n, lastGood, err := Replay(path, func(r Record) error {
		got = append(got, r)
		return nil
	})
	if err != nil {
		t.Fatalf("Replay: %v", err)
	}
	if n != len(records) {
		t.Fatalf("recordsApplied = %d, want %d", n, len(records))
	}
	fi, statErr := os.Stat(path)
	if statErr != nil {
		t.Fatalf("Stat: %v", statErr)
	}
	if lastGood != fi.Size() {
		t.Fatalf("lastGoodOffset = %d, want file size %d", lastGood, fi.Size())
	}
	for i := range records {
		if !reflect.DeepEqual(got[i], records[i]) {
			t.Fatalf("record %d = %+v, want %+v", i, got[i], records[i])
		}
	}
}

// TestReplayTornTail is the important one: it simulates exactly the failure
// mode a real crash produces -- a partially written last record -- and
// checks that Replay recovers everything before it and cleanly discards the
// rest, rather than erroring out or (worse) misinterpreting garbage.
func TestReplayTornTail(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.wal")

	w, err := Create(path)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	good := []Record{
		{Op: OpAdd, Label: 1, Vector: []float32{1, 2, 3}},
		{Op: OpAdd, Label: 2, Vector: []float32{4, 5, 6}},
	}
	for _, r := range good {
		if err := w.Append(r); err != nil {
			t.Fatalf("Append(%+v): %v", r, err)
		}
	}

	// Simulate a crash mid-write of a third record: encode it but write only
	// part of it, bypassing Append (which only ever produces complete,
	// fsynced records).
	tornBuf, err := encode(Record{Op: OpAdd, Label: 3, Vector: []float32{7, 8, 9}})
	if err != nil {
		t.Fatalf("encode: %v", err)
	}
	if _, err := w.f.Write(tornBuf[:len(tornBuf)-5]); err != nil {
		t.Fatalf("writing torn record: %v", err)
	}
	if err := w.f.Sync(); err != nil {
		t.Fatalf("sync: %v", err)
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	var got []Record
	n, lastGood, err := Replay(path, func(r Record) error {
		got = append(got, r)
		return nil
	})
	if err != nil {
		t.Fatalf("Replay should tolerate a torn tail without erroring, got: %v", err)
	}
	if n != len(good) {
		t.Fatalf("recordsApplied = %d, want %d (torn record must be dropped)", n, len(good))
	}
	for i := range good {
		if !reflect.DeepEqual(got[i], good[i]) {
			t.Fatalf("record %d = %+v, want %+v", i, got[i], good[i])
		}
	}

	// The real recovery path: truncate the torn tail via OpenForAppend, then
	// confirm new writes land cleanly right after the last GOOD record.
	w2, err := OpenForAppend(path, lastGood)
	if err != nil {
		t.Fatalf("OpenForAppend after torn tail: %v", err)
	}
	if err := w2.Append(Record{Op: OpAdd, Label: 99, Vector: []float32{0, 0, 0}}); err != nil {
		t.Fatalf("Append after recovery: %v", err)
	}
	if err := w2.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	got = nil
	n, _, err = Replay(path, func(r Record) error {
		got = append(got, r)
		return nil
	})
	if err != nil {
		t.Fatalf("final replay: %v", err)
	}
	if n != 3 {
		t.Fatalf("recordsApplied after recovery = %d, want 3 (2 original + 1 new)", n)
	}
	if got[2].Label != 99 {
		t.Fatalf("third record label = %d, want 99", got[2].Label)
	}
}

func TestReplayMissingFileIsNotAnError(t *testing.T) {
	n, offset, err := Replay(filepath.Join(t.TempDir(), "nope.wal"), func(Record) error {
		t.Fatal("apply must not be called when the WAL file doesn't exist")
		return nil
	})
	if err != nil {
		t.Fatalf("Replay of a missing file should return nil error, got: %v", err)
	}
	if n != 0 || offset != 0 {
		t.Fatalf("Replay of a missing file: got (%d, %d), want (0, 0)", n, offset)
	}
}

func TestApplyErrorStopsReplayAndPropagates(t *testing.T) {
	path := filepath.Join(t.TempDir(), "test.wal")
	w, err := Create(path)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	for _, r := range []Record{
		{Op: OpAdd, Label: 1, Vector: []float32{1}},
		{Op: OpAdd, Label: 2, Vector: []float32{2}},
	} {
		if err := w.Append(r); err != nil {
			t.Fatalf("Append: %v", err)
		}
	}
	if err := w.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	applied := 0
	_, _, err = Replay(path, func(r Record) error {
		applied++
		if r.Label == 2 {
			return errors.New("simulated fatal error")
		}
		return nil
	})
	if err == nil {
		t.Fatal("expected Replay to propagate the error apply returned")
	}
	if applied != 2 {
		t.Fatalf("apply was called %d times, want 2 (it should have been tried on the failing record)", applied)
	}
}

func TestEncodeRejectsMismatchedOpAndVector(t *testing.T) {
	if _, err := encode(Record{Op: OpAdd}); err == nil {
		t.Fatal("OpAdd with no vector should be rejected")
	}
	if _, err := encode(Record{Op: OpMarkDeleted, Vector: []float32{1}}); err == nil {
		t.Fatal("OpMarkDeleted with a vector should be rejected")
	}
}
