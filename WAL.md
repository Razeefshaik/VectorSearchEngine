# Write-ahead log

This covers `go/wal` and `go/durable` — what guarantee they give you, how the
format works, and the two real bugs that showed up while wiring this
together (worth knowing, since both are the kind of thing that's obvious in
hindsight and easy to miss the first time).

## The guarantee

> If `durable.Index.Add` (or `MarkDeleted` / `UnmarkDeleted`) returns `nil`,
> that write survives a crash of the process at any point after that —
> power loss, panic, `kill -9`, doesn't matter.

That's the entire point. Everything below is in service of that one sentence.

## How

Every mutation is appended to a log file and `fsync`'d **before** it's applied
to the in-memory index:

```
client calls Add(vec, label)
        │
        ▼
  wal.Append(record)   ── fsync ──►  durable on disk
        │
        ▼
  idx.Add(vec, label)  ── applies to the in-memory HNSW graph
        │
        ▼
  return nil to the client
```

If the process dies between the fsync and the in-memory apply, that's fine —
the in-memory graph is gone anyway (it was never anything but RAM), and the
WAL record is still there. On restart, `durable.Open` replays it.

If the process dies *before* the fsync completes, the client never got `nil`
back, so as far as they know the write might not have happened — which is
exactly true. No lie was told.

## Format

```
file   := header record*
header := magic(4) version(4)
record := length(4) payload(length) crc32(4)
payload:= op(1) label(8) dim(4) vector(dim*4)
```

`length` covers `payload` only. `dim` is 0 for delete ops (no vector).

### Why a length prefix AND a checksum

Either alone isn't enough:

- **Length only**: a crash mid-write can leave a record where the length
  field itself is intact but the payload bytes after it are truncated or
  garbage. You'd read `length`, try to read that many bytes, and either get a
  short read (detectable) or — worse — if the crash happened to leave stale
  bytes from a previous, longer write sitting after a shorter new one, you'd
  read a complete but *wrong* record and not know it.
- **Checksum only**: without a length prefix you don't know how many bytes to
  checksum without a delimiter, so you'd need some other framing anyway.

Together: read `length`, read that many bytes, read 4 more for the checksum,
verify. Any failure at any of those three steps — short read or checksum
mismatch — means "this record didn't fully make it to disk," full stop,
regardless of which byte the crash happened to land on.

### Why "torn tail" and "corruption" get the same treatment

A WAL has exactly one writer, always appending to the end. The *only* way a
record can be incomplete is a crash while it was being written — there's no
scenario where a fully-written record in the middle of the file spontaneously
loses bytes. So `Replay` treats every decode failure as "stop here, this was
the record being written when we crashed" and returns cleanly with however
many complete records it found. It does not try to skip past a bad record and
keep scanning — bad framing has no reliable resync point, and guessing wrong
would silently drop or duplicate data.

I checked this behavior before writing the Go — `/tmp/wal_format_check.py` (a
Python stand-in for the same algorithm) writes a real file, truncates it at
several different byte offsets mid-last-record, and confirms replay recovers
every complete record before the cut and stops cleanly, no exceptions, no
misread data. Same logic, implemented in `wal.go`'s `readRecord`.

## Idempotent replay (and the bug it caught)

Replay can legitimately see a record for something that's *already* in the
index — this happens whenever a crash lands between a snapshot capturing a
record's effect and the WAL being rotated to drop that now-redundant record
(see `durable.Index.Snapshot`'s doc comment for exactly when that window is
open). So replaying must tolerate "already applied" without treating it as an
error:

```go
func apply(idx *hnsw.Index, r wal.Record) error {
    switch r.Op {
    case wal.OpAdd:
        err := idx.Add(r.Vector, r.Label)
        if errors.Is(err, hnsw.ErrDuplicateLabel) {
            return nil // already present -- fine
        }
        return err
    ...
```

Wiring this up surfaced a real bug in the C API from the earlier session:
`hnsw_mark_deleted` returned a distinct `HNSW_ERR_NOT_FOUND` code, but
`hnsw_add`'s failure path collapsed everything — duplicate label, index full,
anything else — into one generic error code, distinguishable only by
substring-matching the error *message*. Matching on error text to decide
whether a failure is safe to ignore is fragile (the message is a C++
exception's `.what()`, not a stable API). Fixed by giving duplicate-label its
own code (`HNSW_ERR_DUPLICATE`) on the C side and a sentinel `error` value on
the Go side (`hnsw.ErrDuplicateLabel`), checked with `errors.Is` — see the
diff in `include/hnsw_c.h`, `src/hnsw_c.cpp`, and `go/hnsw/hnsw.go`. This is
the kind of thing that's easy to get away with until exactly this replay path
needs to make a real decision based on *which* error came back.

## Snapshot rotation

```
1. idx.Save(path + ".tmp")
2. rename(path + ".tmp", path)              -- atomic on the same filesystem
3. wal.Create(walPath + ".new")
4. close old WAL
5. rename(walPath + ".new", walPath)         -- atomic
```

Two renames, not one big atomic operation, because there isn't a way to
atomically swap two files as a single filesystem operation. The question is
always "what happens if we crash between steps," and the answer here is: any
crash point leaves you with a snapshot and a WAL that are each individually
valid, and possibly overlapping (both describing the same record) — which
idempotent replay handles for free. There's no crash point that loses data or
requires special-case recovery logic. `durable_test.go`'s
`TestReopenTwiceIsIdempotent` exercises exactly this by snapshotting, closing,
and reopening.

## fsync-per-write vs. group commit

Right now every `Append` calls `fsync` before returning. That's the simple,
obviously-correct default — every acknowledged write is durable, no
exceptions. The cost is one disk flush per write, which is sub-millisecond on
SSD but adds up under high write concurrency.

The standard next step, if you measure this as your bottleneck, is **group
commit**: batch several pending `Append` calls behind one `fsync`. Several
goroutines' writes queue up, one of them (or a background flusher) does a
single `fsync` covering all of them, and all the waiters get released
together. This trades a little latency (a write waits for its batch to fill
or a timer to fire) for much higher throughput, without weakening durability
at all — nothing is acknowledged until the fsync it depends on has actually
happened. Worth building once you've profiled and confirmed fsync is the
ceiling, not before.

## Testing

```bash
cd go
go test ./wal/...          # pure stdlib, no build dependencies
```

`go/wal` has no dependency on the C++ side at all, so its tests run
anywhere. The interesting one is `TestReplayTornTail`, which manually writes
a truncated record (bypassing `Append`, which only ever produces complete
ones) to simulate exactly the crash scenario the whole format exists to
handle.

```bash
cmake -B build && cmake --build build -j     # from repo root -- builds libhnsw.so
cd go
go test ./durable/...      # needs the .so; hnsw.go's cgo directives + rpath
                            # should find it at ../../build automatically
```

`TestCrashRecovery` is the load-bearing one: add 50 vectors, drop the handle
with no clean shutdown at all, reopen, confirm all 50 are there and
searchable.
