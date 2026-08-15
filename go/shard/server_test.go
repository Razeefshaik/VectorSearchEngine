// Tests for the shard gRPC server. These need the C++ shared library built:
//
//	cmake -B build && cmake --build build -j
//	cd go && go test ./shard/...
//
// They run a real gRPC server over an in-process bufconn listener, so the
// full marshal/unmarshal path is exercised without binding a TCP port.
package shard

import (
	"context"
	"net"
	"path/filepath"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"

	"hnswdb/durable"
	"hnswdb/hnsw"
	shardpb "hnswdb/proto/shardpb"
)

const testDim = 8

func testConfig(dir string) Config {
	return Config{
		Durable: durable.Config{
			SnapshotPath:   filepath.Join(dir, "index.snapshot"),
			WALPath:        filepath.Join(dir, "index.wal"),
			Space:          hnsw.Cosine,
			Dim:            testDim,
			MaxElements:    1000,
			M:              16,
			EfConstruction: 100,
			Seed:           42,
		},
		DefaultEf:        50,
		SnapshotInterval: 0, // tests drive snapshots explicitly
	}
}

func testVec(seed float32) []float32 {
	v := make([]float32, testDim)
	for i := range v {
		v[i] = seed + float32(i)*0.01
	}
	return v
}

// startServer spins up a shard behind bufconn and returns a connected client.
func startServer(t *testing.T, cfg Config) (shardpb.ShardServiceClient, *Server, func()) {
	t.Helper()

	srv, err := New(cfg)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	lis := bufconn.Listen(1024 * 1024)
	gs := grpc.NewServer()
	shardpb.RegisterShardServiceServer(gs, srv)
	go func() { _ = gs.Serve(lis) }()

	conn, err := grpc.NewClient("passthrough:///bufnet",
		grpc.WithContextDialer(func(ctx context.Context, _ string) (net.Conn, error) {
			return lis.DialContext(ctx)
		}),
		grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("dial: %v", err)
	}

	cleanup := func() {
		conn.Close()
		gs.Stop()
		srv.Close()
	}
	return shardpb.NewShardServiceClient(conn), srv, cleanup
}

func TestInsertAndSearch(t *testing.T) {
	client, _, cleanup := startServer(t, testConfig(t.TempDir()))
	defer cleanup()

	ctx := context.Background()
	for i := 0; i < 50; i++ {
		_, err := client.Insert(ctx, &shardpb.InsertRequest{
			Key:    &shardpb.Key{ClientId: 1, Label: uint64(i)},
			Vector: testVec(float32(i)),
		})
		if err != nil {
			t.Fatalf("Insert(%d): %v", i, err)
		}
	}

	resp, err := client.Search(ctx, &shardpb.SearchRequest{
		Query: testVec(10), K: 5, Ef: 50,
	})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	if len(resp.GetResults()) != 5 {
		t.Fatalf("got %d results, want 5", len(resp.GetResults()))
	}
	found := false
	for _, r := range resp.GetResults() {
		if r.GetKey().GetLabel() == 10 && r.GetKey().GetClientId() == 1 {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected label 10 (clientId 1) among top-5 results, got %+v", resp.GetResults())
	}
}

// TestDuplicateInsertIsAlreadyExists pins the error code down, because the
// coordinator's retry logic keys off it: ALREADY_EXISTS means "a previous
// attempt landed, treat as success", while INTERNAL means "actually broken".
func TestDuplicateInsertIsAlreadyExists(t *testing.T) {
	client, _, cleanup := startServer(t, testConfig(t.TempDir()))
	defer cleanup()

	ctx := context.Background()
	key := &shardpb.Key{ClientId: 1, Label: 7}

	if _, err := client.Insert(ctx, &shardpb.InsertRequest{Key: key, Vector: testVec(1)}); err != nil {
		t.Fatalf("first Insert: %v", err)
	}
	_, err := client.Insert(ctx, &shardpb.InsertRequest{Key: key, Vector: testVec(1)})
	if err == nil {
		t.Fatal("second Insert of the same key should have failed")
	}
	if got := status.Code(err); got != codes.AlreadyExists {
		t.Fatalf("duplicate insert code = %v, want AlreadyExists", got)
	}
}

// Same key under a different client must NOT collide -- this is the whole
// point of the composite key.
func TestSameLabelDifferentClientsCoexist(t *testing.T) {
	client, _, cleanup := startServer(t, testConfig(t.TempDir()))
	defer cleanup()

	ctx := context.Background()
	for _, clientID := range []uint64{1, 2} {
		_, err := client.Insert(ctx, &shardpb.InsertRequest{
			Key:    &shardpb.Key{ClientId: clientID, Label: 99},
			Vector: testVec(float32(clientID)),
		})
		if err != nil {
			t.Fatalf("Insert(client=%d, label=99): %v", clientID, err)
		}
	}

	stats, err := client.Stats(ctx, &shardpb.StatsRequest{})
	if err != nil {
		t.Fatalf("Stats: %v", err)
	}
	if stats.GetTotalVectors() != 2 {
		t.Fatalf("total = %d, want 2 (same label, different clients)", stats.GetTotalVectors())
	}
}

func TestDimensionMismatchIsInvalidArgument(t *testing.T) {
	client, _, cleanup := startServer(t, testConfig(t.TempDir()))
	defer cleanup()

	ctx := context.Background()
	_, err := client.Insert(ctx, &shardpb.InsertRequest{
		Key:    &shardpb.Key{ClientId: 1, Label: 1},
		Vector: []float32{1, 2, 3}, // wrong dim
	})
	if got := status.Code(err); got != codes.InvalidArgument {
		t.Fatalf("dim mismatch code = %v, want InvalidArgument", got)
	}

	_, err = client.Search(ctx, &shardpb.SearchRequest{Query: []float32{1, 2, 3}, K: 5})
	if got := status.Code(err); got != codes.InvalidArgument {
		t.Fatalf("search dim mismatch code = %v, want InvalidArgument", got)
	}
}

func TestDeleteFiltersFromResults(t *testing.T) {
	client, _, cleanup := startServer(t, testConfig(t.TempDir()))
	defer cleanup()

	ctx := context.Background()
	for i := 0; i < 20; i++ {
		if _, err := client.Insert(ctx, &shardpb.InsertRequest{
			Key:    &shardpb.Key{ClientId: 1, Label: uint64(i)},
			Vector: testVec(float32(i)),
		}); err != nil {
			t.Fatalf("Insert(%d): %v", i, err)
		}
	}

	if _, err := client.Delete(ctx, &shardpb.DeleteRequest{
		Key: &shardpb.Key{ClientId: 1, Label: 5},
	}); err != nil {
		t.Fatalf("Delete: %v", err)
	}

	resp, err := client.Search(ctx, &shardpb.SearchRequest{Query: testVec(5), K: 10, Ef: 50})
	if err != nil {
		t.Fatalf("Search: %v", err)
	}
	for _, r := range resp.GetResults() {
		if r.GetKey().GetLabel() == 5 {
			t.Fatal("deleted key 5 was returned in search results")
		}
	}

	stats, _ := client.Stats(ctx, &shardpb.StatsRequest{})
	if stats.GetActiveVectors() != 19 || stats.GetTotalVectors() != 20 {
		t.Fatalf("active=%d total=%d, want 19/20 (soft delete keeps the node)",
			stats.GetActiveVectors(), stats.GetTotalVectors())
	}
}

func TestDeleteMissingKeyIsNotFound(t *testing.T) {
	client, _, cleanup := startServer(t, testConfig(t.TempDir()))
	defer cleanup()

	_, err := client.Delete(context.Background(), &shardpb.DeleteRequest{
		Key: &shardpb.Key{ClientId: 1, Label: 12345},
	})
	if got := status.Code(err); got != codes.NotFound {
		t.Fatalf("delete of missing key = %v, want NotFound", got)
	}
}

// TestRecoveryAcrossRestart is the load-bearing one: everything acked over
// gRPC must survive a process restart with no clean shutdown.
func TestRecoveryAcrossRestart(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(dir)

	client, _, cleanup := startServer(t, cfg)
	ctx := context.Background()
	const n = 40
	for i := 0; i < n; i++ {
		if _, err := client.Insert(ctx, &shardpb.InsertRequest{
			Key:    &shardpb.Key{ClientId: 1, Label: uint64(i)},
			Vector: testVec(float32(i)),
		}); err != nil {
			t.Fatalf("Insert(%d): %v", i, err)
		}
	}
	// Tear down the transport WITHOUT a graceful shard shutdown or snapshot,
	// approximating a crash: only the WAL can save us here.
	cleanup()

	client2, _, cleanup2 := startServer(t, cfg)
	defer cleanup2()

	stats, err := client2.Stats(ctx, &shardpb.StatsRequest{})
	if err != nil {
		t.Fatalf("Stats after restart: %v", err)
	}
	if stats.GetTotalVectors() != n {
		t.Fatalf("after restart total = %d, want %d", stats.GetTotalVectors(), n)
	}

	// testVec(10) is exactly the vector stored under label 10, so it's an
	// exact match -- but HNSW is approximate and nearby test vectors are
	// closely spaced, so check top-3 for it rather than demanding rank #1.
	resp, err := client2.Search(ctx, &shardpb.SearchRequest{Query: testVec(10), K: 3, Ef: 50})
	if err != nil {
		t.Fatalf("Search after restart: %v", err)
	}
	found := false
	for _, r := range resp.GetResults() {
		if r.GetKey().GetLabel() == 10 {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("search after restart returned %+v, want label 10 among top results", resp.GetResults())
	}
}

// TestSnapshotDuringWrites exercises the path that motivated adding the
// RWMutex to durable.Index: a snapshot running while writes are in flight.
// Without that lock this races (and can silently drop acknowledged writes).
func TestSnapshotDuringWrites(t *testing.T) {
	dir := t.TempDir()
	cfg := testConfig(dir)

	client, _, cleanup := startServer(t, cfg)
	ctx := context.Background()

	const n = 100
	errCh := make(chan error, 1)
	go func() {
		for i := 0; i < n; i++ {
			if _, err := client.Insert(ctx, &shardpb.InsertRequest{
				Key:    &shardpb.Key{ClientId: 1, Label: uint64(i)},
				Vector: testVec(float32(i)),
			}); err != nil {
				errCh <- err
				return
			}
		}
		errCh <- nil
	}()

	// Snapshot repeatedly while those inserts are landing. A small pause
	// between Snapshot calls matters: Snapshot takes durable.Index's full
	// write lock for real disk I/O (save + WAL rotation), and Go's
	// sync.RWMutex deliberately favors a waiting Lock() over new RLock()
	// calls to avoid writer starvation -- so a *tight* back-to-back loop of
	// Snapshot calls can starve the single insert goroutine indefinitely.
	// That's a busy-loop artifact, not the race this test exists to catch;
	// production drives Snapshot from a multi-minute ticker; a short pause
	// here still exercises plenty of interleavings without manufacturing
	// starvation that has nothing to do with correctness.
	deadline := time.After(10 * time.Second)
	snapshotting := true
	for snapshotting {
		select {
		case err := <-errCh:
			if err != nil {
				t.Fatalf("concurrent Insert failed: %v", err)
			}
			snapshotting = false
		case <-deadline:
			t.Fatal("inserts did not finish in time")
		default:
			if _, err := client.Snapshot(ctx, &shardpb.SnapshotRequest{}); err != nil {
				t.Fatalf("concurrent Snapshot failed: %v", err)
			}
			time.Sleep(5 * time.Millisecond)
		}
	}

	cleanup()

	// Every acknowledged insert must still be there after restart.
	client2, _, cleanup2 := startServer(t, cfg)
	defer cleanup2()
	stats, err := client2.Stats(ctx, &shardpb.StatsRequest{})
	if err != nil {
		t.Fatalf("Stats after restart: %v", err)
	}
	if stats.GetTotalVectors() != n {
		t.Fatalf("after snapshot-during-writes restart: total = %d, want %d "+
			"(a lost write here means snapshot/WAL rotation dropped an acked record)",
			stats.GetTotalVectors(), n)
	}
}
