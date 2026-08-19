// Integration tests for the full coordinator + shard stack. These need the
// C++ shared library built (cmake -B build && cmake --build build -j) and
// run real gRPC servers over in-process bufconn listeners -- no TCP ports,
// but the full marshal/unmarshal/routing/merge path is exercised for real.
package coordinator

import (
	"context"
	"net"
	"path/filepath"
	"sort"
	"sync"
	"testing"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"

	"hnswdb/durable"
	"hnswdb/hnsw"
	coordinatorpb "hnswdb/proto/coordinatorpb"
	shardpb "hnswdb/proto/shardpb"
	"hnswdb/shard"
)

const testDim = 16

func testVec(seed float32) []float32 {
	v := make([]float32, testDim)
	for i := range v {
		v[i] = seed + float32(i)*0.013
	}
	return v
}

type pair struct{ clientID, label uint64 }

type scoredPair struct {
	key      pair
	distance float32
}

// tieEpsilon is the distance tolerance for treating two candidates as tied.
// testVec's near-parallel synthetic vectors routinely put several distinct
// keys within float32 rounding noise of each other (observed gaps around
// 1e-6 in cosine distance), so an absolute tolerance of 1e-5 -- roughly 100x
// float32 machine epsilon -- comfortably covers that noise on real distance
// computations without being loose enough to hide an actual scoring bug on
// data with realistic separation.
const tieEpsilon = 1e-5

// assertTopKMatchesWithTieTolerance checks that the coordinator's merged
// top-k and a monolithic index's top-k contain the same candidates, EXCEPT
// where the sharded graphs (4 separate graphs, different node orderings) and
// the monolithic graph (one graph) can legitimately disagree: candidates
// tied at or near the k-th cutoff distance. Whichever tied candidate lands
// in the last slot is order-dependent on graph structure, not a routing or
// merge defect -- see the query 6 case where both sides agreed on 9/10 keys
// and the 10th pair ({1,40} vs {1,50}) shared the identical float32 distance
// 3.8146973e-06.
//
// Candidates strictly outside the tie band at the cutoff must match exactly:
// a real routing or merge bug (wrong shard, dropped shard, bad sort) would
// show up as a mismatch there, not just at the boundary.
func assertTopKMatchesWithTieTolerance(t *testing.T, q int, coord, mono []scoredPair) {
	t.Helper()

	sortByDistance := func(s []scoredPair) {
		sort.Slice(s, func(i, j int) bool { return s[i].distance < s[j].distance })
	}
	sortByDistance(coord)
	sortByDistance(mono)

	k := len(coord)
	if k == 0 {
		return
	}
	// The boundary is the looser (larger) of the two k-th distances: a
	// candidate within tieEpsilon of EITHER side's cutoff is in the
	// contested zone, regardless of which side is doing the looking.
	boundary := coord[k-1].distance
	if mono[k-1].distance > boundary {
		boundary = mono[k-1].distance
	}
	core := boundary - tieEpsilon

	coreKeys := func(s []scoredPair) map[pair]bool {
		m := make(map[pair]bool)
		for _, sp := range s {
			if sp.distance < core {
				m[sp.key] = true
			}
		}
		return m
	}
	coordCore := coreKeys(coord)
	monoCore := coreKeys(mono)

	if len(coordCore) != len(monoCore) {
		t.Fatalf("query %d: unambiguous (non-tied) result counts differ: coordinator=%d, monolithic=%d\n"+
			"  coordinator: %+v\n  monolithic:  %+v", q, len(coordCore), len(monoCore), coord, mono)
	}
	for key := range coordCore {
		if !monoCore[key] {
			t.Fatalf("query %d: unambiguous (non-tied, distance well inside cutoff) key %+v "+
				"present in coordinator's top-k but not monolithic's -- this is NOT a tie-breaking "+
				"artifact, it points at a real routing or merge bug\n"+
				"  coordinator: %+v\n  monolithic:  %+v", q, key, coord, mono)
		}
	}
}

// cluster bundles everything an integration test needs and cleans up after
// itself: numShards in-process shard servers, a Pool wired to them over
// bufconn, and a ready-to-use coordinator Server.
type cluster struct {
	server *Server
	pool   *Pool
	router *Router
	shards []*shard.Server
	closer func()

	// transportClosers[i] tears down ONLY shard i's gRPC transport (client
	// conn + server), leaving its durable.Index and WAL untouched. This is
	// what actually simulates "shard unreachable" for a single shard --
	// shard.Server.Close() alone does not: it only closes the WAL file, and
	// durable.Index.Search() deliberately never touches the WAL (see the
	// comment on durable.Index.Search), so a WAL-closed shard keeps serving
	// Search calls just fine. Killing the transport is what actually makes
	// RPCs to that shard fail.
	transportClosers []func()
}

// killShard severs shard i's gRPC transport so calls to it fail the way a
// genuinely unreachable shard would, without touching the other shards or
// this shard's underlying index/WAL state.
func (c *cluster) killShard(i int) {
	c.transportClosers[i]()
}

func startCluster(t *testing.T, numShards int) *cluster {
	t.Helper()

	shards := make([]*shard.Server, numShards)
	clients := make([]shardpb.ShardServiceClient, numShards)
	var closers []func()
	var transportClosers []func()

	for i := 0; i < numShards; i++ {
		dir := t.TempDir()
		srv, err := shard.New(shard.Config{
			Durable: durable.Config{
				SnapshotPath:   filepath.Join(dir, "index.snapshot"),
				WALPath:        filepath.Join(dir, "index.wal"),
				Space:          hnsw.Cosine,
				Dim:            testDim,
				MaxElements:    5000,
				M:              16,
				EfConstruction: 100,
				Seed:           uint64(42 + i),
			},
			DefaultEf: 100,
		})
		if err != nil {
			t.Fatalf("shard.New(%d): %v", i, err)
		}
		shards[i] = srv

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
			t.Fatalf("dial shard %d: %v", i, err)
		}
		clients[i] = shardpb.NewShardServiceClient(conn)

		// No manual per-iteration capture needed: go.mod targets Go 1.25,
		// where `for i := ...` variables (including srv/gs/conn declared
		// earlier in this same loop body) are already scoped per iteration.
		var killOnce sync.Once
		killTransport := func() {
			killOnce.Do(func() {
				conn.Close()
				gs.Stop()
			})
		}
		transportClosers = append(transportClosers, killTransport)
		closers = append(closers, func() {
			killTransport()
			srv.Close()
		})
	}

	pool := NewPoolFromClients(clients)
	router := NewRouter(numShards)
	coordServer := NewServer(Config{Pool: pool, Router: router, DefaultEf: 100})

	return &cluster{
		server: coordServer,
		pool:   pool,
		router: router,
		shards: shards,
		closer: func() {
			for _, c := range closers {
				c()
			}
		},
		transportClosers: transportClosers,
	}
}

func TestInsertRoutesToExactlyOneShard(t *testing.T) {
	c := startCluster(t, 4)
	defer c.closer()
	ctx := context.Background()

	key := hnsw.Key{ClientID: 1, Label: 42}
	wantShard := c.router.ShardFor(key)

	_, err := c.server.Insert(ctx, &coordinatorpb.InsertRequest{
		Key:    &coordinatorpb.Key{ClientId: key.ClientID, Label: key.Label},
		Vector: testVec(1),
	})
	if err != nil {
		t.Fatalf("Insert: %v", err)
	}

	for i, s := range c.shards {
		want := 0
		if i == wantShard {
			want = 1
		}
		if got := s.Len(); got != want {
			t.Errorf("shard %d has %d vectors, want %d (key should route to shard %d only)",
				i, got, want, wantShard)
		}
	}
}

// TestCoordinatorMatchesMonolithicIndex is the correctness proof for the
// entire merge design: insert the same data into (a) a sharded cluster
// through the coordinator, and (b) a single monolithic index, then confirm
// the coordinator's top-k is identical -- not just similar-recall, but the
// same keys, same order. If scatter-gather is implemented correctly this
// must hold, because both sides run the identical HNSW search algorithm; any
// divergence points at a routing or merge bug, not an ANN approximation gap.
func TestCoordinatorMatchesMonolithicIndex(t *testing.T) {
	const n = 2000
	const numShards = 4
	const numClients = 5
	const k = 10

	c := startCluster(t, numShards)
	defer c.closer()
	ctx := context.Background()

	mono, err := hnsw.New(hnsw.Cosine, testDim, n, 16, 100, 42)
	if err != nil {
		t.Fatalf("hnsw.New: %v", err)
	}
	defer mono.Close()

	for i := 0; i < n; i++ {
		key := hnsw.Key{ClientID: uint64(i%numClients) + 1, Label: uint64(i)}
		vec := testVec(float32(i) * 0.05)

		if _, err := c.server.Insert(ctx, &coordinatorpb.InsertRequest{
			Key:    &coordinatorpb.Key{ClientId: key.ClientID, Label: key.Label},
			Vector: vec,
		}); err != nil {
			t.Fatalf("coordinator Insert(%d): %v", i, err)
		}
		if err := mono.Add(vec, key); err != nil {
			t.Fatalf("monolithic Add(%d): %v", i, err)
		}
	}

	// Several query points, not just one -- a routing bug that only affects
	// some shards could hide behind a single lucky query.
	for q := 0; q < 20; q++ {
		query := testVec(float32(q) * 0.37)

		coordResp, err := c.server.Search(ctx, &coordinatorpb.SearchRequest{
			Query: query, K: uint32(k), Ef: 200,
		})
		if err != nil {
			t.Fatalf("coordinator Search(query %d): %v", q, err)
		}
		if got := int(coordResp.GetShardsFailed()); got != 0 {
			t.Fatalf("query %d: %d shards failed, want 0", q, got)
		}

		monoResults, err := mono.Search(query, k, 200)
		if err != nil {
			t.Fatalf("monolithic Search(query %d): %v", q, err)
		}

		if len(coordResp.GetResults()) != len(monoResults) {
			t.Fatalf("query %d: coordinator returned %d results, monolithic returned %d",
				q, len(coordResp.GetResults()), len(monoResults))
		}

		coordScored := make([]scoredPair, len(coordResp.GetResults()))
		for i, r := range coordResp.GetResults() {
			coordScored[i] = scoredPair{
				key:      pair{r.GetKey().GetClientId(), r.GetKey().GetLabel()},
				distance: r.GetDistance(),
			}
		}
		monoScored := make([]scoredPair, len(monoResults))
		for i, r := range monoResults {
			monoScored[i] = scoredPair{
				key:      pair{r.Key.ClientID, r.Key.Label},
				distance: r.Distance,
			}
		}

		assertTopKMatchesWithTieTolerance(t, q, coordScored, monoScored)
	}
}

func TestSearchAllowPartialToleratesOneShardDown(t *testing.T) {
	c := startCluster(t, 4)
	ctx := context.Background()

	for i := 0; i < 40; i++ {
		key := hnsw.Key{ClientID: 1, Label: uint64(i)}
		if _, err := c.server.Insert(ctx, &coordinatorpb.InsertRequest{
			Key:    &coordinatorpb.Key{ClientId: key.ClientID, Label: key.Label},
			Vector: testVec(float32(i)),
		}); err != nil {
			t.Fatalf("Insert(%d): %v", i, err)
		}
	}

	// Take shard 0 down without going through c.closer (which tears down
	// everything): sever just its gRPC transport. shard.Server.Close() alone
	// would NOT do this -- it only closes the WAL file, and
	// durable.Index.Search() never touches the WAL, so a WAL-closed shard
	// keeps answering Search calls fine. Killing the transport is what
	// actually makes RPCs to shard 0 fail the way a downed shard would.
	c.killShard(0)

	_, err := c.server.Search(ctx, &coordinatorpb.SearchRequest{
		Query: testVec(10), K: 5, Ef: 50, AllowPartial: false,
	})
	if err == nil {
		t.Fatal("expected Search to fail with one shard down and allowPartial=false")
	}

	resp, err := c.server.Search(ctx, &coordinatorpb.SearchRequest{
		Query: testVec(10), K: 5, Ef: 50, AllowPartial: true,
	})
	if err != nil {
		t.Fatalf("Search with allowPartial=true should not fail outright: %v", err)
	}
	if resp.GetShardsFailed() != 1 {
		t.Fatalf("shards_failed = %d, want 1", resp.GetShardsFailed())
	}
	if resp.GetShardsQueried() != 4 {
		t.Fatalf("shards_queried = %d, want 4 (queried count is independent of failures)",
			resp.GetShardsQueried())
	}

	c.closer()
}

func TestDeleteMissingKeyPropagatesNotFound(t *testing.T) {
	c := startCluster(t, 4)
	defer c.closer()

	_, err := c.server.Delete(context.Background(), &coordinatorpb.DeleteRequest{
		Key: &coordinatorpb.Key{ClientId: 1, Label: 99999},
	})
	if err == nil {
		t.Fatal("expected Delete of a missing key to fail")
	}
}
