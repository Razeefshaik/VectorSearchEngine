package coordinator

import (
	"testing"

	"hnswdb/hnsw"
)

// These expected hashes were computed independently in Python, reproducing
// Go's fnv.New64a + binary.LittleEndian byte layout, and cross-checked
// against src/shard_sim.cpp's fnv1a64 for the same inputs. If this test
// fails, the Go router has drifted from the C++ implementation that
// shard_sim.cpp already validated for distribution and merge correctness --
// that validation would no longer apply to this code.
func TestHashKeyMatchesCppReference(t *testing.T) {
	cases := []struct {
		clientID, label uint64
		wantHash        uint64
	}{
		{1, 0, 0x392209f14dea4c24},
		{1, 1, 0x581cd0fa58d99645},
		{7, 42, 0x75ada7760b729448},
		{5, 99, 0x3972e90d804d0da3},
		{0, 0, 0x88201fb960ff6465},
		{^uint64(0), ^uint64(0), 0xd6607508f5a1e855},
	}
	for _, c := range cases {
		got := HashKey(hnsw.Key{ClientID: c.clientID, Label: c.label})
		if got != c.wantHash {
			t.Errorf("HashKey(%d, %d) = %#x, want %#x", c.clientID, c.label, got, c.wantHash)
		}
	}
}

// TestDistributionMatchesSimulation reproduces shard_sim.cpp's exact
// distribution test (20000 keys, 5 clients, 4 shards) and checks it lands on
// the same 4998/5002/4998/5002 split the C++ run produced.
func TestDistributionMatchesSimulation(t *testing.T) {
	r := NewRouter(4)
	counts := make([]int, 4)
	const n = 20000
	for i := 0; i < n; i++ {
		clientID := uint64(i%5) + 1
		counts[r.ShardFor(hnsw.Key{ClientID: clientID, Label: uint64(i)})]++
	}

	want := []int{4998, 5002, 4998, 5002}
	for s := range counts {
		if counts[s] != want[s] {
			t.Errorf("shard %d: got %d vectors, want %d (does this Go build "+
				"still match shard_sim.cpp's hash?)", s, counts[s], want[s])
		}
	}
}

func TestShardForIsInRange(t *testing.T) {
	r := NewRouter(7)
	for i := uint64(0); i < 1000; i++ {
		s := r.ShardFor(hnsw.Key{ClientID: i % 3, Label: i})
		if s < 0 || s >= 7 {
			t.Fatalf("ShardFor returned %d, want [0,7)", s)
		}
	}
}

func TestShardForIsDeterministic(t *testing.T) {
	r := NewRouter(4)
	key := hnsw.Key{ClientID: 7, Label: 42}
	first := r.ShardFor(key)
	for i := 0; i < 100; i++ {
		if got := r.ShardFor(key); got != first {
			t.Fatalf("ShardFor(%+v) = %d on call %d, want %d (routing must be pure)",
				key, got, i, first)
		}
	}
}
