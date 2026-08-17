// Package coordinator implements the stateless routing/fan-out layer that
// sits in front of a fixed set of shard servers.
package coordinator

import (
	"encoding/binary"
	"hash/fnv"

	"hnswdb/hnsw"
)

// Router maps a key to a shard index. It holds no state beyond the shard
// count -- every coordinator process computes the same answer for the same
// key with no coordination, no lookup table, and no cache to invalidate.
//
// The hash MUST match src/shard_sim.cpp's fnv1a64(clientId || label) exactly:
// FNV-1a over the 16 bytes of clientId then label, each little-endian. That
// simulation is what validated even distribution and scatter-gather
// correctness against the real HNSW core; if this implementation drifts from
// it, that validation no longer says anything about this code.
type Router struct {
	numShards int
}

func NewRouter(numShards int) *Router {
	if numShards <= 0 {
		panic("coordinator: numShards must be > 0")
	}
	return &Router{numShards: numShards}
}

func (r *Router) NumShards() int { return r.numShards }

// ShardFor returns which shard owns key, in [0, NumShards()).
func (r *Router) ShardFor(key hnsw.Key) int {
	return int(HashKey(key) % uint64(r.numShards))
}

// HashKey is exported separately from ShardFor so tests (and anything doing
// capacity planning) can inspect the raw hash without also depending on a
// particular shard count.
func HashKey(key hnsw.Key) uint64 {
	h := fnv.New64a()
	var buf [16]byte
	binary.LittleEndian.PutUint64(buf[0:8], key.ClientID)
	binary.LittleEndian.PutUint64(buf[8:16], key.Label)
	h.Write(buf[:]) // hash.Hash64.Write never returns an error
	return h.Sum64()
}
