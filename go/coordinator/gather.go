package coordinator

import (
	"context"
	"fmt"
	"sort"
	"sync"

	"golang.org/x/sync/errgroup"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	shardpb "hnswdb/proto/shardpb"
)

// candidate is one shard's scored result, kept alongside which shard it came
// from purely for observability (logging, debugging an uneven merge) -- it
// plays no role in the merge itself.
type candidate struct {
	key      *shardpb.Key
	distance float32
	shard    int
}

// GatherResult carries the merged results plus per-shard outcome, so a
// caller can distinguish "10/10 correct results" from "10/10 results, but
// only from 3 of 4 shards" -- see SearchResponse.shards_failed.
type GatherResult struct {
	Results       []candidate
	ShardsQueried int
	ShardsFailed  int
}

// Search asks every shard for the full k (never k/numShards -- see
// CLUSTER_ARCHITECTURE.md section 4 and src/shard_sim.cpp's Q3 for why that
// shortcut silently drops up to the entire correct result set) and merges by
// distance.
//
// allowPartial=false (the default): any shard error aborts the whole search
// and returns that error. The in-flight requests to other shards are
// canceled via the errgroup's shared context, not left to run to completion
// uselessly.
//
// allowPartial=true: a failed shard is recorded in ShardsFailed and excluded
// from the merge; the search still returns whatever the healthy shards found.
// This can under-return -- a completely correct top-10 might not exist if a
// shard is down, since that shard could hold some of the true nearest
// neighbours. That tradeoff is the caller's to make, not this function's.
func Search(ctx context.Context, pool *Pool, query []float32, k, ef int, allowPartial bool) (*GatherResult, error) {
	numShards := pool.NumShards()

	var (
		mu     sync.Mutex
		all    []candidate
		failed int
	)

	g, gctx := errgroup.WithContext(ctx)
	for s := 0; s < numShards; s++ {
		s := s // https://go.dev/wiki/CommonMistakes -- loop var capture,
		// same class of bug fixed in stress.cpp earlier in this project
		g.Go(func() error {
			resp, err := pool.Shard(s).Search(gctx, &shardpb.SearchRequest{
				Query: query,
				K:     uint32(k),
				Ef:    uint32(ef),
			})
			if err != nil {
				if allowPartial {
					mu.Lock()
					failed++
					mu.Unlock()
					return nil // swallow: this shard just contributes nothing
				}
				return fmt.Errorf("shard %d: %w", s, err)
			}

			mu.Lock()
			for _, r := range resp.GetResults() {
				all = append(all, candidate{key: r.GetKey(), distance: r.GetDistance(), shard: s})
			}
			mu.Unlock()
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, toGatherError(err)
	}

	sort.Slice(all, func(i, j int) bool { return all[i].distance < all[j].distance })
	if len(all) > k {
		all = all[:k]
	}

	return &GatherResult{
		Results:       all,
		ShardsQueried: numShards,
		ShardsFailed:  failed,
	}, nil
}

// toGatherError preserves the failing shard's gRPC code where there is one,
// so a client can distinguish "a shard rejected my request" (InvalidArgument
// -- their bug) from "a shard was unreachable" (Unavailable -- retry later)
// instead of everything collapsing into an opaque Internal.
func toGatherError(err error) error {
	if st, ok := status.FromError(err); ok && st.Code() != codes.Unknown {
		return err
	}
	return status.Errorf(codes.Internal, "scatter-gather: %v", err)
}
