package coordinator

import (
	"context"
	"fmt"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	shardpb "hnswdb/proto/shardpb"
)

// Pool holds one gRPC connection per shard, indexed by shard number. Shard
// membership is static configuration -- there is no discovery mechanism and
// no reshuffling at runtime, matching the fixed-shard-count scope documented
// in CLUSTER_ARCHITECTURE.md section 8.
type Pool struct {
	conns   []*grpc.ClientConn
	clients []shardpb.ShardServiceClient
}

// NewPool dials every address in order; addresses[i] becomes shard i. Dialing
// is lazy in the sense that grpc.NewClient does not block on connecting, so
// this returns quickly even if a shard is down -- failures surface on first
// use, not at startup. That matches the "shard down is normal" stance in the
// architecture doc.
func NewPool(addresses []string) (*Pool, error) {
	if len(addresses) == 0 {
		return nil, fmt.Errorf("coordinator: no shard addresses configured")
	}
	p := &Pool{
		conns:   make([]*grpc.ClientConn, len(addresses)),
		clients: make([]shardpb.ShardServiceClient, len(addresses)),
	}
	for i, addr := range addresses {
		conn, err := grpc.NewClient(addr, grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			p.Close() // don't leak connections already opened in this loop
			return nil, fmt.Errorf("coordinator: dialing shard %d (%s): %w", i, addr, err)
		}
		p.conns[i] = conn
		p.clients[i] = shardpb.NewShardServiceClient(conn)
	}
	return p, nil
}

func (p *Pool) NumShards() int { return len(p.clients) }

// NewPoolFromClients builds a Pool directly from existing gRPC clients,
// bypassing dialing entirely. This is the seam that lets tests wire a Pool
// to in-process bufconn servers instead of real network addresses -- see
// go/coordinator/coordinator_integration_test.go. Close on a Pool built this
// way is a no-op for connections, since it never opened any; callers remain
// responsible for closing whatever they passed in.
func NewPoolFromClients(clients []shardpb.ShardServiceClient) *Pool {
	return &Pool{clients: clients}
}

func (p *Pool) Shard(i int) shardpb.ShardServiceClient { return p.clients[i] }

func (p *Pool) Close() error {
	var firstErr error
	for _, c := range p.conns {
		if c == nil {
			continue
		}
		if err := c.Close(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// HealthCheck dials every shard's Stats RPC with a short timeout, for use at
// coordinator startup and by a readiness probe. It intentionally does not
// fail the coordinator's own startup -- a shard being briefly unreachable at
// boot is not a coordinator problem, per the same "shard down is normal"
// reasoning that governs Search's partial-failure handling.
func (p *Pool) HealthCheck(ctx context.Context) map[int]error {
	results := make(map[int]error)
	var mu sync.Mutex
	var wg sync.WaitGroup

	for i := range p.clients {
		wg.Add(1)
		go func(shard int) {
			defer wg.Done()
			cctx, cancel := context.WithTimeout(ctx, 3*time.Second)
			defer cancel()
			_, err := p.clients[shard].Stats(cctx, &shardpb.StatsRequest{})
			mu.Lock()
			results[shard] = err
			mu.Unlock()
		}(i)
	}
	wg.Wait()
	return results
}
