// Package shard exposes a single durable.Index over gRPC.
//
// This layer is deliberately thin: unmarshal, validate, call durable.Index,
// marshal. It holds no vector data of its own and knows nothing about other
// shards or about routing. If logic starts accumulating here, it probably
// belongs in the coordinator or in durable.
package shard

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"hnswdb/durable"
	"hnswdb/hnsw"
	shardpb "hnswdb/proto/shardpb"
)

type Config struct {
	// Durable holds index + WAL + snapshot paths and index parameters.
	Durable durable.Config

	// DefaultEf is used when a SearchRequest leaves ef at 0, so the
	// coordinator need not know each shard's tuning.
	DefaultEf int

	// SnapshotInterval drives the background snapshot ticker. Zero disables
	// it -- but note that without snapshots the WAL grows without bound and
	// recovery gets slower forever, so leave it on outside of tests.
	SnapshotInterval time.Duration

	Logger *slog.Logger
}

type Server struct {
	shardpb.UnimplementedShardServiceServer

	idx *durable.Index
	cfg Config
	log *slog.Logger

	stopOnce sync.Once
	stopCh   chan struct{}
	wg       sync.WaitGroup
}

// New opens (or recovers) the underlying index. It does not start the
// background snapshot loop; call Start for that.
func New(cfg Config) (*Server, error) {
	if cfg.DefaultEf <= 0 {
		cfg.DefaultEf = 100
	}
	log := cfg.Logger
	if log == nil {
		log = slog.Default()
	}

	idx, err := durable.Open(cfg.Durable)
	if err != nil {
		return nil, fmt.Errorf("shard: opening index: %w", err)
	}

	log.Info("shard index ready",
		"vectors", idx.Len(),
		"active", idx.ActiveLen(),
		"dim", idx.Dim(),
		"snapshot", cfg.Durable.SnapshotPath,
		"wal", cfg.Durable.WALPath)

	return &Server{
		idx:    idx,
		cfg:    cfg,
		log:    log,
		stopCh: make(chan struct{}),
	}, nil
}

// Start launches the background snapshot ticker.
func (s *Server) Start() {
	if s.cfg.SnapshotInterval <= 0 {
		s.log.Warn("background snapshots disabled; WAL will grow without bound")
		return
	}
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		t := time.NewTicker(s.cfg.SnapshotInterval)
		defer t.Stop()
		for {
			select {
			case <-s.stopCh:
				return
			case <-t.C:
				start := time.Now()
				if err := s.idx.Snapshot(); err != nil {
					// Not fatal: the WAL still holds everything since the last
					// good snapshot, so durability is intact. It just means
					// recovery will replay more and the WAL keeps growing.
					s.log.Error("periodic snapshot failed", "err", err)
					continue
				}
				s.log.Info("snapshot complete",
					"took", time.Since(start),
					"vectors", s.idx.Len())
			}
		}
	}()
}

// Close stops the snapshot loop, takes one final snapshot so restart does not
// have to replay the whole WAL, and closes the index.
func (s *Server) Close() error {
	var err error
	s.stopOnce.Do(func() {
		close(s.stopCh)
		s.wg.Wait()

		if s.cfg.SnapshotInterval > 0 {
			if serr := s.idx.Snapshot(); serr != nil {
				s.log.Error("final snapshot failed", "err", serr)
			}
		}
		err = s.idx.Close()
	})
	return err
}

// ---------------------------------------------------------------------------
// RPC handlers
// ---------------------------------------------------------------------------

func (s *Server) Insert(ctx context.Context, req *shardpb.InsertRequest) (*shardpb.InsertResponse, error) {
	if req.GetKey() == nil {
		return nil, status.Error(codes.InvalidArgument, "key is required")
	}
	if len(req.GetVector()) == 0 {
		return nil, status.Error(codes.InvalidArgument, "vector is required")
	}
	if got, want := len(req.GetVector()), s.idx.Dim(); got != want {
		return nil, status.Errorf(codes.InvalidArgument,
			"vector has %d dimensions, this shard is configured for %d", got, want)
	}

	key := keyFromProto(req.GetKey())
	if err := s.idx.Add(req.GetVector(), key); err != nil {
		return nil, toStatus(err, "insert")
	}
	return &shardpb.InsertResponse{}, nil
}

func (s *Server) Delete(ctx context.Context, req *shardpb.DeleteRequest) (*shardpb.DeleteResponse, error) {
	if req.GetKey() == nil {
		return nil, status.Error(codes.InvalidArgument, "key is required")
	}
	if err := s.idx.MarkDeleted(keyFromProto(req.GetKey())); err != nil {
		return nil, toStatus(err, "delete")
	}
	return &shardpb.DeleteResponse{}, nil
}

func (s *Server) Undelete(ctx context.Context, req *shardpb.UndeleteRequest) (*shardpb.UndeleteResponse, error) {
	if req.GetKey() == nil {
		return nil, status.Error(codes.InvalidArgument, "key is required")
	}
	if err := s.idx.UnmarkDeleted(keyFromProto(req.GetKey())); err != nil {
		return nil, toStatus(err, "undelete")
	}
	return &shardpb.UndeleteResponse{}, nil
}

func (s *Server) Search(ctx context.Context, req *shardpb.SearchRequest) (*shardpb.SearchResponse, error) {
	if len(req.GetQuery()) == 0 {
		return nil, status.Error(codes.InvalidArgument, "query is required")
	}
	if got, want := len(req.GetQuery()), s.idx.Dim(); got != want {
		return nil, status.Errorf(codes.InvalidArgument,
			"query has %d dimensions, this shard is configured for %d", got, want)
	}
	k := int(req.GetK())
	if k <= 0 {
		return nil, status.Error(codes.InvalidArgument, "k must be > 0")
	}
	ef := int(req.GetEf())
	if ef <= 0 {
		ef = s.cfg.DefaultEf
	}

	// Bail out early if the caller already gave up. Saves doing the search at
	// all when the coordinator's deadline has passed.
	if err := ctx.Err(); err != nil {
		return nil, status.FromContextError(err).Err()
	}

	results, err := s.idx.Search(req.GetQuery(), k, ef)
	if err != nil {
		return nil, toStatus(err, "search")
	}

	out := make([]*shardpb.ScoredKey, len(results))
	for i, r := range results {
		out[i] = &shardpb.ScoredKey{
			Key:      &shardpb.Key{ClientId: r.Key.ClientID, Label: r.Key.Label},
			Distance: r.Distance,
		}
	}
	return &shardpb.SearchResponse{Results: out}, nil
}

func (s *Server) Snapshot(ctx context.Context, _ *shardpb.SnapshotRequest) (*shardpb.SnapshotResponse, error) {
	if err := s.idx.Snapshot(); err != nil {
		return nil, toStatus(err, "snapshot")
	}
	return &shardpb.SnapshotResponse{}, nil
}

func (s *Server) Stats(ctx context.Context, _ *shardpb.StatsRequest) (*shardpb.StatsResponse, error) {
	return &shardpb.StatsResponse{
		TotalVectors:  uint64(s.idx.Len()),
		ActiveVectors: uint64(s.idx.ActiveLen()),
		Dim:           uint32(s.idx.Dim()),
		Capacity:      uint64(s.idx.Capacity()),
		MemoryBytes:   uint64(s.idx.MemoryBytes()),
	}, nil
}

// ---------------------------------------------------------------------------

func keyFromProto(k *shardpb.Key) hnsw.Key {
	return hnsw.Key{ClientID: k.GetClientId(), Label: k.GetLabel()}
}

// toStatus maps domain errors onto gRPC codes. This matters more than it
// looks: the coordinator makes retry and idempotency decisions from the code
// alone, so an ALREADY_EXISTS must never be conflated with an INTERNAL.
func toStatus(err error, op string) error {
	switch {
	case errors.Is(err, hnsw.ErrDuplicateLabel):
		return status.Errorf(codes.AlreadyExists, "%s: key already present", op)
	case errors.Is(err, hnsw.ErrNotFound):
		return status.Errorf(codes.NotFound, "%s: key not found", op)
	case errors.Is(err, hnsw.ErrFull):
		return status.Errorf(codes.ResourceExhausted,
			"%s: shard is at capacity (maxElements reached)", op)
	default:
		return status.Errorf(codes.Internal, "%s: %v", op, err)
	}
}
