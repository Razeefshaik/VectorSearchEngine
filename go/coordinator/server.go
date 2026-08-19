package coordinator

import (
	"context"
	"log/slog"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"hnswdb/hnsw"
	coordinatorpb "hnswdb/proto/coordinatorpb"
	shardpb "hnswdb/proto/shardpb"
)

type Server struct {
	coordinatorpb.UnimplementedVectorSearchServer

	pool      *Pool
	router    *Router
	defaultEf int
	log       *slog.Logger
}

type Config struct {
	Pool      *Pool
	Router    *Router
	DefaultEf int
	Logger    *slog.Logger
}

func NewServer(cfg Config) *Server {
	if cfg.DefaultEf <= 0 {
		cfg.DefaultEf = 100
	}
	log := cfg.Logger
	if log == nil {
		log = slog.Default()
	}
	return &Server{pool: cfg.Pool, router: cfg.Router, defaultEf: cfg.DefaultEf, log: log}
}

func (s *Server) Insert(ctx context.Context, req *coordinatorpb.InsertRequest) (*coordinatorpb.InsertResponse, error) {
	if req.GetKey() == nil {
		return nil, status.Error(codes.InvalidArgument, "key is required")
	}
	if len(req.GetVector()) == 0 {
		return nil, status.Error(codes.InvalidArgument, "vector is required")
	}

	key := hnsw.Key{ClientID: req.GetKey().GetClientId(), Label: req.GetKey().GetLabel()}
	shard := s.router.ShardFor(key)

	_, err := s.pool.Shard(shard).Insert(ctx, &shardpb.InsertRequest{
		Key:    &shardpb.Key{ClientId: key.ClientID, Label: key.Label},
		Vector: req.GetVector(),
	})
	if err != nil {
		// Pass the shard's status code through unchanged (AlreadyExists,
		// InvalidArgument, ResourceExhausted, ...) rather than wrapping it --
		// the client's retry logic needs the real code, and the shard already
		// chose it deliberately (see shard/server.go's toStatus).
		return nil, err
	}
	return &coordinatorpb.InsertResponse{}, nil
}

func (s *Server) Delete(ctx context.Context, req *coordinatorpb.DeleteRequest) (*coordinatorpb.DeleteResponse, error) {
	if req.GetKey() == nil {
		return nil, status.Error(codes.InvalidArgument, "key is required")
	}
	key := hnsw.Key{ClientID: req.GetKey().GetClientId(), Label: req.GetKey().GetLabel()}
	shard := s.router.ShardFor(key)

	_, err := s.pool.Shard(shard).Delete(ctx, &shardpb.DeleteRequest{
		Key: &shardpb.Key{ClientId: key.ClientID, Label: key.Label},
	})
	if err != nil {
		return nil, err
	}
	return &coordinatorpb.DeleteResponse{}, nil
}

func (s *Server) Search(ctx context.Context, req *coordinatorpb.SearchRequest) (*coordinatorpb.SearchResponse, error) {
	if len(req.GetQuery()) == 0 {
		return nil, status.Error(codes.InvalidArgument, "query is required")
	}
	k := int(req.GetK())
	if k <= 0 {
		return nil, status.Error(codes.InvalidArgument, "k must be > 0")
	}
	ef := int(req.GetEf())
	if ef <= 0 {
		ef = s.defaultEf
	}

	result, err := Search(ctx, s.pool, req.GetQuery(), k, ef, req.GetAllowPartial())
	if err != nil {
		return nil, err
	}

	if result.ShardsFailed > 0 {
		s.log.Warn("search completed with degraded shard coverage",
			"shards_queried", result.ShardsQueried,
			"shards_failed", result.ShardsFailed)
	}

	out := make([]*coordinatorpb.ScoredKey, len(result.Results))
	for i, c := range result.Results {
		out[i] = &coordinatorpb.ScoredKey{
			Key:      &coordinatorpb.Key{ClientId: c.key.GetClientId(), Label: c.key.GetLabel()},
			Distance: c.distance,
		}
	}
	return &coordinatorpb.SearchResponse{
		Results:       out,
		ShardsQueried: uint32(result.ShardsQueried),
		ShardsFailed:  uint32(result.ShardsFailed),
	}, nil
}
