// Command shardd runs a single shard: one durable HNSW index behind a gRPC
// server.
//
//	shardd -listen :7001 -data ./data/shard0 -dim 128 -max-elements 300000
//
// Shard identity lives entirely in the coordinator's config, not here -- a
// shard does not know its own index number, which keeps it substitutable.
package main

import (
	"flag"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"

	"hnswdb/durable"
	"hnswdb/hnsw"
	shardpb "hnswdb/proto/shardpb"
	"hnswdb/shard"
)

func main() {
	var (
		listen           = flag.String("listen", ":7001", "gRPC listen address")
		dataDir          = flag.String("data", "./data/shard0", "directory for snapshot + WAL")
		dim              = flag.Int("dim", 128, "vector dimensionality")
		maxElements      = flag.Int("max-elements", 1_000_000, "index capacity (fixed at creation)")
		m                = flag.Int("m", 16, "HNSW M (graph out-degree)")
		efConstruction   = flag.Int("ef-construction", 200, "HNSW efConstruction (build beam width)")
		defaultEf        = flag.Int("default-ef", 100, "efSearch used when a request omits it")
		spaceFlag        = flag.String("space", "cosine", "distance space: cosine | l2")
		snapshotInterval = flag.Duration("snapshot-interval", 5*time.Minute, "background snapshot period (0 disables)")
		seed             = flag.Uint64("seed", 100, "RNG seed for level assignment")
	)
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))

	var space hnsw.Space
	switch *spaceFlag {
	case "cosine":
		space = hnsw.Cosine
	case "l2":
		space = hnsw.L2
	default:
		log.Error("invalid -space, want cosine or l2", "got", *spaceFlag)
		os.Exit(1)
	}

	if err := os.MkdirAll(*dataDir, 0o755); err != nil {
		log.Error("cannot create data directory", "dir", *dataDir, "err", err)
		os.Exit(1)
	}

	srv, err := shard.New(shard.Config{
		Durable: durable.Config{
			SnapshotPath:   filepath.Join(*dataDir, "index.snapshot"),
			WALPath:        filepath.Join(*dataDir, "index.wal"),
			Space:          space,
			Dim:            *dim,
			MaxElements:    *maxElements,
			M:              *m,
			EfConstruction: *efConstruction,
			Seed:           *seed,
		},
		DefaultEf:        *defaultEf,
		SnapshotInterval: *snapshotInterval,
		Logger:           log,
	})
	if err != nil {
		log.Error("failed to start shard", "err", err)
		os.Exit(1)
	}
	srv.Start()

	lis, err := net.Listen("tcp", *listen)
	if err != nil {
		log.Error("cannot listen", "addr", *listen, "err", err)
		srv.Close()
		os.Exit(1)
	}

	// A 768-dim float32 vector is ~3 KB on the wire; the 4 MB default would
	// cap a bulk insert batch well below what is useful. Raise both ends.
	grpcServer := grpc.NewServer(
		grpc.MaxRecvMsgSize(32*1024*1024),
		grpc.MaxSendMsgSize(32*1024*1024),
	)
	shardpb.RegisterShardServiceServer(grpcServer, srv)

	// Health service so the coordinator (or k8s) can probe readiness.
	hs := health.NewServer()
	hs.SetServingStatus("", healthpb.HealthCheckResponse_SERVING)
	healthpb.RegisterHealthServer(grpcServer, hs)

	// Reflection makes grpcurl work without hand-feeding it the .proto, which
	// is worth a lot when debugging by hand.
	reflection.Register(grpcServer)

	go func() {
		log.Info("shard listening", "addr", *listen, "data", *dataDir)
		if err := grpcServer.Serve(lis); err != nil {
			log.Error("grpc serve stopped", "err", err)
		}
	}()

	// Graceful shutdown: stop accepting, drain in-flight RPCs, then snapshot.
	// Order matters -- snapshotting first would race the writes still in
	// flight, and those writes are exactly what we want captured.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	sig := <-sigCh
	log.Info("shutting down", "signal", sig.String())

	hs.SetServingStatus("", healthpb.HealthCheckResponse_NOT_SERVING)
	grpcServer.GracefulStop()

	if err := srv.Close(); err != nil {
		log.Error("error during shutdown", "err", err)
		os.Exit(1)
	}
	log.Info("shutdown complete")
}
