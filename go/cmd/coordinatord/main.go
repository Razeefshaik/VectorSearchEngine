// Command coordinatord runs the stateless coordinator in front of a fixed
// set of shard servers.
//
//	coordinatord -listen :8000 -shards localhost:7001,localhost:7002,localhost:7003,localhost:7004
//
// The coordinator holds no vector data and no durable state of its own --
// restarting it loses nothing but in-flight requests. Run as many as you
// want behind a load balancer with zero coordination between them.
package main

import (
	"context"
	"flag"
	"log/slog"
	"net"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	healthpb "google.golang.org/grpc/health/grpc_health_v1"
	"google.golang.org/grpc/reflection"

	"hnswdb/coordinator"
	coordinatorpb "hnswdb/proto/coordinatorpb"
)

func main() {
	var (
		listen    = flag.String("listen", ":8000", "gRPC listen address")
		shardAddr = flag.String("shards", "", "comma-separated shard addresses, in shard-index order")
		defaultEf = flag.Int("default-ef", 100, "efSearch used when a request omits it")
	)
	flag.Parse()

	log := slog.New(slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{Level: slog.LevelInfo}))

	if *shardAddr == "" {
		log.Error("-shards is required, e.g. -shards localhost:7001,localhost:7002")
		os.Exit(1)
	}
	addrs := strings.Split(*shardAddr, ",")
	for i := range addrs {
		addrs[i] = strings.TrimSpace(addrs[i])
	}

	pool, err := coordinator.NewPool(addrs)
	if err != nil {
		log.Error("failed to create shard pool", "err", err)
		os.Exit(1)
	}
	defer pool.Close()

	// Fail loudly at startup if a shard is unreachable -- an operator running
	// coordinatord by hand wants to know now, not on the first client request.
	// This is diagnostic only: the coordinator still starts either way, since
	// "shard down" is a normal operating condition, not a coordinator fault.
	hctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	for shard, herr := range pool.HealthCheck(hctx) {
		if herr != nil {
			log.Warn("shard unreachable at startup", "shard", shard, "addr", addrs[shard], "err", herr)
		} else {
			log.Info("shard healthy", "shard", shard, "addr", addrs[shard])
		}
	}
	cancel()

	router := coordinator.NewRouter(len(addrs))
	coordServer := coordinator.NewServer(coordinator.Config{
		Pool: pool, Router: router, DefaultEf: *defaultEf, Logger: log,
	})

	lis, err := net.Listen("tcp", *listen)
	if err != nil {
		log.Error("cannot listen", "addr", *listen, "err", err)
		os.Exit(1)
	}

	grpcServer := grpc.NewServer(
		grpc.MaxRecvMsgSize(32*1024*1024),
		grpc.MaxSendMsgSize(32*1024*1024),
	)
	coordinatorpb.RegisterVectorSearchServer(grpcServer, coordServer)

	hs := health.NewServer()
	hs.SetServingStatus("", healthpb.HealthCheckResponse_SERVING)
	healthpb.RegisterHealthServer(grpcServer, hs)
	reflection.Register(grpcServer)

	go func() {
		log.Info("coordinator listening", "addr", *listen, "shards", addrs)
		if err := grpcServer.Serve(lis); err != nil {
			log.Error("grpc serve stopped", "err", err)
		}
	}()

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	sig := <-sigCh
	log.Info("shutting down", "signal", sig.String())

	hs.SetServingStatus("", healthpb.HealthCheckResponse_NOT_SERVING)
	grpcServer.GracefulStop()
	log.Info("shutdown complete")
}
