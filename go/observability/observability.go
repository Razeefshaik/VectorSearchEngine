// Package observability wires Prometheus metrics and an HTTP liveness probe
// for coordinatord and shardd. Both processes already register the gRPC
// Health Checking Protocol (google.golang.org/grpc/health) for their own
// liveness semantics -- this package is additive, not a replacement: it
// gives Prometheus something to scrape (gRPC has no native metrics
// exposition format) on a separate HTTP port, so a metrics-server outage
// can never affect gRPC serving or the existing health service.
//
// Metric names follow the vsgw_<service>_... convention used across the
// whole vectorsearch-gateway system (see that repo's docs/MONITORING.md),
// so coordinatord/shardd metrics read as one family with gatewayd/consumer/
// embed-service metrics in the shared Grafana dashboard, not a separate
// system bolted on.
package observability

import (
	"context"
	"log/slog"
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"google.golang.org/grpc"
	"google.golang.org/grpc/status"
)

// StartServer exposes Prometheus metrics on /metrics and a liveness probe on
// /healthz at addr (e.g. ":9105"). Listen failures are logged, not fatal --
// a metrics outage should never take down gRPC serving.
func StartServer(addr, serviceName string, log *slog.Logger) *http.Server {
	mux := http.NewServeMux()
	mux.Handle("/metrics", promhttp.Handler())
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	srv := &http.Server{Addr: addr, Handler: mux}
	go func() {
		log.Info("metrics/health server listening", "service", serviceName, "addr", addr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Warn("metrics server stopped", "service", serviceName, "err", err)
		}
	}()
	return srv
}

// Shutdown gives the metrics server up to 5s to stop cleanly.
func Shutdown(srv *http.Server) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = srv.Shutdown(ctx)
}

// GRPCServerMetrics is the per-RPC instrumentation recorded by
// UnaryInterceptor. Construct one per process via NewGRPCServerMetrics and
// install the interceptor at grpc.NewServer time.
type GRPCServerMetrics struct {
	requestsTotal   *prometheus.CounterVec
	requestDuration *prometheus.HistogramVec
	inFlight        *prometheus.GaugeVec
}

// NewGRPCServerMetrics registers gRPC server metrics under namespace (e.g.
// "vsgw_coordinator" or "vsgw_shard"). Call once per process.
func NewGRPCServerMetrics(namespace string) *GRPCServerMetrics {
	m := &GRPCServerMetrics{
		requestsTotal: prometheus.NewCounterVec(prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: "grpc",
			Name:      "requests_total",
			Help:      "Total unary gRPC requests handled, by method and status code.",
		}, []string{"method", "code"}),
		requestDuration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Namespace: namespace,
			Subsystem: "grpc",
			Name:      "request_duration_seconds",
			Help:      "Unary gRPC request latency in seconds, by method.",
			Buckets:   prometheus.DefBuckets,
		}, []string{"method"}),
		inFlight: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Namespace: namespace,
			Subsystem: "grpc",
			Name:      "requests_in_flight",
			Help:      "Unary gRPC requests currently being handled, by method.",
		}, []string{"method"}),
	}
	prometheus.MustRegister(m.requestsTotal, m.requestDuration, m.inFlight)
	return m
}

// UnaryInterceptor records request count, latency, and in-flight gauge for
// every unary RPC this server handles. It never alters the response or
// error -- purely observational.
func (m *GRPCServerMetrics) UnaryInterceptor() grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		m.inFlight.WithLabelValues(info.FullMethod).Inc()
		defer m.inFlight.WithLabelValues(info.FullMethod).Dec()

		start := time.Now()
		resp, err := handler(ctx, req)
		duration := time.Since(start).Seconds()

		code := status.Code(err)
		m.requestsTotal.WithLabelValues(info.FullMethod, code.String()).Inc()
		m.requestDuration.WithLabelValues(info.FullMethod).Observe(duration)

		return resp, err
	}
}
