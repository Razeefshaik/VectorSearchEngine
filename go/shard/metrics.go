package shard

import (
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// Index-state gauges and snapshot metrics live here because, unlike the
// generic gRPC request metrics (see go/observability), they reflect the
// durable.Index's own state rather than anything request-scoped -- updated
// after every mutation and by the background snapshot loop, not by an
// interceptor.
var (
	indexVectors = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: "vsgw_shard",
		Subsystem: "index",
		Name:      "vectors_total",
		Help:      "Total vectors in this shard's index, including tombstoned (deleted-but-not-compacted) entries.",
	})
	indexActiveVectors = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: "vsgw_shard",
		Subsystem: "index",
		Name:      "active_vectors",
		Help:      "Vectors in this shard's index that are not tombstoned -- what Search actually considers.",
	})
	indexCapacity = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: "vsgw_shard",
		Subsystem: "index",
		Name:      "capacity",
		Help:      "Fixed max-elements capacity this shard was created with.",
	})
	indexMemoryBytes = prometheus.NewGauge(prometheus.GaugeOpts{
		Namespace: "vsgw_shard",
		Subsystem: "index",
		Name:      "memory_bytes",
		Help:      "Approximate memory held by the C++ HNSW index arena.",
	})

	snapshotsTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "vsgw_shard",
		Subsystem: "snapshot",
		Name:      "total",
		Help:      "Snapshot attempts, by result.",
	}, []string{"result"})
	snapshotDuration = prometheus.NewHistogram(prometheus.HistogramOpts{
		Namespace: "vsgw_shard",
		Subsystem: "snapshot",
		Name:      "duration_seconds",
		Help:      "Time to complete Index.Snapshot(), which blocks writes for its duration -- see durable.Index.mu.",
		Buckets:   []float64{.01, .05, .1, .5, 1, 5, 10, 30, 60},
	})
)

func init() {
	prometheus.MustRegister(
		indexVectors, indexActiveVectors, indexCapacity, indexMemoryBytes,
		snapshotsTotal, snapshotDuration,
	)
}

// updateIndexGauges refreshes the index-size gauges from the current index
// state. Called after every mutating RPC (cheap -- these are field reads
// on the C++ side, not a scan) and once at startup.
func (s *Server) updateIndexGauges() {
	indexVectors.Set(float64(s.idx.Len()))
	indexActiveVectors.Set(float64(s.idx.ActiveLen()))
	indexCapacity.Set(float64(s.idx.Capacity()))
	indexMemoryBytes.Set(float64(s.idx.MemoryBytes()))
}

// recordSnapshot is called around every Index.Snapshot() call, scheduled or
// on-demand via the Snapshot RPC, so snapshot duration/failure is visible
// per shard without grepping logs.
func recordSnapshot(start time.Time, err error) {
	result := "success"
	if err != nil {
		result = "error"
	}
	snapshotsTotal.WithLabelValues(result).Inc()
	snapshotDuration.Observe(time.Since(start).Seconds())
}
