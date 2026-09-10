package coordinator

import (
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"
)

// Metrics beyond the generic gRPC server interceptor (see go/observability)
// live here because they need context the interceptor doesn't have: which
// shard a call went to, and how many shards a given Search actually reached
// vs. lost. "shard" is the numeric index rather than a network address --
// addresses are static per-process config (see Pool), so the index is the
// stable, low-cardinality label.
var (
	shardCallsTotal = prometheus.NewCounterVec(prometheus.CounterOpts{
		Namespace: "vsgw_coordinator",
		Subsystem: "shard_client",
		Name:      "requests_total",
		Help:      "Calls the coordinator made to a shard, by shard index, method, and result.",
	}, []string{"shard", "method", "result"})

	shardCallDuration = prometheus.NewHistogramVec(prometheus.HistogramOpts{
		Namespace: "vsgw_coordinator",
		Subsystem: "shard_client",
		Name:      "request_duration_seconds",
		Help:      "Latency of a coordinator -> shard call, by shard index and method.",
		Buckets:   prometheus.DefBuckets,
	}, []string{"shard", "method"})

	shardsQueriedTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "vsgw_coordinator",
		Subsystem: "search",
		Name:      "shards_queried_total",
		Help:      "Sum of shards_queried across every Search, i.e. total shard-queries issued.",
	})

	shardsFailedTotal = prometheus.NewCounter(prometheus.CounterOpts{
		Namespace: "vsgw_coordinator",
		Subsystem: "search",
		Name:      "shards_failed_total",
		Help:      "Sum of shards_failed across every Search. Compared against shards_queried_total, this is the system's real degraded-coverage rate.",
	})
)

func init() {
	prometheus.MustRegister(shardCallsTotal, shardCallDuration, shardsQueriedTotal, shardsFailedTotal)
}

// recordShardCall is called around every coordinator -> shard RPC (Insert,
// Delete, Search) so per-shard health is visible independently of overall
// coordinator request success -- a single hot/failing shard should be
// diagnosable from this metric alone, without correlating gateway-level
// error rates back to a shard index by hand.
func recordShardCall(shard int, method string, start time.Time, err error) {
	result := "success"
	if err != nil {
		result = "error"
	}
	label := strconv.Itoa(shard)
	shardCallsTotal.WithLabelValues(label, method, result).Inc()
	shardCallDuration.WithLabelValues(label, method).Observe(time.Since(start).Seconds())
}
