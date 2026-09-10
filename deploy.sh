#!/usr/bin/env bash
# Single-command build/deploy/teardown for the standalone VectorSearchEngine
# cluster (4 shards + 1 coordinator). Bash equivalent of deploy.ps1 -- see
# that file's header comment for the full explanation. Usage:
#   ./deploy.sh              build + start + wait for health, print address
#   ./deploy.sh down         stop and remove all containers
#   ./deploy.sh down -v      also remove shard data volumes
#   ./deploy.sh --logs       after a successful deploy, stream logs
#
# This repo has no monitoring stack of its own -- for metrics/dashboards,
# run the combined stack from the sibling vectorsearch-gateway repo instead
# (its docker-compose.yml builds this repo's images and wires them into its
# own Prometheus/Grafana). Use this script for the vector engine standalone.
set -euo pipefail

TIMEOUT="${DEPLOY_TIMEOUT:-300}"
FOLLOW_LOGS=false

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker is not on PATH. Install Docker and ensure it's running." >&2
    exit 1
fi

if [[ "${1:-}" == "down" ]]; then
    if [[ "${2:-}" == "-v" ]]; then
        read -r -p "This will delete every shard's snapshot + WAL data (shard0-data..shard3-data). Continue? [y/N] " confirm
        [[ "$confirm" == "y" ]] || { echo "Aborted."; exit 0; }
        docker compose down --volumes
    else
        docker compose down
    fi
    exit $?
fi

for arg in "$@"; do
    [[ "$arg" == "--logs" ]] && FOLLOW_LOGS=true
done

echo "==> Building images (C++ core via CMake/Ninja, then Go via cgo)..."
docker compose build

echo "==> Starting shard0-3 + coordinator and waiting for health checks (timeout: ${TIMEOUT}s)..."
if ! docker compose up -d --wait --wait-timeout "$TIMEOUT"; then
    echo
    echo "One or more services did not become healthy in time." >&2
    echo "Check status with:  docker compose ps" >&2
    echo "Check logs with:    docker compose logs <service>" >&2
    exit 1
fi

echo
echo "==> Cluster is up and healthy."
echo
echo "  coordinator (gRPC)  localhost:8000"
echo "  shards 0-3          internal-only (default network) -- see docker compose logs shard0..shard3"
echo
echo "  No metrics scraping here -- this compose file has no Prometheus of"
echo "  its own. Run vectorsearch-gateway's ./deploy.sh for the combined"
echo "  stack with dashboards/alerts (it builds this repo's images too)."
echo
echo "  Test with grpcurl:  grpcurl -plaintext localhost:8000 list"
echo "  Tear down:          ./deploy.sh down"
echo

$FOLLOW_LOGS && docker compose logs -f

exit 0
