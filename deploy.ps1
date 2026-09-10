<#
.SYNOPSIS
    Single-command build/deploy/teardown for the standalone VectorSearchEngine
    cluster (4 shards + 1 coordinator).

.DESCRIPTION
    Wraps `docker compose` so there is exactly one command to build and start
    the cluster and confirm it's actually healthy before handing control
    back -- not just "containers started", but "healthchecks passed" (via
    `docker compose up --wait`, which blocks on each service's HEALTHCHECK).

    This repo has no .env and no monitoring stack of its own -- when it's
    used standalone (this script), you get the coordinator's gRPC endpoint
    and nothing else. For metrics/dashboards/alerts, run it as part of the
    combined stack instead: the sibling vectorsearch-gateway repo's
    `docker-compose.yml` builds this repo's coordinator/shard images and
    wires them into its own Prometheus/Grafana -- see that repo's
    docs/MONITORING.md. Use *this* script when you just want the vector
    engine on its own (e.g. hands-on testing, or a client that isn't
    vectorsearch-gateway).

.PARAMETER Down
    Stop and remove all containers (add -Volumes to also drop each shard's
    data volume).

.PARAMETER Volumes
    Used with -Down: also remove shard0-data..shard3-data. Destructive --
    confirms before running.

.PARAMETER Logs
    After a successful deploy, stream logs from every service.

.PARAMETER Timeout
    Seconds to wait for all services to report healthy. Default 300 -- the
    image build compiles the C++ HNSW core via CMake/Ninja and cross-links
    the Go binaries via cgo, which can take a few minutes on a cold cache.

.EXAMPLE
    ./deploy.ps1
    Build and start shard0-3 + coordinator, wait for health, print the
    coordinator's address.

.EXAMPLE
    ./deploy.ps1 -Down
    Tear everything down (containers only, shard data kept).

.EXAMPLE
    ./deploy.ps1 -Down -Volumes
    Tear everything down including each shard's snapshot + WAL data.
#>
param(
    [switch]$Down,
    [switch]$Volumes,
    [switch]$Logs,
    [int]$Timeout = 300
)

$ErrorActionPreference = "Stop"

function Assert-CommandExists($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        Write-Host "ERROR: '$name' is not on PATH. Install Docker Desktop and ensure it's running." -ForegroundColor Red
        exit 1
    }
}

Assert-CommandExists "docker"

if ($Down) {
    if ($Volumes) {
        Write-Host "This will delete every shard's snapshot + WAL data (shard0-data..shard3-data). Continue? [y/N]" -ForegroundColor Yellow
        $confirm = Read-Host
        if ($confirm -ne "y") { Write-Host "Aborted."; exit 0 }
        docker compose down --volumes
    } else {
        docker compose down
    }
    exit $LASTEXITCODE
}

Write-Host "==> Building images (C++ core via CMake/Ninja, then Go via cgo)..." -ForegroundColor Cyan
docker compose build
if ($LASTEXITCODE -ne 0) { Write-Host "Build failed." -ForegroundColor Red; exit $LASTEXITCODE }

Write-Host "==> Starting shard0-3 + coordinator and waiting for health checks (timeout: ${Timeout}s)..." -ForegroundColor Cyan
docker compose up -d --wait --wait-timeout $Timeout
$upResult = $LASTEXITCODE

if ($upResult -ne 0) {
    Write-Host ""
    Write-Host "One or more services did not become healthy in time." -ForegroundColor Red
    Write-Host "Check status with:  docker compose ps" -ForegroundColor Yellow
    Write-Host "Check logs with:    docker compose logs <service>" -ForegroundColor Yellow
    exit $upResult
}

Write-Host ""
Write-Host "==> Cluster is up and healthy." -ForegroundColor Green
Write-Host ""
Write-Host "  coordinator (gRPC)  localhost:8000"
Write-Host "  shards 0-3          internal-only (vsgw/default network) -- see docker compose logs shard0..shard3"
Write-Host ""
Write-Host "  No metrics scraping here -- this compose file has no Prometheus of" -ForegroundColor Yellow
Write-Host "  its own. Run vectorsearch-gateway's ./deploy.ps1 for the combined" -ForegroundColor Yellow
Write-Host "  stack with dashboards/alerts (it builds this repo's images too)." -ForegroundColor Yellow
Write-Host ""
Write-Host "  Test with grpcurl:  grpcurl -plaintext localhost:8000 list"
Write-Host "  Tear down:          ./deploy.ps1 -Down"
Write-Host ""

if ($Logs) {
    docker compose logs -f
}
