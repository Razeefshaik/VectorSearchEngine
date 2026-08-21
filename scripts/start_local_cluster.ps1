# start_local_cluster.ps1 -- launches 4 shardd processes + 1 coordinatord
# locally, for hands-on testing. See DEPLOYMENT.md for the full walkthrough.
#
# Usage:
#   .\scripts\start_local_cluster.ps1                 # dim=8, fast smoke test
#   .\scripts\start_local_cluster.ps1 -Dim 128 -Clean # real dim, fresh data
#
# Assumes bin\shardd.exe and bin\coordinatord.exe are already built (see
# DEPLOYMENT.md section 3) and that the C++ shared library has been built via
# cmake into .\build\.

param(
    [int]$Dim = 8,
    [int]$NumShards = 4,
    [int]$BasePort = 7001,
    [int]$CoordinatorPort = 8000,
    [int]$MaxElements = 50000,
    [switch]$Clean
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$binDir = Join-Path $root "bin"
$dataDir = Join-Path $root "data"
$logDir = Join-Path $root "logs"

# --- locate the built shared library -------------------------------------
# CMake with Ninja (this repo's default) or Visual Studio generators land the
# DLL in different places; check both rather than guessing wrong silently.
$dllCandidates = @(
    (Join-Path $root "build\libhnsw.dll"),
    (Join-Path $root "build\Debug\libhnsw.dll"),
    (Join-Path $root "cmake-build-debug\libhnsw.dll")
)
$dll = $dllCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $dll) {
    Write-Error "libhnsw.dll not found in any of: $($dllCandidates -join ', ')`n" +
                "Build it first: cmake -B build -G Ninja; cmake --build build -j"
    exit 1
}
Write-Host "using shared library: $dll"

# --- fresh directories -----------------------------------------------------
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
if ($Clean -and (Test-Path $dataDir)) {
    Write-Host "removing existing $dataDir (fresh start)"
    Remove-Item -Recurse -Force $dataDir
}
New-Item -ItemType Directory -Force -Path $dataDir | Out-Null

# --- the Windows rpath gotcha -- see DEPLOYMENT.md section 2 --------------
# rpath (used in hnsw.go's cgo LDFLAGS) is a no-op on Windows. Copy the DLL
# next to each executable so the loader finds it.
foreach ($exe in @("shardd.exe", "coordinatord.exe")) {
    $dest = Join-Path $binDir $exe
    if (-not (Test-Path $dest)) {
        Write-Error "$dest not found. Build it first -- see DEPLOYMENT.md section 3."
        exit 1
    }
    Copy-Item $dll (Join-Path $binDir "libhnsw.dll") -Force
}

# --- waits for a TCP port to accept connections, so we don't race the -----
# coordinator's startup health check against a shard that hasn't bound yet.
function Wait-ForPort([string]$HostName, [int]$Port, [int]$TimeoutSec = 15) {
    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        try {
            $client = New-Object System.Net.Sockets.TcpClient
            $client.Connect($HostName, $Port)
            $client.Close()
            return $true
        } catch {
            Start-Sleep -Milliseconds 300
        }
    }
    return $false
}

# --- launch shards -----------------------------------------------------
$pids = @()
$shardAddrs = @()

for ($i = 0; $i -lt $NumShards; $i++) {
    $port = $BasePort + $i
    $shardData = Join-Path $dataDir "shard$i"
    $logFile = Join-Path $logDir "shard$i.log"
    $shardAddrs += "localhost:$port"

    Write-Host "starting shard $i on :$port (data: $shardData)"
    # Start-Process -ArgumentList joins array elements with a plain space and
    # does NOT quote elements containing spaces (a Windows PowerShell 5.1
    # gotcha) -- if this repo lives under a path with a space (e.g. "B projs"),
    # unquoted path args get split and everything after the split point is
    # silently dropped by Go's flag parser (it stops at the first non-flag
    # token). Quote every value defensively so this can't bite again.
    $p = Start-Process -FilePath (Join-Path $binDir "shardd.exe") `
        -ArgumentList @(
            "-listen", "`":$port`"",
            "-data", "`"$shardData`"",
            "-dim", "`"$Dim`"",
            "-max-elements", "`"$MaxElements`"",
            "-snapshot-interval", "`"2m`""
        ) `
        -WorkingDirectory $binDir `
        -RedirectStandardOutput $logFile `
        -RedirectStandardError "$logFile.err" `
        -PassThru -WindowStyle Hidden

    $pids += $p.Id
}

Write-Host "waiting for all shards to accept connections..."
for ($i = 0; $i -lt $NumShards; $i++) {
    $port = $BasePort + $i
    if (-not (Wait-ForPort "localhost" $port)) {
        Write-Error "shard $i did not come up on port $port -- check logs\shard$i.log.err"
        exit 1
    }
    Write-Host "  shard $i ready"
}

# --- launch coordinator -----------------------------------------------
$shardList = $shardAddrs -join ","
$coordLog = Join-Path $logDir "coordinator.log"
Write-Host "starting coordinator on :$CoordinatorPort (shards: $shardList)"
$cp = Start-Process -FilePath (Join-Path $binDir "coordinatord.exe") `
    -ArgumentList @("-listen", "`":$CoordinatorPort`"", "-shards", "`"$shardList`"") `
    -WorkingDirectory $binDir `
    -RedirectStandardOutput $coordLog `
    -RedirectStandardError "$coordLog.err" `
    -PassThru -WindowStyle Hidden
$pids += $cp.Id

if (-not (Wait-ForPort "localhost" $CoordinatorPort)) {
    Write-Error "coordinator did not come up on port $CoordinatorPort -- check logs\coordinator.log.err"
    exit 1
}

$pids | Out-File -FilePath (Join-Path $logDir "pids.txt")

Write-Host ""
Write-Host "cluster is up:"
Write-Host "  coordinator : localhost:$CoordinatorPort"
for ($i = 0; $i -lt $NumShards; $i++) {
    Write-Host "  shard $i     : localhost:$($BasePort + $i)"
}
Write-Host ""
Write-Host "dim=$Dim -- every Insert/Search must use vectors of this length"
Write-Host "logs: $logDir\*.log   pids: $logDir\pids.txt"
Write-Host "stop with: .\scripts\stop_local_cluster.ps1"
