# stop_local_cluster.ps1 -- stops everything start_local_cluster.ps1 started.
#
# Usage:
#   .\scripts\stop_local_cluster.ps1            # graceful (final snapshot runs)
#   .\scripts\stop_local_cluster.ps1 -Force     # hard kill, for testing crash
#                                                # recovery -- see DEPLOYMENT.md
#                                                # section 4.4

param(
    [switch]$Force
)

$root = Split-Path -Parent $PSScriptRoot
$pidFile = Join-Path $root "logs\pids.txt"

if (-not (Test-Path $pidFile)) {
    Write-Warning "no logs\pids.txt found -- nothing to stop (or it was already cleaned up)"
    exit 0
}

$processIds = Get-Content $pidFile | Where-Object { $_.Trim() -ne "" }

foreach ($processId in $processIds) {
    $proc = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if (-not $proc) {
        continue  # already exited
    }

    if ($Force) {
        Write-Host "force-killing PID $processId ($($proc.ProcessName))"
        Stop-Process -Id $processId -Force
    } else {
        # CTRL+C via taskkill (no /F) lets the process's own signal handler
        # run -- shardd's graceful path takes a final snapshot before exit
        # (see cmd/shardd/main.go). GenerateConsoleCtrlEvent would be more
        # correct but requires attaching a console; taskkill without /F sends
        # WM_CLOSE, which Go's signal.Notify(syscall.SIGTERM) on Windows
        # generally does NOT receive the same way it would on POSIX -- if
        # graceful shutdown doesn't seem to be taking a final snapshot, that's
        # a real Windows signal-handling gap worth raising, not something to
        # silently work around here.
        Write-Host "stopping PID $processId ($($proc.ProcessName))"
        taskkill /PID $processId | Out-Null
    }
}

Start-Sleep -Seconds 1
Remove-Item $pidFile -Force
Write-Host "done"
