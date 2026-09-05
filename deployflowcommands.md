# Docker cluster test flow

Full command sequence to test the containerized cluster end to end. Run
everything from the repo root, in PowerShell:

```powershell
cd "s:\StudyResource\TechBoooo\Backend\B projs\VectorSearchEngine"
```

## 0. Build and start the cluster

```powershell
docker compose up --build
```

Wait for all 4 shards to report `Healthy` and the coordinator to log
`"coordinator listening"`.

## 1. Confirm the coordinator answers (reflection works)

```powershell
.\bin\grpcurl.exe -plaintext localhost:8000 list
```

## 2. Insert a vector

```powershell
$rng = [System.Random]::new(1)
$vec = 1..128 | ForEach-Object { [math]::Round($rng.NextDouble() * 2 - 1, 4) }
$insertBody = @{ key = @{ client_id = 1; label = 42 }; vector = $vec } | ConvertTo-Json -Compress
$insertBody | .\bin\grpcurl.exe -plaintext -d "@" localhost:8000 vectorsearch.coordinator.v1.VectorSearch/Insert
```

Expect: `{}`

## 3. Search for it

```powershell
$searchBody = @{ query = $vec; k = 5; ef = 50 } | ConvertTo-Json -Compress
$searchBody | .\bin\grpcurl.exe -plaintext -d "@" localhost:8000 vectorsearch.coordinator.v1.VectorSearch/Search
```

Expect: your vector back (`clientId: 1, label: 42`, distance ≈ 0),
`shardsQueried: 4`.

## 4. Routing check -- which shard actually got the write

```powershell
docker compose exec shard0 ls -la /data
docker compose exec shard1 ls -la /data
docker compose exec shard2 ls -la /data
docker compose exec shard3 ls -la /data
```

Whichever shard's `index.wal` is bigger than the ~8-byte empty header got
the insert; the other three should be untouched.

## 5. Kill a shard, test `allow_partial`

Use whichever shard number showed the WAL growth in step 4 -- example below
uses `shard2`, swap in the real one:

```powershell
docker compose stop shard2

# default (allow_partial omitted) -- should fail
$searchBody | .\bin\grpcurl.exe -plaintext -d "@" localhost:8000 vectorsearch.coordinator.v1.VectorSearch/Search

# allow_partial=true -- should succeed with shardsFailed: 1
$searchBodyPartial = @{ query = $vec; k = 5; ef = 50; allow_partial = $true } | ConvertTo-Json -Compress
$searchBodyPartial | .\bin\grpcurl.exe -plaintext -d "@" localhost:8000 vectorsearch.coordinator.v1.VectorSearch/Search
```

## 6. Crash recovery -- bring the shard back, confirm data survived

```powershell
docker compose start shard2
Start-Sleep -Seconds 3

$searchBody | .\bin\grpcurl.exe -plaintext -d "@" localhost:8000 vectorsearch.coordinator.v1.VectorSearch/Search
```

Expect the original result back again, `shardsQueried: 4`, no failures.

## 7. Watch logs (optional -- run in a second terminal for a live view)

```powershell
docker compose logs -f
```

## 8. Shut down

```powershell
docker compose down          # stop everything, keep data
docker compose down -v       # stop everything, wipe data volumes too
```
