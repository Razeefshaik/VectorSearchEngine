#!/usr/bin/env bash
# Regenerates Go stubs from proto/*.proto.
#
# One-time setup:
#   # protoc itself: https://github.com/protocolbuffers/protobuf/releases
#   # (on Windows, easiest is `choco install protoc` or `scoop install protobuf`)
#   go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
#   go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest
#   # ensure $(go env GOPATH)/bin is on PATH
#
# Then, from the repo root:
#   ./scripts/gen_proto.sh

set -euo pipefail

OUT=go/proto/shardpb
mkdir -p "$OUT"

protoc \
  --proto_path=proto \
  --go_out="$OUT" --go_opt=paths=source_relative \
  --go-grpc_out="$OUT" --go-grpc_opt=paths=source_relative \
  proto/shard.proto

echo "generated:"
find "$OUT" -name '*.pb.go' | sort
