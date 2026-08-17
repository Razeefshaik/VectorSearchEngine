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

# paths=source_relative mirrors the SOURCE filename under --go_out, ignoring
# go_package's directory component -- so each proto needs its own --go_out
# matching its own go_package, not one shared directory for both.

SHARD_OUT=go/proto/shardpb
mkdir -p "$SHARD_OUT"
protoc \
  --proto_path=proto \
  --go_out="$SHARD_OUT" --go_opt=paths=source_relative \
  --go-grpc_out="$SHARD_OUT" --go-grpc_opt=paths=source_relative \
  proto/shard.proto

COORDINATOR_OUT=go/proto/coordinatorpb
mkdir -p "$COORDINATOR_OUT"
protoc \
  --proto_path=proto \
  --go_out="$COORDINATOR_OUT" --go_opt=paths=source_relative \
  --go-grpc_out="$COORDINATOR_OUT" --go-grpc_opt=paths=source_relative \
  proto/coordinator.proto

echo "generated:"
find "$SHARD_OUT" "$COORDINATOR_OUT" -name '*.pb.go' | sort
