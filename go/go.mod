// Placeholder module path -- rename to match your actual repo, e.g.
// "github.com/yourname/hnsw-vectordb", and update the two import lines in
// durable/durable.go to match. `go mod edit -module <new-path>` does the
// rename in this file for you; the durable.go imports need a manual edit.
module hnswdb

go 1.25.0

require (
	google.golang.org/grpc v1.83.0
	google.golang.org/protobuf v1.36.12
)

require (
	golang.org/x/net v0.55.0 // indirect
	golang.org/x/sys v0.45.0 // indirect
	golang.org/x/text v0.37.0 // indirect
	google.golang.org/genproto/googleapis/rpc v0.0.0-20260526163538-3dc84a4a5aaa // indirect
)
