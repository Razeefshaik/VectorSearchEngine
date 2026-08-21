# Multi-stage build: compile the C++ HNSW core, then the Go binaries that
# link against it via cgo, then ship a slim runtime image with just the
# binaries + the shared library they need.

FROM ubuntu:24.04 AS cpp-build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake ninja-build \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt ./
COPY include/ include/
COPY src/ src/
RUN cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && cmake --build build -j

FROM golang:1.25-bookworm AS go-build
# cgo needs a C++ toolchain even though it's only linking, not compiling C++.
RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY --from=cpp-build /src/build/libhnsw.so /src/build/libhnsw.so
COPY --from=cpp-build /src/include/ /src/include/
COPY go/go.mod go/go.sum ./go/
WORKDIR /src/go
RUN go mod download
WORKDIR /src
COPY go/ go/
WORKDIR /src/go
ENV CGO_ENABLED=1
RUN go build -o /out/shardd ./cmd/shardd \
    && go build -o /out/coordinatord ./cmd/coordinatord

FROM ubuntu:24.04 AS runtime
# netcat is only here for docker-compose's healthcheck (nc -z localhost PORT);
# these binaries have no HTTP endpoint to curl.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates netcat-openbsd \
    && rm -rf /var/lib/apt/lists/*
# hnsw.go's cgo LDFLAGS bake in -Wl,-rpath,${SRCDIR}/../../build (an absolute
# build-time path, /src/build) -- placing the .so there directly relies on
# that path surviving unchanged into the runtime image, which is fragile.
# Install it as a normal system library instead: LD_LIBRARY_PATH is checked
# by the dynamic linker before DT_RUNPATH, so this works regardless of
# exactly how the rpath got encoded at link time, and ldconfig covers any
# tool that ignores LD_LIBRARY_PATH.
COPY --from=cpp-build /src/build/libhnsw.so /usr/local/lib/libhnsw.so
RUN ldconfig
ENV LD_LIBRARY_PATH=/usr/local/lib
COPY --from=go-build /out/shardd /out/coordinatord /usr/local/bin/
