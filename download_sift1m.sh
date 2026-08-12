#!/usr/bin/env bash
# download_sift1m.sh -- fetches the standard SIFT1M ANN benchmark dataset
# (1M 128-dim base vectors, 10K queries, precomputed top-100 ground truth)
# from the original texmex corpus.
#
# Source: http://corpus-texmex.irisa.fr/  (same source used by faiss, hnswlib,
# and ann-benchmarks -- this is the canonical SIFT1M distribution).
#
# Usage: ./download_sift1m.sh   (run from the repo root)

set -euo pipefail

DATA_DIR="data"
URL="ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz"

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

if [ -f sift/sift_base.fvecs ]; then
    echo "already present at $DATA_DIR/sift/ -- skipping download"
    exit 0
fi

echo "downloading SIFT1M (~168 MB)..."
if command -v wget >/dev/null 2>&1; then
    wget -O sift.tar.gz "$URL"
elif command -v curl >/dev/null 2>&1; then
    curl -o sift.tar.gz "$URL"
else
    echo "need wget or curl" >&2
    exit 1
fi

echo "extracting..."
tar -xzf sift.tar.gz
rm sift.tar.gz

echo "done. files:"
ls -la sift/

cat <<'EOF'

If the ftp:// download above failed (some corporate/campus networks block
outbound FTP), alternatives:

  - Kaggle mirror: search "sift1m" on kaggle.com/datasets
  - Hugging Face:  huggingface.co/datasets/maknee/sift1m (different format --
    .fbin, not .fvecs; would need a different reader than sift_bench.cpp)
  - ann-benchmarks HDF5 mirror: http://ann-benchmarks.com/sift-128-euclidean.hdf5
    (also a different format, needs libhdf5 to read)

The .fvecs/.ivecs texmex format is what sift_bench.cpp expects, so the
original ftp source above is the path of least resistance if it works
for you.
EOF
