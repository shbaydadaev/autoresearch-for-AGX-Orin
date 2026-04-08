#!/usr/bin/env bash
set -euo pipefail

docker run --rm -it \
  --runtime nvidia \
  --network host \
  --ipc host \
  -v "$(pwd)":/workspace/autoresearch \
  -v autoresearch-cache:/workspace/.cache/autoresearch \
  autoresearch:jetson-r35 \
  bash
