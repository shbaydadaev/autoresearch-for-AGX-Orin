#!/usr/bin/env bash
set -euo pipefail

docker build -f Dockerfile.jetson -t autoresearch:jetson-r35 .
