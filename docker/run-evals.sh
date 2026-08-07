#!/usr/bin/env bash
# 在隔离容器中运行 pi-evals（Linux/macOS）。
# 每次先重建镜像，确保 src 改动进入容器，再执行评测。
# 用法：
#   ./docker/run-evals.sh --provider deepseek --model deepseek-v4-flash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$root/work/temp"
mkdir -p "$scratch"

cd "$root"
docker compose -f docker/compose.yaml build pi
docker compose -f docker/compose.yaml run --rm --entrypoint pi-evals pi \
    "--artifact-dir" "/workspace/.eval" "$@"
