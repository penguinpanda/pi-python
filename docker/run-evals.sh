#!/usr/bin/env bash
# 在隔离容器中运行 pi-evals（Linux/macOS）。
# 用法：
#   ./docker/run-evals.sh --provider deepseek --model deepseek-v4-flash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$root/work/temp"
mkdir -p "$scratch"

cd "$root"
docker compose -f docker/compose.yaml run --rm --entrypoint pi-evals pi \
    "--artifact-dir" "/workspace/.eval" "$@"
