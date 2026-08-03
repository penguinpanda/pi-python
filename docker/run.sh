#!/usr/bin/env bash
# 在隔离容器中运行 pi coding-agent（Linux/macOS）。
# 用法：
#   ./docker/run.sh -p "read notes.md and summarize"
#   ./docker/run.sh --mode tui
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scratch="$root/work/temp"
mkdir -p "$scratch"

cd "$root"
docker compose -f docker/compose.yaml run --rm pi "$@"
