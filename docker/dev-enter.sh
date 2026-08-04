#!/usr/bin/env bash
# 重建镜像 → 强制重建常驻容器 → 进入容器 shell（防遗忘 --force-recreate）。
# 用法：
#   ./docker/dev-enter.sh
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"

docker compose -f docker/compose.yaml build
docker compose -f docker/compose.yaml up -d --force-recreate pi-dev
# 按服务名进入（不写死容器名 docker-pi-dev-1，项目名变化也不受影响）
docker compose -f docker/compose.yaml exec pi-dev bash
