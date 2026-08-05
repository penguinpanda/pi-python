# 重建镜像 → 强制重建常驻容器 → 进入容器 shell（防遗忘 --force-recreate）。
# 用法：
#   .\docker\dev-enter.ps1
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    docker compose -f docker/compose.yaml build
    if ($LASTEXITCODE -ne 0) {
        Write-Error "docker compose build failed (exit $LASTEXITCODE); keeping the previous container."
        exit $LASTEXITCODE
    }
    docker compose -f docker/compose.yaml up -d --force-recreate pi-dev
    # 按服务名进入（不写死容器名 docker-pi-dev-1，项目名变化也不受影响）
    docker compose -f docker/compose.yaml exec pi-dev bash
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
