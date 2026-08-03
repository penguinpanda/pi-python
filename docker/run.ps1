# 在隔离容器中运行 pi coding-agent（Windows PowerShell）。
# 用法：
#   .\docker\run.ps1 -p "read notes.md and summarize"
#   .\docker\run.ps1 --mode tui
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$scratch = Join-Path $root "work\temp"
New-Item -ItemType Directory -Force -Path $scratch | Out-Null

Push-Location $root
try {
    docker compose -f docker/compose.yaml run --rm pi @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
