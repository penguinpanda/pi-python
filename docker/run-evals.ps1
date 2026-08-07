# 在隔离容器中运行 pi-evals（Windows PowerShell）。
# 用法：
#   .\docker\run-evals.ps1 --provider deepseek --model deepseek-v4-flash
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$scratch = Join-Path $root "work\temp"
New-Item -ItemType Directory -Force -Path $scratch | Out-Null

Push-Location $root
try {
    docker compose -f docker/compose.yaml run --rm --entrypoint pi-evals pi `
        "--artifact-dir" "/workspace/.eval" @args
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
