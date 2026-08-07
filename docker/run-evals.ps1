# 在隔离容器中运行 pi-evals（Windows PowerShell）。
# 每次先重建镜像，确保 src 改动进入容器，再执行评测。
# 用法：
#   .\docker\run-evals.ps1 --provider deepseek --model deepseek-v4-flash
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$scratch = Join-Path $root "work\temp"
New-Item -ItemType Directory -Force -Path $scratch | Out-Null

Push-Location $root
try {
    docker compose -f docker/compose.yaml build pi
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
    docker compose -f docker/compose.yaml run --rm --entrypoint pi-evals pi `
        "--artifact-dir" "/workspace/.eval" @args
    $code = $LASTEXITCODE
    exit $code
}
finally {
    Pop-Location
}
