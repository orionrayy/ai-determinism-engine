$ErrorActionPreference = "Stop"

$node = Get-Command node -ErrorAction SilentlyContinue
if (-not $node) {
    Write-Error "Node.js 20+ is required. Install Node.js and retry."
    exit 2
}

$nodeMajor = [int](& node -p "Number(process.versions.node.split('.')[0])")
if ($nodeMajor -lt 20) {
    Write-Error "Node.js 20+ is required; found $(& node --version)."
    exit 2
}

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$config = Join-Path $scriptDir "mcp.config.json"

Push-Location $scriptDir
try {
    & npx --yes @playwright/mcp@0.0.86 --config $config
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
