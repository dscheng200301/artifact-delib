$ErrorActionPreference = "Stop"

$RootDir = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$EnvFile = Join-Path $RootDir ".env"

if (-not (Test-Path -LiteralPath $EnvFile)) {
    Write-Error "Missing $EnvFile. Copy .env.example to .env and fill VLM_API_KEY."
}

function Get-DotEnvValue([string]$Name) {
    $line = Get-Content -LiteralPath $EnvFile | Where-Object {
        $_ -match ("^" + [regex]::Escape($Name) + "=")
    } | Select-Object -Last 1
    if ($null -eq $line) { return "" }
    return (($line -split "=", 2)[1]).Trim().Trim([char]13)
}

$VlmApiKey = Get-DotEnvValue "VLM_API_KEY"
$LlmApiKey = Get-DotEnvValue "LLM_API_KEY"
$PaidCalls = Get-DotEnvValue "API_ALLOW_PAID_CALLS"

if ([string]::IsNullOrWhiteSpace($VlmApiKey) -and [string]::IsNullOrWhiteSpace($LlmApiKey)) {
    Write-Error "VLM_API_KEY or LLM_API_KEY is required in .env."
}
if ($PaidCalls -ne "true") {
    Write-Error "Set API_ALLOW_PAID_CALLS=true in .env before this paid smoke test."
}

Set-Location -LiteralPath $RootDir
$env:PYTHONPATH = "$RootDir\src" + $(if ($env:PYTHONPATH) { ";$env:PYTHONPATH" } else { "" })

conda run --no-capture-output -n histo-delib `
    python -m histodelib.cli run `
    --mode api `
    --method direct_vlm `
    --config configs/api/default.yaml `
    --output-root outputs
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
