param([string]$CondaEnvironment = 'onnx-gpu')
$ErrorActionPreference = 'Stop'
$toolboxRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $toolboxRoot
try {
    conda run --no-capture-output -n $CondaEnvironment python -m cosmos_toolbox.field_dataset_ui
    if ($LASTEXITCODE -ne 0) { throw "Dataset tool exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
