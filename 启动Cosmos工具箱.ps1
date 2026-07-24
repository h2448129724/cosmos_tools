param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ToolboxArgs
)

$ErrorActionPreference = 'Stop'
$ToolboxRoot = $PSScriptRoot
$CosmosRoot = (Resolve-Path (Join-Path $ToolboxRoot '..\..')).Path
$env:COSMOS_ROOT = $CosmosRoot
$env:COSMOS_TOOLBOX_ROOT = $ToolboxRoot
$PythonPathEntries = @(
    $ToolboxRoot,
    (Join-Path $ToolboxRoot 'modules'),
    (Join-Path $ToolboxRoot 'shared\cabf_common'),
    $CosmosRoot,
    $env:PYTHONPATH
) | Where-Object { $_ }
$env:PYTHONPATH = $PythonPathEntries -join ';'

$Python = Join-Path $env:LOCALAPPDATA 'miniconda3\envs\onnx-gpu\python.exe'
if (-not (Test-Path $Python)) {
    $Python = Join-Path $env:USERPROFILE 'miniconda3\envs\onnx-gpu\python.exe'
}
if (-not (Test-Path $Python)) {
    throw "找不到 conda 环境 onnx-gpu 的 python.exe"
}

Push-Location $ToolboxRoot
try {
    & $Python -m cosmos_toolbox @ToolboxArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
