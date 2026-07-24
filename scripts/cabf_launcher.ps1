param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ArgsList
)

$ErrorActionPreference = "Stop"

$entry = Join-Path $PSScriptRoot "cabf_launcher.py"

function Resolve-CabfPython {
    if ($env:CABF_PYTHON) {
        return @{
            Command = $env:CABF_PYTHON
            Prefix = @()
            Display = $env:CABF_PYTHON
            Install = "$env:CABF_PYTHON -m pip install PySide6"
        }
    }

    $envName = if ($env:CABF_CONDA_ENV) { $env:CABF_CONDA_ENV } else { "onnx-gpu" }
    $condaRoots = @(
        $env:CABF_CONDA_ROOT,
        (Join-Path $env:USERPROFILE "miniconda3"),
        (Join-Path $env:USERPROFILE "anaconda3"),
        (Join-Path $env:LOCALAPPDATA "miniconda3"),
        (Join-Path $env:LOCALAPPDATA "anaconda3"),
        "C:\ProgramData\miniconda3",
        "C:\ProgramData\anaconda3"
    ) | Where-Object { $_ }
    foreach ($root in $condaRoots) {
        $pythonExe = Join-Path $root "envs\$envName\python.exe"
        if (Test-Path $pythonExe) {
            return @{
                Command = $pythonExe
                Prefix = @()
                Display = $pythonExe
                Install = "$pythonExe -m pip install PySide6"
            }
        }
    }

    if (Get-Command conda -ErrorAction SilentlyContinue) {
        try {
            $condaInfo = conda info --envs --json | ConvertFrom-Json
            foreach ($envPath in $condaInfo.envs) {
                if ((Split-Path $envPath -Leaf) -eq $envName) {
                    $pythonExe = Join-Path $envPath "python.exe"
                    if (Test-Path $pythonExe) {
                        return @{
                            Command = $pythonExe
                            Prefix = @()
                            Display = $pythonExe
                            Install = "$pythonExe -m pip install PySide6"
                        }
                    }
                }
            }
        } catch {
            # Fall back to conda run below.
        }
        return @{
            Command = "conda"
            Prefix = @("run", "-n", $envName, "python")
            Display = "conda run -n $envName python"
            Install = "conda run -n $envName python -m pip install PySide6"
        }
    }

    return @{
        Command = "python"
        Prefix = @()
        Display = "python"
        Install = "python -m pip install PySide6"
    }
}

function Invoke-CabfPython {
    param([string[]]$PythonArgs)
    & $script:PythonInfo.Command @($script:PythonInfo.Prefix + $PythonArgs)
}

$script:PythonInfo = Resolve-CabfPython

try {
    $checkArgs = @("-c", "import sys; import PySide6; print(sys.executable)")
    Invoke-CabfPython $checkArgs | Out-Host
} catch {
    Write-Host ""
    Write-Host "CABF launcher failed: PySide6 is missing in the selected Python environment." -ForegroundColor Red
    Write-Host "Selected Python: $($script:PythonInfo.Display)"
    Write-Host ""
    Write-Host "Install with:" -ForegroundColor Yellow
    Write-Host "  $($script:PythonInfo.Install)"
    Write-Host ""
    Write-Host "By default this script uses conda env 'onnx-gpu' when conda is available."
    Write-Host "Set CABF_CONDA_ENV to use another conda env, or CABF_PYTHON to use a specific python.exe."
    Write-Host "Examples:"
    Write-Host '  $env:CABF_CONDA_ENV="pytorch"'
    Write-Host '  $env:CABF_PYTHON="C:\path\to\env\python.exe"'
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

try {
    $runArgs = @($entry) + $ArgsList
    Invoke-CabfPython $runArgs
} catch {
    Write-Host ""
    Write-Host "CABF launcher runtime error:" -ForegroundColor Red
    Write-Host $_
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}
