param(
    [ValidateSet("audit", "fit")]
    [string]$Mode = "audit",
    [ValidateSet("deeplab", "whu")]
    [string]$Model = "deeplab",
    [int]$Epochs = 20,
    [string]$Image = "",
    [string]$Labels = "",
    [double]$Threshold = 0.50,
    [switch]$AssumeLabelsComplete
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ScriptPath = Join-Path $ProjectRoot "tools\lalpur_model_lab.py"
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $ScriptPath)) {
    throw "Could not find tools\lalpur_model_lab.py. Extract the kit into the SIH project root."
}
if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $PythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $PythonCommand) { throw "Python was not found. Install Python 3.11 or 3.12, then rerun." }
    $Python = $PythonCommand.Source
}

Push-Location $ProjectRoot
try {
    $PythonArgs = @(
        $ScriptPath,
        "--mode", $Mode,
        "--model", $Model,
        "--epochs", "$Epochs",
        "--threshold", "$Threshold"
    )
    if ($Image) { $PythonArgs += @("--image", $Image) }
    if ($Labels) { $PythonArgs += @("--labels", $Labels) }
    if ($AssumeLabelsComplete) { $PythonArgs += @("--assume-labels-complete") }

    Write-Host "Project root: $ProjectRoot"
    Write-Host "Python: $Python"
    Write-Host "Mode: $Mode | Model: $Model | Epochs: $Epochs"
    & $Python @PythonArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Lalpur model lab failed with exit code $LASTEXITCODE. Read the error above; original baseline files were not changed."
    }
} finally {
    Pop-Location
}
