param(
    [string]$PythonExe = "python",
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path,
    [Parameter(Mandatory=$true)]
    [string]$Checkpoint,
    [string]$OutputDir = "",
    [int]$BatchSize = 32,
    [int]$NumWorkers = 0,
    [switch]$SaveCrops
)

$ErrorActionPreference = "Stop"

if (-not $OutputDir) {
    $OutputDir = Join-Path $ProjectRoot "outputs\external_validation_sensitivity95"
}

$ArgsList = @(
    "-m", "graph_film_group_refine.external_validate_graph_film",
    "--checkpoint", $Checkpoint,
    "--datasets", "origa,g1020,refuge",
    "--merge-origa-splits",
    "--merge-refuge-train-val",
    "--threshold-modes", "sensitivity_0.95",
    "--batch-size", "$BatchSize",
    "--num-workers", "$NumWorkers",
    "--output-dir", $OutputDir
)
if ($SaveCrops) {
    $ArgsList += "--save-crops"
}

Write-Host "Zero-shot external validation with fixed internal sensitivity_0.95 threshold"
Write-Host "Checkpoint: $Checkpoint"
Write-Host "OutputDir:  $OutputDir"
Push-Location $ProjectRoot
try {
    & $PythonExe @ArgsList
    if ($LASTEXITCODE -ne 0) {
        throw "External validation failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
