param(
    [int[]]$Seeds = @(123, 42, 7, 1004, 99),
    [int]$Epochs = 15,
    [int]$BatchSize = 16,
    [int]$NumWorkers = 4,
    [int]$BootstrapReplicates = 2000,
    [string]$PythonExe = "python",
    [string]$ProjectRoot = (Resolve-Path "$PSScriptRoot\..").Path,
    [string]$RgbCheckpoint = "",
    [switch]$SkipTrain,
    [switch]$SkipMetrics
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [string]$Label,
        [string[]]$Arguments
    )
    Write-Host ""
    Write-Host "==== $Label ===="
    Push-Location $ProjectRoot
    try {
        & $PythonExe @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Command failed with exit code ${LASTEXITCODE}: $Label"
        }
    }
    finally {
        Pop-Location
    }
}

foreach ($Seed in $Seeds) {
    $OutputDir = Join-Path $ProjectRoot "checkpoints\seed_sweep\graph_film_refine_seed$Seed"
    $Checkpoint = Join-Path $OutputDir "best.pt"
    $MetricsXlsx = Join-Path $OutputDir "test_threshold_metrics.xlsx"

    Write-Host ""
    Write-Host "############################################################"
    Write-Host "Seed $Seed"
    Write-Host "############################################################"

    if (-not $SkipTrain) {
        $TrainArgs = @(
            "-m", "graph_film_group_refine.train_rgs_graph_film_refine",
            "--seed", "$Seed",
            "--epochs", "$Epochs",
            "--batch-size", "$BatchSize",
            "--num-workers", "$NumWorkers",
            "--output-dir", $OutputDir,
            "--checkpoint", $Checkpoint
        )
        if ($RgbCheckpoint) {
            $TrainArgs += @("--rgb-checkpoint", $RgbCheckpoint, "--freeze-rgb-baseline")
        }
        Invoke-Step -Label "Train graph_film_refine seed=$Seed" -Arguments $TrainArgs
    }

    if (-not $SkipMetrics) {
        Invoke-Step `
            -Label "Export test metrics seed=$Seed" `
            -Arguments @(
                "-m", "graph_film_group_refine.export_test_threshold_metrics",
                "--checkpoint", $Checkpoint,
                "--output-xlsx", $MetricsXlsx,
                "--bootstrap-seed", "$Seed",
                "--bootstrap-replicates", "$BootstrapReplicates",
                "--num-workers", "0"
            )
    }
}

Write-Host ""
Write-Host "Done."
