param(
    [ValidateSet("audiomae", "hubert", "ast")]
    [string]$Model = "audiomae",

    [int]$Epochs = 5,
    [int]$BatchSize = 4,
    [double]$LearningRate = 5e-5,
    [string]$DatasetPath = "./data/raw/datasets/archive/Respiratory_Sound_Database/Respiratory_Sound_Database",
    [int]$EarlyStoppingPatience = 2,
    [int]$SchedulerPatience = 1,
    [double]$SchedulerFactor = 0.5
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

if ($Model -ne "audiomae") {
    Write-Host "Model '$Model' is reserved for the next heavy-model integrations." -ForegroundColor Yellow
    Write-Host "Current codebase already supports only: audiomae" -ForegroundColor Yellow
    exit 1
}

$CheckpointPath = "results/checkpoints/icbhi_${Model}_best.pth"

python scripts/train_icbhi_baseline.py `
    --model-name $Model `
    --dataset-path $DatasetPath `
    --epochs $Epochs `
    --batch-size $BatchSize `
    --learning-rate $LearningRate `
    --checkpoint-path $CheckpointPath `
    --early-stopping-patience $EarlyStoppingPatience `
    --scheduler-patience $SchedulerPatience `
    --scheduler-factor $SchedulerFactor

