# Frozen structure + 512 semantic protected contours, full-data 40+40 schedule.
param(
    [string]$ThermalRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$RunName = "flir_v2_struct_sem512_protected_seed42_cuda_20261013_40_40",
    [ValidateRange(0, 80)][int]$StopAfterEpoch = 5
)

$ErrorActionPreference = "Stop"
$ThermalRoot = (Resolve-Path -LiteralPath $ThermalRoot).Path
$dataRoot = Join-Path $ThermalRoot "FLIR_datasets"
$trainRoot = Join-Path $ThermalRoot "pytorch-CycleGAN-and-pix2pix"
$checkpoints = Join-Path $ThermalRoot "runs\checkpoints"
$results = Join-Path $ThermalRoot "runs\results"
$semantic = Join-Path $ThermalRoot "runs\pretrained\segformer_fmb_b3\best_model.pth"
if (-not (Test-Path -LiteralPath $semantic -PathType Leaf)) { throw "Copy the complete frozen semantic weight here: $semantic" }
if (Test-Path -LiteralPath (Join-Path $checkpoints $RunName)) { throw "Experiment already exists. Choose a new -RunName; this launcher starts from scratch." }
if (Test-Path -LiteralPath (Join-Path $results $RunName)) { throw "Result directory already exists. Choose a new -RunName." }

python -c "import torch,numpy,PIL,scipy,skimage,omegaconf; print('PyTorch:',torch.__version__); print('CUDA:',torch.cuda.is_available()); print('GPU:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); raise SystemExit(0 if torch.cuda.is_available() and hasattr(torch.serialization,'safe_globals') else 1)"
if ($LASTEXITCODE -ne 0) { throw "Use a CUDA PyTorch environment supporting safe_globals and install scripts\requirements_semantic_preview.txt." }
$expectedCounts = @{ trainA = 6191; trainB = 2110; testA = 696; testB = 112 }
foreach ($split in @("trainA", "trainB", "testA", "testB")) {
    $directory = Join-Path $dataRoot $split
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) { throw "Missing dataset: $directory" }
    $count = @(Get-ChildItem -LiteralPath $directory -File | Where-Object { $_.Extension.ToLower() -in @(".jpg", ".jpeg", ".png") }).Count
    if ($count -ne $expectedCounts[$split]) { throw "$split has $count files; expected $($expectedCounts[$split]). Use the same processed FLIR data." }
}

$common = @(
    "--dataroot", $dataRoot, "--name", $RunName, "--checkpoints_dir", $checkpoints,
    "--model", "semantic_contour_cycle_gan", "--semantic_checkpoint", $semantic,
    "--condition_mode", "soft_saliency", "--fusion_mode", "direct",
    "--dataset_mode", "unaligned", "--direction", "BtoA", "--input_nc", "1", "--output_nc", "3"
)
$train = @(
    "--lambda_identity", "0", "--preprocess", "crop", "--crop_size", "256",
    "--batch_size", "1", "--num_threads", "4", "--seed", "42",
    "--n_epochs", "40", "--n_epochs_decay", "40",
    "--print_freq", "100", "--display_freq", "400", "--update_html_freq", "400",
    "--save_latest_freq", "5000", "--save_epoch_freq", "5"
)
if ($StopAfterEpoch -gt 0) { $train += @("--stop_after_epoch", "$StopAfterEpoch") }
$lastEpoch = if ($StopAfterEpoch -gt 0) { $StopAfterEpoch } else { 80 }
Push-Location $trainRoot
try {
    python -u train.py @common @train
    if ($LASTEXITCODE -ne 0) { throw "Protected-contour training failed; test export has not started." }
    python -u test.py @common `
        --results_dir $results --epoch "$lastEpoch" --num_test 112 `
        --preprocess none --no_flip --eval
    if ($LASTEXITCODE -ne 0) { throw "Test export failed; training checkpoints are preserved." }
} finally {
    Pop-Location
}
Write-Host "Completed. Second channel: structure + semantic protected contour; third: zero."
Write-Host "Training preview: $(Join-Path $checkpoints "$RunName\web\index.html")"
Write-Host "Fixed 112 testB results: $(Join-Path $results "$RunName\test_$lastEpoch\index.html")"
