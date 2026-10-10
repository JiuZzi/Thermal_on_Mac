# Run from an activated Windows Python environment with CUDA PyTorch.
# Preserve full-data 40+40 training; stop each variant after epoch 5.
param(
    [string]$ThermalRoot = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
$ThermalRoot = (Resolve-Path -LiteralPath $ThermalRoot).Path
$dataRoot = Join-Path $ThermalRoot "FLIR_datasets"
$checkpointRoot = Join-Path $ThermalRoot "runs\checkpoints"
$trainRoot = Join-Path $ThermalRoot "pytorch-CycleGAN-and-pix2pix"
$plainName = "flir_v2_jr0_c2c_joint_plain_cuda_20261010_40_40"
$contrastiveName = "flir_v2_jr1_c2c_joint_contrastive_cuda_20261010_40_40"
$previewRoot = Join-Path $ThermalRoot "runs\edge_audit\joint_contours_cuda_epoch005_20261010"

python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); raise SystemExit(0 if torch.cuda.is_available() else 1)"
if ($LASTEXITCODE -ne 0) { throw "CUDA unavailable. Use the existing CUDA-enabled Python environment." }

foreach ($split in @("trainA", "trainB")) {
    $splitRoot = Join-Path $dataRoot $split
    if (-not (Test-Path -LiteralPath $splitRoot -PathType Container)) { throw "Missing dataset: $splitRoot" }
    $count = @(Get-ChildItem -LiteralPath $splitRoot -File | Where-Object { $_.Extension.ToLower() -in @(".jpg", ".jpeg", ".png") }).Count
    $expected = if ($split -eq "trainA") { 6191 } else { 2110 }
    if ($count -ne $expected) { throw "$split has $count images; expected $expected. Verify the existing FLIR protocol." }
}
foreach ($name in @($plainName, $contrastiveName)) {
    $directory = Join-Path $checkpointRoot $name
    if (Test-Path -LiteralPath $directory) { throw "Experiment already exists: $directory. Choose new names or use explicit resume commands." }
}
if (Test-Path -LiteralPath $previewRoot) { throw "Preview output already exists: $previewRoot" }

$commonArgs = @(
    "--dataroot", $dataRoot,
    "--checkpoints_dir", $checkpointRoot,
    "--model", "joint_contour_cycle_gan",
    "--condition_mode", "soft_saliency",
    "--fusion_mode", "direct",
    "--dataset_mode", "unaligned",
    "--direction", "BtoA",
    "--input_nc", "1",
    "--output_nc", "3",
    "--lambda_identity", "0",
    "--lambda_contour_anchor", "1.0",
    "--lambda_contour_nce", "0.1",
    "--preprocess", "crop",
    "--crop_size", "256",
    "--batch_size", "1",
    "--num_threads", "4",
    "--seed", "2026",
    "--n_epochs", "40",
    "--n_epochs_decay", "40",
    "--stop_after_epoch", "5",
    "--print_freq", "100",
    "--display_freq", "400",
    "--update_html_freq", "1000",
    "--save_latest_freq", "5000",
    "--save_epoch_freq", "5"
)

Push-Location $trainRoot
try {
    python -u train.py @commonArgs --name $plainName --joint_variant plain
    if ($LASTEXITCODE -ne 0) { throw "JR0 failed; JR1 has not been started." }
    python -u train.py @commonArgs --name $contrastiveName --joint_variant contrastive
    if ($LASTEXITCODE -ne 0) { throw "JR1 failed; inspect its log before proceeding." }
} finally {
    Pop-Location
}

# Full-frame preview does not require the earlier review package.
# Reuse fixed target crops only when their source-image hashes match exactly.
$review = Join-Path $ThermalRoot "runs\edge_audit\important_contour_review_v1\review_manifest.json"
$reviewArgs = @()
if (Test-Path -LiteralPath $review) {
    python -c "import hashlib,json,pathlib,sys; r=json.loads(pathlib.Path(sys.argv[1]).read_text(encoding='utf-8')); d=pathlib.Path(sys.argv[2]); ok=all((d / x['filename']).is_file() and hashlib.sha256((d / x['filename']).read_bytes()).hexdigest()==r['source_hashes'].get(x['filename']) for x in r['selection']['samples']); raise SystemExit(0 if ok else 1)" $review (Join-Path $dataRoot "trainB")
    if ($LASTEXITCODE -eq 0) {
        $reviewArgs = @("--review-manifest", $review)
    } else {
        Write-Warning "Review images differ from the Mac package. Exporting full-frame comparisons without fixed target crops."
    }
}
python -u (Join-Path $ThermalRoot "scripts\preview_joint_contours.py") `
    --plain-checkpoint (Join-Path $checkpointRoot "$plainName\5_net_G_A.pth") `
    --contrastive-checkpoint (Join-Path $checkpointRoot "$contrastiveName\5_net_G_A.pth") `
    --image-dir (Join-Path $dataRoot "trainB") `
    --output-dir $previewRoot `
    --num-images 40 `
    --device cuda `
    @reviewArgs
if ($LASTEXITCODE -ne 0) { throw "Preview export failed; the epoch-5 checkpoints are preserved." }
Write-Host "Both epoch-5 runs and preview export completed: $previewRoot"
