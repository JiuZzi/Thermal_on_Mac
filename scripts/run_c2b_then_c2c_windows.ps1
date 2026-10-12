[CmdletBinding()]
param(
    [string]$PythonExe = 'D:\anaconda\envs\isnet\python.exe',
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
$thermalRoot = Split-Path -Parent $PSScriptRoot
$projectDir = Join-Path $thermalRoot 'pytorch-CycleGAN-and-pix2pix'
$checkpointDir = Join-Path $thermalRoot 'runs\checkpoints'
$datasetDir = Join-Path $thermalRoot 'FLIR_datasets'

foreach ($path in @($PythonExe, $projectDir, $datasetDir)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required path does not exist: $path"
    }
}

$commonArgs = @(
    '--dataroot', $datasetDir,
    '--checkpoints_dir', $checkpointDir,
    '--model', 'conditioned_cycle_gan',
    '--fusion_mode', 'direct',
    '--edge_sigma', '1.0',
    '--edge_low_threshold', '0.08',
    '--edge_high_threshold', '0.16',
    '--edge_soft_width', '1.0',
    '--soft_high_thresholds', '0.12,0.16,0.20',
    '--dataset_mode', 'unaligned',
    '--direction', 'BtoA',
    '--input_nc', '1',
    '--output_nc', '3',
    '--no_dropout',
    '--lambda_identity', '0',
    '--preprocess', 'crop',
    '--crop_size', '256',
    '--batch_size', '1',
    '--num_threads', '4',
    '--n_epochs', '40',
    '--n_epochs_decay', '40',
    '--seed', '42',
    '--print_freq', '100',
    '--display_freq', '400',
    '--update_html_freq', '1000',
    '--save_latest_freq', '5000',
    '--save_epoch_freq', '5'
)

$stages = @(
    [PSCustomObject]@{
        Name = 'flir_v2_c2b_soft_multi_40_40_seed42_win'
        Mode = 'soft_multi'
        Extra = @()
    },
    [PSCustomObject]@{
        Name = 'flir_v2_c2c_adaptive_v2_40_40_seed42_win'
        Mode = 'soft_saliency'
        Extra = @('--saliency_sigmas', '0.7,1.0,1.6', '--saliency_background_gain', '0.5')
    }
)

# Validate BOTH target directories before starting the first long run.
foreach ($stage in $stages) {
    $runDir = Join-Path $checkpointDir $stage.Name
    if (Test-Path -LiteralPath $runDir) {
        $weights = @(Get-ChildItem -LiteralPath $runDir -File -Filter '*.pth')
        if ($weights.Count -gt 0) {
            throw "Existing weights found in $runDir. This fresh-training runner will not overwrite them."
        }
    }
}

if ($DryRun) {
    Write-Output "Python: $PythonExe"
    Write-Output "Working directory: $projectDir"
    foreach ($stage in $stages) {
        $trainArgs = @('-u', 'train.py') + $commonArgs + @('--name', $stage.Name, '--condition_mode', $stage.Mode) + $stage.Extra
        Write-Output ("Planned command: {0} {1}" -f $PythonExe, ($trainArgs -join ' '))
    }
    Write-Output 'Dry run complete. No training, logging, or power-setting changes were performed.'
    return
}

$originalLocation = Get-Location
$transcriptStarted = $false
$keepAwake = $false
try {
    Set-Location -LiteralPath $projectDir
    $logDir = Join-Path $thermalRoot 'runs\logs'
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $logPath = Join-Path $logDir ("c2bc_windows_{0}.log" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
    Start-Transcript -Path $logPath | Out-Null
    $transcriptStarted = $true
    Write-Host "Combined training log: $logPath"

    & $PythonExe -c "import torch, skimage, scipy, dominate; assert torch.cuda.is_available(), 'CUDA is unavailable'; print('GPU:', torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) {
        throw 'CUDA/dependency check failed. Training has not started.'
    }

    if (-not ('ThermalC2Power' -as [type])) {
        Add-Type -TypeDefinition @'
using System.Runtime.InteropServices;
public static class ThermalC2Power {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
    }
    # ES_CONTINUOUS | ES_SYSTEM_REQUIRED. The display may still turn off.
    if ([ThermalC2Power]::SetThreadExecutionState([uint32]2147483649) -eq 0) {
        throw 'Could not prevent automatic system sleep. Check Windows power settings before training.'
    }
    $keepAwake = $true
    Write-Host 'Automatic idle sleep is blocked while this runner is active.'

    foreach ($stage in $stages) {
        Write-Host ("Starting {0} at {1}" -f $stage.Name, (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'))
        $trainArgs = @('-u', 'train.py') + $commonArgs + @('--name', $stage.Name, '--condition_mode', $stage.Mode) + $stage.Extra
        & $PythonExe @trainArgs
        $trainExitCode = $LASTEXITCODE
        if ($trainExitCode -ne 0) {
            throw "$($stage.Name) failed with exit code $trainExitCode. No subsequent experiment will start."
        }
        $runDir = Join-Path $checkpointDir $stage.Name
        foreach ($netName in @('G_A', 'G_B', 'D_A', 'D_B')) {
            $weight = Join-Path $runDir ("80_net_{0}.pth" -f $netName)
            if (-not (Test-Path -LiteralPath $weight -PathType Leaf)) {
                throw "Missing final checkpoint: $weight. No subsequent experiment will start."
            }
            if ((Get-Item -LiteralPath $weight).Length -eq 0) {
                throw "Empty final checkpoint: $weight. No subsequent experiment will start."
            }
        }
        Write-Host "Completed $($stage.Name); all four epoch-80 checkpoints exist."
    }
    Write-Host 'C2b and C2c completed successfully.'
}
finally {
    if ($keepAwake) {
        [ThermalC2Power]::SetThreadExecutionState([uint32]2147483648) | Out-Null
    }
    if ($transcriptStarted) {
        Stop-Transcript | Out-Null
    }
    Set-Location -LiteralPath $originalLocation.Path
}
