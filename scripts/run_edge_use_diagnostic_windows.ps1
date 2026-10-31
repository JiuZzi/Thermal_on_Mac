param(
    [string]$Python = "python",
    [string]$RunDir = "",
    [string]$Conditions = "",
    [string]$Prior = "runs/learned_color/rgb_prior_seed42_loss_v1.json",
    [string]$CacheDir = "runs/D/model_cache",
    [string]$OutputDir = "",
    [int]$Limit = 4,
    [int]$Step = -1,
    [switch]$Best,
    [switch]$ListRuns,
    [switch]$CheckOnly,
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
if ($Limit -lt 1) { throw "Limit must be positive." }
if ($Step -lt -1) { throw "Step must be nonnegative, or omitted for latest." }
if ($Best -and $Step -ge 0) { throw "Choose Best or Step, not both." }
if ($Resume -and -not $OutputDir) { throw "Supply the original -OutputDir with -Resume." }
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $discoveryArgs = @("-m", "D.diagnostics.discover", "--prior", $Prior)
    if ($RunDir) { $discoveryArgs += @("--run-dir", $RunDir) }
    $inventoryText = & $Python @discoveryArgs
    if ($LASTEXITCODE -ne 0) { throw "Discovery failed. Use the Python environment from D training." }
    $inventory = ($inventoryText -join "`n") | ConvertFrom-Json
    $runs = @($inventory.runs)
    if ($runs.Count -eq 0) { throw "No formal D run found under runs/D/unpaired/training. Supply -RunDir if stored elsewhere." }
    for ($i = 0; $i -lt $runs.Count; $i++) {
        $r = $runs[$i]
        Write-Host ("[{0}] {1} | latest update={2} | matching val caches={3} | prior match={4}" -f ($i+1), $r.name, $r.latest_step, @($r.conditions).Count, $r.prior_matches)
        if ($r.checkpoint_error) { Write-Host ("    Latest pointer: " + $r.checkpoint_error) }
    }
    if ($ListRuns) { return }
    $index = 0
    if ($runs.Count -gt 1) {
        $answer = Read-Host "Select the run number for this fixed-weight diagnostic"
        if (-not [int]::TryParse($answer, [ref]$index) -or $index -lt 1 -or $index -gt $runs.Count) { throw "Invalid run number." }
        $index--
    }
    $selected = $runs[$index]
    if (-not $selected.prior_matches) { throw "The original learned prior is missing or differs. Supply -Prior." }
    if (-not $Conditions) {
        if (@($selected.conditions).Count -eq 0) { throw "No matching TIR validation cache found. Prepare it using the original run recipe, then supply -Conditions." }
        $Conditions = @($selected.conditions)[0].path
    }
    if (-not $OutputDir) { $OutputDir = Join-Path $selected.run_dir ("diagnostics/edge_use_" + (Get-Date -Format "yyyyMMdd_HHmmss")) }
    $compareArgs = @("-u", "-m", "D.diagnostics.compare_edge_use", "compare", "--backend", "formal",
        "--device", "cuda:0", "--run-dir", $selected.run_dir, "--conditions", $Conditions,
        "--prior", $Prior, "--cache-dir", $CacheDir, "--limit", "$Limit", "--output-dir", $OutputDir)
    if ($Best) { $compareArgs += "--best" }
    elseif ($Step -ge 0) { $compareArgs += @("--step", "$Step") }
    if ($Resume) { $compareArgs += "--resume" }
    Write-Host "Checking checkpoint, code, original prior and P/E/Q identities; no training."
    & $Python @compareArgs --check-only
    if ($LASTEXITCODE -ne 0) { throw "Preflight failed. Existing run and conditions were not changed." }
    if ($CheckOnly) { return }
    & $Python @compareArgs
    if ($LASTEXITCODE -ne 0) { throw "Comparison stopped. Inspect output status/log; the training run is untouched." }
    Write-Host ("Open: " + (Join-Path $OutputDir "index.html"))
} finally {
    Pop-Location
}
