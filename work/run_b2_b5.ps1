$ErrorActionPreference = 'Continue'
$python = '.\.venv\Scripts\python.exe'
$cfg = 'configs/assist2017_v46_ablation.json'
$variants = @(
    'no_B2_state_memory',
    'no_B3_attention_fusion',
    'no_B4_concept_routing',
    'no_B5_calibrated_readout'
)
foreach ($v in $variants) {
    $out = "runs\ablation_${v}_20260920"
    if (Test-Path $out) { Write-Output "SKIP $v (exists)"; continue }
    Write-Output "=== START $v $(Get-Date -Format 'HH:mm:ss') ==="
    & $python -m a2g.experiment train --config $cfg --output $out --data-dir data\assist2017 --variant $v 2>&1 | Tee-Object -FilePath "runs\ablation_${v}_console.log" | Select-Object -Last 3
    Write-Output "=== END $v exit=$LASTEXITCODE $(Get-Date -Format 'HH:mm:ss') ==="
}
Write-Output "ALL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
