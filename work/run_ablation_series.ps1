$ErrorActionPreference = 'Continue'
$python = '.\.venv\Scripts\python.exe'
$cfg = 'configs/assist2017_v46_ablation.json'
$variants = @('full','no_evidence','no_ssm','no_attention','no_split_boundary','no_rwce','no_memory_readout','no_evidence_equivalent_residual','no_shared_embeddings')
foreach ($v in $variants) {
  $out = "runs\ablation_${v}_20260919"
  if (Test-Path $out) { Write-Output "SKIP $v (exists)"; continue }
  Write-Output "=== START $v $(Get-Date -Format 'HH:mm:ss') ==="
  & $python -m a2g.experiment train --config $cfg --output $out --data-dir data\assist2017 --variant $v 2>&1 | Tee-Object -FilePath "runs\ablation_${v}_console.log" | Select-Object -Last 3
  Write-Output "=== END $v exit=$LASTEXITCODE $(Get-Date -Format 'HH:mm:ss') ==="
}
Write-Output "ALL DONE $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
