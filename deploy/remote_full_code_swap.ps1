param([string]$QuantRoot='C:\quant',[string]$Archive='C:\Users\Administrator\AppData\Local\Temp\tencent-full-code-v0.1.tgz')
$ErrorActionPreference='Stop'
$stamp=Get-Date -Format 'yyyyMMddHHmmss'
$backup=Join-Path $QuantRoot "backup\full-code-$stamp"
$stage=Join-Path $env:TEMP "quant-full-stage-$stamp"
New-Item -ItemType Directory -Force -Path $backup,$stage | Out-Null
$archiveHash=(Get-FileHash $Archive -Algorithm SHA256).Hash
tar -xzf $Archive -C $stage
foreach($name in @('quant_system','quant_web','scripts','deploy')){
  $old=Join-Path $QuantRoot $name
  if(Test-Path $old){ Move-Item $old (Join-Path $backup $name) -Force }
  Move-Item (Join-Path $stage $name) (Join-Path $QuantRoot $name) -Force
}
# Config code/templates are refreshed file-wise; secrets and runtime databases remain untouched.
$srcConfig=Join-Path $stage 'config'; $dstConfig=Join-Path $QuantRoot 'config'; New-Item -ItemType Directory -Force $dstConfig | Out-Null
Get-ChildItem $srcConfig -File -Recurse | Where-Object { $_.FullName -notmatch '\\.env\.secrets$|\.sqlite|\.db$' } | ForEach-Object {
  $relative=$_.FullName.Substring($srcConfig.Length).TrimStart('\'); $target=Join-Path $dstConfig $relative
  New-Item -ItemType Directory -Force (Split-Path $target) | Out-Null; Copy-Item $_.FullName $target -Force
}
$files=@('quant_system\product_contract.py','quant_system\data_release.py','quant_system\strategy_engine.py','quant_system\context.py','quant_system\production_pipeline.py','quant_web\server.py','quant_web\static\ops_console.html','scripts\resident_ops.py','scripts\build_data_release.py','scripts\export_production_targets.py','scripts\release_repair.py')
$hashes=[ordered]@{}
foreach($rel in $files){$p=Join-Path $QuantRoot $rel;if(Test-Path $p){$hashes[$rel]=(Get-FileHash $p -Algorithm SHA256).Hash}}
$manifest=[ordered]@{schema='quant-full-code-swap/v1';installed_at=(Get-Date).ToString('o');archive_sha256=$archiveHash;backup=$backup;files=$hashes;preserved=@('data_warehouse','generated','config\.env.secrets','*.db','logs')}
$manifest | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $backup 'deploy_manifest.json') -Encoding UTF8
Remove-Item -Recurse -Force $stage
Write-Output ($manifest | ConvertTo-Json -Depth 5)
