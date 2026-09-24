# Tencent Windows resident-node incremental installer. Run elevated on C:\quant host.
param(
  [string]$QuantRoot = 'C:\quant',
  [string]$Python = 'C:\Program Files\Python312\python.exe'
)
$ErrorActionPreference = 'Stop'
if (!(Test-Path $QuantRoot)) { throw "Quant root missing: $QuantRoot" }
if (!(Test-Path $Python)) { throw "Python missing: $Python" }
$stamp = Get-Date -Format 'yyyyMMddHHmmss'
$backup = Join-Path $QuantRoot "backup\resident-$stamp"
New-Item -ItemType Directory -Force -Path $backup | Out-Null
$bat = Join-Path $QuantRoot 'run_after_close_unified.bat'
if (Test-Path $bat) { Copy-Item $bat (Join-Path $backup 'run_after_close_unified.bat') -Force }

# Bind every existing after-close delivery to the immutable release gate.
$content = Get-Content $bat -Raw
if ($content -notmatch 'data_release') {
  $needle = 'call C:\quant\run_task.bat daily_review_chain.py >> "%LOG%" 2>&1'
  $replacement = "call C:\quant\deploy\run_data_release.bat >> `"%LOG%`" 2>&1`r`nif errorlevel 1 goto :fail`r`ncall C:\quant\run_task.bat export_production_targets.py >> `"%LOG%`" 2>&1`r`nif errorlevel 1 goto :fail`r`n$needle"
  if ($content -notlike "*$needle*") { throw 'after-close batch shape changed; refusing automatic patch' }
  $content = $content.Replace($needle, $replacement)
  Set-Content -Path $bat -Value $content -Encoding ASCII
}
if ($content -notmatch 'export_production_targets') {
  $needle = 'call C:\quant\run_task.bat daily_review_chain.py >> "%LOG%" 2>&1'
  $replacement = "call C:\quant\run_task.bat export_production_targets.py >> `"%LOG%`" 2>&1`r`nif errorlevel 1 goto :fail`r`n$needle"
  if ($content -notlike "*$needle*") { throw 'after-close batch target insertion point missing' }
  $content = $content.Replace($needle, $replacement)
  Set-Content -Path $bat -Value $content -Encoding ASCII
}
if ($content -notmatch 'production_pipeline') {
  $needle = 'call C:\quant\run_task.bat research_fusion_snapshot.py >> "%LOG%" 2>&1'
  $replacement = "call C:\quant\run_task.bat run_production_pipeline.py >> `"%LOG%`" 2>&1`r`n$needle"
  if ($content -notlike "*$needle*") { throw 'after-close production audit insertion point missing' }
  $content = $content.Replace($needle, $replacement)
  Set-Content -Path $bat -Value $content -Encoding ASCII
}

$env:QUANT_ROOT = $QuantRoot
$env:PYTHONPATH = $QuantRoot
$env:QUANT_RESIDENT_COMMAND = "$QuantRoot\run_after_close_unified.bat"
$actionRecovery = New-ScheduledTaskAction -Execute $Python -Argument "-u $QuantRoot\scripts\resident_ops.py catch-up" -WorkingDirectory $QuantRoot
$actionDigest = New-ScheduledTaskAction -Execute $Python -Argument "-u $QuantRoot\scripts\resident_ops.py digest" -WorkingDirectory $QuantRoot
$triggerBoot = New-ScheduledTaskTrigger -AtStartup
$triggerDaily = New-ScheduledTaskTrigger -Daily -At 7:10PM
$principal = New-ScheduledTaskPrincipal -UserId 'Administrator' -LogonType ServiceAccount -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Hours 3) -RestartCount 2 -RestartInterval (New-TimeSpan -Minutes 10) -StartWhenAvailable
Register-ScheduledTask -TaskName 'quant_resident_recovery' -Action $actionRecovery -Trigger $triggerBoot -Principal $principal -Settings $settings -Force | Out-Null
Register-ScheduledTask -TaskName 'quant_resident_digest' -Action $actionDigest -Trigger $triggerDaily -Principal $principal -Settings $settings -Force | Out-Null

# Do not send a test Feishu message; only validate that required code and secrets file exist.
$checks = @{
  resident_ops = Test-Path "$QuantRoot\scripts\resident_ops.py"
  data_release = Test-Path "$QuantRoot\quant_system\data_release.py"
  feishu_config = Test-Path "$QuantRoot\config\.env.secrets"
  after_close = Test-Path $bat
}
if ($checks.Values -contains $false) { throw "resident install incomplete: $($checks | ConvertTo-Json -Compress)" }
$files=@("scripts\\resident_ops.py","scripts\\build_data_release.py","quant_system\\data_release.py","quant_web\\server.py","quant_web\\static\\ops_console.html")
$manifest=[ordered]@{schema='quant-windows-deploy/v1';installed_at=(Get-Date).ToString('o');backup=$backup;after_close=$bat;files=@{};release_pointer=(Join-Path $QuantRoot 'generated\\data_releases\\latest.json')}
foreach($rel in $files){$p=Join-Path $QuantRoot $rel;if(Test-Path $p){$manifest.files[$rel]=(Get-FileHash $p -Algorithm SHA256).Hash}}
$manifest | ConvertTo-Json -Depth 5 | Set-Content (Join-Path $backup 'deploy_manifest.json') -Encoding UTF8
Write-Output ($checks | ConvertTo-Json -Compress)
Get-ScheduledTask -TaskName quant_resident_recovery,quant_resident_digest | Select-Object TaskName,State | Format-Table -AutoSize
