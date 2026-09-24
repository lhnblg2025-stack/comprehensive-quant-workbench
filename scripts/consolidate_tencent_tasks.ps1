$ErrorActionPreference='Stop'
$root='C:\quant'; $stamp=Get-Date -Format 'yyyyMMddHHmmss'; $backup=Join-Path $root "backup\task-consolidation-$stamp"; New-Item -ItemType Directory -Force $backup | Out-Null
$keep=@('quant_after_close','quant_after_close_unified','quant_resident_recovery','quant_resident_digest','quant_health_guardian')
$disable=@('quant_pipeline_daily','quant_daily_report','quant_review_fusion','quant_html_report','quant_push_after_close')
foreach($name in ($keep+$disable)){
  $task=Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue
  if($task){ Export-ScheduledTask -TaskName $name | Set-Content -Path (Join-Path $backup ($name+'.xml')) -Encoding UTF8 }
}
foreach($name in $disable){
  if(Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue){ Disable-ScheduledTask -TaskName $name | Out-Null }
}
$manifest=[ordered]@{schema='quant-task-consolidation/v1';created_at=(Get-Date).ToString('o');keep=$keep;disabled=$disable;backup=$backup;note='disabled only; tasks remain manually restorable'}
$manifest | ConvertTo-Json -Depth 4 | Set-Content (Join-Path $backup 'manifest.json') -Encoding UTF8
Get-ScheduledTask -TaskName ($keep+$disable) -ErrorAction SilentlyContinue | Select-Object TaskName,State | ConvertTo-Json
