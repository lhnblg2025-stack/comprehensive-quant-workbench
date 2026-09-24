# cloud_task_rename2.ps1 - robust rewrite: analysis_v11 -> analysis_core in all quant_* schtasks
$ErrorActionPreference = 'Continue'
$log = 'C:\quant\logs\task_rename2.log'
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START" | Out-File $log -Encoding utf8

# enumerate task names robustly (strip UTF-16 NULs, take 2nd CSV column)
$names = @()
foreach ($raw in (schtasks /query /fo csv)) {
    $clean = ($raw -replace "`0", "").Trim()
    if ($clean -eq '') { continue }
    $parts = $clean.Split(',')
    if ($parts.Count -ge 2) {
        $tn = $parts[1].Trim('"')
        if ($tn -like '\quant_*') { $names += $tn }
    }
}
"tasks_found=$($names.Count)" | Out-File $log -Append -Encoding utf8

$done = 0
foreach ($n in $names) {
    $xml = (schtasks /query /tn $n /xml | Out-String)
    if ($xml -match 'analysis_v11') {
        $new = $xml.Replace('analysis_v11', 'analysis_core')
        $tmp = 'C:\quant\logs\task_tmp3.xml'
        [IO.File]::WriteAllText($tmp, $new, [Text.Encoding]::Unicode)
        schtasks /delete /tn $n /f | Out-Null
        schtasks /create /tn $n /xml $tmp /f | Out-Null
        "[$(Get-Date -Format 'HH:mm:ss')] RENAMED $n" | Out-File $log -Append -Encoding utf8
        $done++
    }
}
"args_rewritten=$done" | Out-File $log -Append -Encoding utf8
"RESULT=DONE" | Out-File $log -Append -Encoding utf8
