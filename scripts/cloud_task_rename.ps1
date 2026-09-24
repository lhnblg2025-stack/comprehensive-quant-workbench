# cloud_task_rename.ps1 - rename analysis_v11 -> analysis_core in all quant_* schtasks (V12.3 decoupling)
$ErrorActionPreference = 'Continue'
$log = 'C:\quant\logs\task_rename.log'
"[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] START" | Out-File $log -Encoding utf8

# 1) rewrite task action args: analysis_v11 -> analysis_core
$names = @(schtasks /query /fo csv | ForEach-Object { ($_ -split ',')[1].Trim('"') } | Where-Object { $_ -like '\quant_*' })
$done = 0
foreach ($n in $names) {
    $xml = (schtasks /query /tn $n /xml | Out-String)
    if ($xml -match 'analysis_v11') {
        $new = $xml.Replace('analysis_v11', 'analysis_core')
        $tmp = 'C:\quant\logs\task_tmp.xml'
        [IO.File]::WriteAllText($tmp, $new, [Text.Encoding]::Unicode)
        schtasks /delete /tn $n /f | Out-Null
        schtasks /create /tn $n /xml $tmp /f | Out-Null
        "[$(Get-Date -Format 'HH:mm:ss')] RENAMED $n" | Out-File $log -Append -Encoding utf8
        $done++
    }
}
"args_rewritten=$done" | Out-File $log -Append -Encoding utf8

# 2) task name quant_v11_weekly -> quant_weekly
try {
    $xml = (schtasks /query /tn quant_v11_weekly /xml | Out-String)
    if ($xml -match 'quant_v11_weekly') {
        $new = $xml.Replace('quant_v11_weekly', 'quant_weekly').Replace('analysis_v11', 'analysis_core')
        $tmp = 'C:\quant\logs\task_tmp2.xml'
        [IO.File]::WriteAllText($tmp, $new, [Text.Encoding]::Unicode)
        schtasks /delete /tn quant_v11_weekly /f | Out-Null
        schtasks /create /tn quant_weekly /xml $tmp /f | Out-Null
        "taskname_renamed=quant_weekly" | Out-File $log -Append -Encoding utf8
    } else {
        "taskname_renamed=no_v11_weekly" | Out-File $log -Append -Encoding utf8
    }
} catch {
    "taskname_rename_error=$($_.Exception.Message)" | Out-File $log -Append -Encoding utf8
}
"RESULT=DONE" | Out-File $log -Append -Encoding utf8
