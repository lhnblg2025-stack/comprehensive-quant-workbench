@echo off
rem ===== cloud_task_rename.bat — 云端 schtasks 参数 analysis_v11→analysis_core (V12.3 解耦) =====
rem 用法: cloud_task_rename.bat   (需管理员, 在 C:\quant 下运行)
setlocal enabledelayedexpansion
set LOG=C:\quant\logs\task_rename.log
echo [%date% %time%] START > %LOG%

rem ---- 1. 全部 quant_* 任务: 参数内 analysis_v11 -> analysis_core ----
for /f "tokens=*" %%T in ('schtasks /query /fo csv 2^>nul ^| findstr /i "\\quant_" ^| findstr /v "\\quant_web_"') do (
  for /f "tokens=1 delims=," %%N in ("%%T") do (
    set TN=%%~N
    call :rename_one "!TN!"
  )
)

rem ---- 2. quant_v11_weekly -> quant_weekly (任务名也改) ----
call :rename_task_name "quant_v11_weekly" "quant_weekly"

echo [%date% %time%] DONE >> %LOG%
echo RENAME_DONE
exit /b 0

:rename_one
set TN=%~1
schtasks /query /tn %TN% /xml > C:\quant\logs\task_tmp.xml 2>nul
if errorlevel 1 goto :eof
findstr /c:"analysis_v11" C:\quant\logs\task_tmp.xml >nul 2>&1
if errorlevel 1 goto :eof
echo [%date% %time%] RENAME %TN% >> %LOG%
powershell -NoProfile -Command "$p='C:\quant\logs\task_tmp.xml'; $x=[IO.File]::ReadAllText($p,[Text.Encoding]::Unicode); $x=$x.Replace('analysis_v11','analysis_core'); [IO.File]::WriteAllText($p,$x,[Text.Encoding]::Unicode)"
schtasks /delete /tn %TN% /f >nul 2>&1
schtasks /create /tn %TN% /xml C:\quant\logs\task_tmp.xml /f >nul 2>&1
goto :eof

:rename_task_name
set OLD=%~1
set NEW=%~2
schtasks /query /tn %OLD% /xml > C:\quant\logs\task_tmp2.xml 2>nul
if errorlevel 1 goto :eof
powershell -NoProfile -Command "$p='C:\quant\logs\task_tmp2.xml'; $x=[IO.File]::ReadAllText($p,[Text.Encoding]::Unicode); $x=$x.Replace('quant_v11_weekly','quant_weekly').Replace('analysis_v11','analysis_core'); [IO.File]::WriteAllText($p,$x,[Text.Encoding]::Unicode)"
schtasks /delete /tn %OLD% /f >nul 2>&1
schtasks /create /tn %NEW% /xml C:\quant\logs\task_tmp2.xml /f >nul 2>&1
echo [%date% %time%] TASKNAME %OLD% -> %NEW% >> %LOG%
goto :eof
