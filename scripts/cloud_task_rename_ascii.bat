@echo off
rem cloud_task_rename_ascii.bat - analysis_v11 -> analysis_core in quant_* schtasks (V12.3)
setlocal enabledelayedexpansion
set LOG=C:\quant\logs\task_rename3.log
echo [%date% %time%] START > %LOG%

for /f "skip=1 tokens=2 delims=," %%A in ('schtasks /query /fo csv 2^>nul') do (
  for /f "usebackq delims=" %%N in ("%%~A") do (
    set TN=%%N
    if /i "!TN:~0,1!"=="\" (
      echo !TN! | findstr /i "\\quant_" >nul 2>&1 && call :rewrite "!TN!"
    )
  )
)
echo RESULT=DONE >> %LOG%
echo DONE
exit /b 0

:rewrite
set TN=%~1
schtasks /query /tn %TN% /xml > C:\quant\logs\task_tmp4.xml 2>nul
if errorlevel 1 exit /b 0
findstr /c:"analysis_v11" C:\quant\logs\task_tmp4.xml >nul 2>&1
if errorlevel 1 exit /b 0
powershell -NoProfile -Command "$p='C:\quant\logs\task_tmp4.xml'; $x=[IO.File]::ReadAllText($p,[Text.Encoding]::Unicode).Replace('analysis_v11','analysis_core'); [IO.File]::WriteAllText($p,$x,[Text.Encoding]::Unicode)"
schtasks /delete /tn %TN% /f >nul 2>&1
schtasks /create /tn %TN% /xml C:\quant\logs\task_tmp4.xml /f >nul 2>&1
echo [%date% %time%] RENAMED %TN% >> %LOG%
exit /b 0
