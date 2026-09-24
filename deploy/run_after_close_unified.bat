@echo off
setlocal
cd /d C:\quant
if not exist logs mkdir logs
set LOG=C:\quant\logs\after_close_unified.log
echo [%date% %time%] START >> "%LOG%"
call C:\quant\run_task.bat ic_oos_report.py --n 150 --seed 42 --forward 5 >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
call C:\quant\run_task.bat build_factor_quality_registry.py >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
call C:\quant\run_task.bat factor_revalidation.py >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
call C:\quant\run_task.bat collect_industry_chain_history.py >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
call C:\quant\run_task.bat daily_review_chain.py >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
call C:\quant\run_task.bat research_fusion_snapshot.py >> "%LOG%" 2>&1
call C:\quant\run_task.bat unified_decision_snapshot.py --mode after_close >> "%LOG%" 2>&1
call C:\quant\run_task.bat html_report_generator.py >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
call C:\quant\run_task.bat push_after_close_report.py >> "%LOG%" 2>&1
if errorlevel 1 goto :fail
echo [%date% %time%] COMPLETE >> "%LOG%"
exit /b 0
:fail
echo [%date% %time%] FAILED rc=%errorlevel% >> "%LOG%"
exit /b %errorlevel%
