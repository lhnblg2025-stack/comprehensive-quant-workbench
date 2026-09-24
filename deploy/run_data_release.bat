@echo off
setlocal
set QUANT_ROOT=C:\quant
set PYTHONPATH=C:\quant
set PYTHONIOENCODING=utf-8
call C:\quant\run_task.bat build_data_release.py
exit /b %errorlevel%
