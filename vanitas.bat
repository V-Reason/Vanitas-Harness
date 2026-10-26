@echo off
setlocal
set "VANITAS_ORIG_CWD=%CD%"
cd /d "%~dp0"
python -m vanitas %*
exit /b %errorlevel%
