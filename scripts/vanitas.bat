@echo off
setlocal
cd /d "%~dp0.."
python -m vanitas %*
exit /b %errorlevel%
