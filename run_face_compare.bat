@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    call setup.bat
    if errorlevel 1 exit /b 1
)

call ".venv\Scripts\activate.bat"
python compare_faces.py %*
pause
