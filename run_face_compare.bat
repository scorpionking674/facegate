@echo off
setlocal

cd /d "%~dp0"

if exist "local_settings.bat" call "local_settings.bat"

if not exist ".venv\Scripts\python.exe" (
    call setup.bat
    if errorlevel 1 exit /b 1
)

".venv\Scripts\python.exe" compare_faces.py %*
pause
