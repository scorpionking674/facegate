@echo off
setlocal

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Creating local virtual environment in "%CD%\.venv"
    py -3.10 -m venv .venv
    if errorlevel 1 (
        echo Python 3.10 launcher failed. Trying default python.
        python -m venv .venv
    )
)

if not exist ".venv\Scripts\python.exe" (
    echo Could not create local virtual environment.
    echo Install Python 3.10 or 3.11, then run setup.bat again.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

python -m ensurepip --upgrade
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency install failed.
    pause
    exit /b 1
)

if not exist "reference_image" mkdir "reference_image"
if not exist "insightface\models\antelopev2" mkdir "insightface\models\antelopev2"

echo.
echo Setup complete.
echo Put reference images in: reference_image
echo Put antelopev2 ONNX model files in: insightface\models\antelopev2
pause
