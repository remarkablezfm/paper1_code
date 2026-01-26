@echo off
chcp 65001 >nul
REM ==========================================
REM  WOD-E2E Windows Conda 环境安装脚本
REM  基于 WSL waymo_clean 环境配置
REM ==========================================

echo.
echo ==========================================
echo  WOD-E2E Environment Setup for Windows
echo  (Based on WSL waymo_clean)
echo ==========================================
echo.

REM 检查 conda
where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Conda not found!
    echo Please install Anaconda or Miniconda first.
    echo Download: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo [INFO] Found conda
conda --version
echo.

REM 检查环境是否已存在
conda env list | findstr /C:"wod_e2e" >nul 2>nul
if not errorlevel 1 (
    echo [INFO] Environment 'wod_e2e' already exists.
    set /p CONFIRM="Remove and recreate? (y/N): "
    if /i "%CONFIRM%"=="y" (
        echo [INFO] Removing existing environment...
        conda env remove -n wod_e2e -y
    ) else (
        echo [INFO] Keeping existing environment.
        echo [INFO] To activate: conda activate wod_e2e
        pause
        exit /b 0
    )
)

echo.
echo [Step 1/5] Creating conda environment (Python 3.9)...
conda create -n wod_e2e python=3.9 -y
if errorlevel 1 (
    echo [ERROR] Failed to create conda environment!
    pause
    exit /b 1
)

echo.
echo [Step 2/5] Activating environment...
call conda activate wod_e2e
if errorlevel 1 (
    echo [ERROR] Failed to activate environment!
    pause
    exit /b 1
)

echo.
echo [Step 3/5] Installing core dependencies...
pip install numpy==1.23.5 pandas==1.5.3 opencv-python==4.5.5.64 tqdm==4.67.1 pillow==9.2.0
if errorlevel 1 (
    echo [WARNING] Some core packages may have failed.
)

echo.
echo [Step 4/5] Installing TensorFlow 2.13...
pip install tensorflow==2.13.0
if errorlevel 1 (
    echo [ERROR] TensorFlow installation failed!
    echo [INFO] Try: pip install tensorflow==2.13.0 --no-cache-dir
    pause
    exit /b 1
)

echo.
echo [INFO] Installing protobuf 3.20.3 (required version)...
pip install protobuf==3.20.3
if errorlevel 1 (
    echo [WARNING] protobuf installation may have issues.
)

echo.
echo [Step 5/5] Installing Waymo Open Dataset SDK...
echo [WARNING] This package officially supports Linux only.
echo [WARNING] Installation may fail on Windows.
echo.
pip install waymo-open-dataset-tf-2-12-0==1.6.7

if errorlevel 1 (
    echo.
    echo ==========================================
    echo  [WARNING] Waymo SDK installation failed!
    echo ==========================================
    echo  The Waymo SDK officially only supports Linux.
    echo.
    echo  Options:
    echo    1. Use WSL (recommended)
    echo    2. Try: pip install waymo-open-dataset-tf-2-12-0
    echo       (without version constraint)
    echo ==========================================
) else (
    echo.
    echo ==========================================
    echo  [SUCCESS] All packages installed!
    echo ==========================================
)

echo.
echo ==========================================
echo  Verifying installation...
echo ==========================================
echo.

python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
python -c "from waymo_open_dataset.protos import end_to_end_driving_data_pb2; print('Waymo SDK: OK')" 2>nul
if errorlevel 1 (
    echo Waymo SDK: FAILED (use WSL for full functionality)
)

echo.
echo ==========================================
echo  To activate environment:
echo      conda activate wod_e2e
echo.
echo  To run test:
echo      run_test.bat
echo ==========================================
echo.

pause
