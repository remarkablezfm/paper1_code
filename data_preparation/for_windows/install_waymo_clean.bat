@echo off
chcp 65001 >nul
REM ==========================================
REM  安装 waymo_clean 环境 (Windows)
REM  基于 WSL waymo_clean 配置
REM ==========================================

echo.
echo ==========================================
echo  waymo_clean Environment Setup (Windows)
echo ==========================================
echo.

REM 检查 conda
where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Conda not found!
    echo Please install Anaconda or Miniconda first.
    pause
    exit /b 1
)

echo [INFO] Conda version:
conda --version
echo.

REM 选择安装方式
echo Choose installation method:
echo   1. Full install (all packages from waymo_clean)
echo   2. Minimal install (only required packages)
echo.
set /p CHOICE="Enter choice (1 or 2): "

if "%CHOICE%"=="2" (
    set REQ_FILE=waymo_clean_minimal.txt
    echo [INFO] Using minimal requirements
) else (
    set REQ_FILE=waymo_clean_requirements.txt
    echo [INFO] Using full requirements
)

REM 检查环境是否已存在
conda env list | findstr /C:"waymo_clean" >nul 2>nul
if not errorlevel 1 (
    echo.
    echo [INFO] Environment 'waymo_clean' already exists.
    set /p CONFIRM="Remove and recreate? (y/N): "
    if /i "%CONFIRM%"=="y" (
        echo [INFO] Removing existing environment...
        conda env remove -n waymo_clean -y
    ) else (
        echo [INFO] Keeping existing environment.
        pause
        exit /b 0
    )
)

echo.
echo [Step 1/3] Creating conda environment (Python 3.9)...
conda create -n waymo_clean python=3.9 -y
if errorlevel 1 (
    echo [ERROR] Failed to create environment!
    pause
    exit /b 1
)

echo.
echo [Step 2/3] Activating environment...
call conda activate waymo_clean
if errorlevel 1 (
    echo [ERROR] Failed to activate environment!
    pause
    exit /b 1
)

echo.
echo [Step 3/3] Installing packages from %REQ_FILE%...
echo [INFO] This may take several minutes...
echo.

pip install -r %REQ_FILE%

if errorlevel 1 (
    echo.
    echo ==========================================
    echo  [WARNING] Some packages may have failed
    echo ==========================================
    echo  Waymo SDK officially supports Linux only.
    echo  If it failed, use WSL instead.
    echo ==========================================
) else (
    echo.
    echo ==========================================
    echo  [SUCCESS] Installation complete!
    echo ==========================================
)

echo.
echo Verifying installation...
echo.
python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
python -c "from waymo_open_dataset.protos import end_to_end_driving_data_pb2; print('Waymo SDK: OK')" 2>nul || echo Waymo SDK: FAILED

echo.
echo ==========================================
echo  To activate: conda activate waymo_clean
echo ==========================================
echo.

pause
