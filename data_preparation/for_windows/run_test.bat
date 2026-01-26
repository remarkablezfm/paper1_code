@echo off
chcp 65001 >nul
REM ==========================================
REM  WOD-E2E Exporter Windows 测试脚本
REM ==========================================

REM ============ 配置区域 - 请修改为你的实际路径 ============
set DATASET_ROOT=D:\Datasets\WOD_E2E_Camera_v1
set OUT_ROOT=D:\Datasets\WOD_E2E_Camera_v1\test_output_win
set INDEX_CSV=D:\Datasets\WOD_E2E_Camera_v1\val_index_filled.csv
set CONDA_ENV=wod_e2e
REM ========================================================

echo.
echo ==========================================
echo  WOD-E2E Exporter Test (Windows)
echo ==========================================
echo  Dataset:  %DATASET_ROOT%
echo  Output:   %OUT_ROOT%
echo  Env:      %CONDA_ENV%
echo ==========================================
echo.

REM 检查 conda
where conda >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Conda not found!
    pause
    exit /b 1
)

REM 激活 conda 环境
echo [INFO] Activating conda environment: %CONDA_ENV%
call conda activate %CONDA_ENV%
if errorlevel 1 (
    echo [ERROR] Failed to activate '%CONDA_ENV%'
    echo [INFO] Run install.bat first.
    pause
    exit /b 1
)

REM 显示 Python 版本
python --version

REM 检查数据集路径
if not exist "%DATASET_ROOT%" (
    echo.
    echo [ERROR] Dataset not found: %DATASET_ROOT%
    echo [INFO] Please edit DATASET_ROOT in this script.
    pause
    exit /b 1
)

REM 切换到脚本目录
cd /d "%~dp0"

echo.
echo [INFO] Starting export...
echo [INFO] Max segments: 1
echo [INFO] Workers: 1
echo.

REM 运行最小测试
python -m wod_e2e_exporter ^
    --dataset_root "%DATASET_ROOT%" ^
    --out_root "%OUT_ROOT%" ^
    --max_segments 1 ^
    --num_workers 1 ^
    --log_level DEBUG

if errorlevel 1 (
    echo.
    echo ==========================================
    echo  [ERROR] Export failed!
    echo ==========================================
    echo  Common issues:
    echo    - Waymo SDK not working on Windows
    echo    - Dataset path incorrect
    echo    - TFRecord files missing
    echo.
    echo  Solution: Use WSL instead
    echo ==========================================
) else (
    echo.
    echo ==========================================
    echo  [SUCCESS] Export completed!
    echo ==========================================
    echo  Output: %OUT_ROOT%
    echo ==========================================
)

echo.
pause
