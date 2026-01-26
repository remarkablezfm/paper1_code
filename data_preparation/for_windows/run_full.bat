@echo off
chcp 65001 >nul
REM ==========================================
REM  WOD-E2E Exporter 完整导出脚本 (Windows)
REM ==========================================

REM ============ 配置区域 - 请修改 ============
set DATASET_ROOT=D:\Datasets\WOD_E2E_Camera_v1
set OUT_ROOT=D:\Datasets\WOD_E2E_Camera_v1\output
set INDEX_CSV=D:\Datasets\WOD_E2E_Camera_v1\val_index_filled.csv
set SCENARIO=Cut_ins
set MAX_SEGMENTS=
set NUM_WORKERS=1
set CONDA_ENV=wod_e2e
REM ==========================================

echo.
echo ==========================================
echo  WOD-E2E Full Export (Windows)
echo ==========================================
echo  Dataset:   %DATASET_ROOT%
echo  Output:    %OUT_ROOT%
echo  Scenario:  %SCENARIO%
echo  Workers:   %NUM_WORKERS%
echo ==========================================
echo.

REM 激活环境
call conda activate %CONDA_ENV%
if errorlevel 1 (
    echo [ERROR] Failed to activate '%CONDA_ENV%'
    pause
    exit /b 1
)

cd /d "%~dp0"

echo [INFO] Starting full export...
echo.

REM 构建命令
set CMD=python -m wod_e2e_exporter
set CMD=%CMD% --dataset_root "%DATASET_ROOT%"
set CMD=%CMD% --out_root "%OUT_ROOT%"
set CMD=%CMD% --num_workers %NUM_WORKERS%
set CMD=%CMD% --log_level INFO

if defined INDEX_CSV (
    if exist "%INDEX_CSV%" (
        set CMD=%CMD% --index_csv "%INDEX_CSV%"
    )
)

if not "%SCENARIO%"=="" (
    if not "%SCENARIO%"=="ALL" (
        set CMD=%CMD% --scenario_cluster %SCENARIO%
    )
)

if defined MAX_SEGMENTS (
    set CMD=%CMD% --max_segments %MAX_SEGMENTS%
)

echo [CMD] %CMD%
echo.

%CMD%

if errorlevel 1 (
    echo.
    echo [ERROR] Export failed!
) else (
    echo.
    echo [SUCCESS] Export completed!
    echo Output: %OUT_ROOT%
)

pause
