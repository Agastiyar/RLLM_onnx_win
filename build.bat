@echo off
title RLLM Build
set PYO3_PYTHON=E:\Python310\python.exe
echo ========================================
echo   RLLM Build
echo ========================================
echo.
echo [1/2] Building...
cd /d E:\3\R\MC\RLLM_onnx_win
cargo build --release
if %errorlevel% neq 0 (
    echo BUILD FAILED!
    pause
    exit /b 1
)
echo.
echo [2/2] Deploying to Python 3.10...
copy /Y "E:\3\R\MC\RLLM_onnx_win\target\release\rllm.dll" "E:\Python310\Lib\site-packages\rllm.pyd"
echo.
echo [3/3] Verifying...
E:\Python310\python.exe -c "import rllm; print(rllm.version(), rllm.cuda_info())"
echo.
echo Done!
pause
