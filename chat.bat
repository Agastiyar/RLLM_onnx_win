@echo off
title RLLM Chat
set PYO3_PYTHON=E:\Python310\python.exe
set PYTHONIOENCODING=utf-8
echo ========================================
echo   RLLM Chat - SmolLM 135M on T600
echo ========================================
echo.
echo Loading model (may take 10-30 seconds)...
echo.
E:\Python310\python.exe -u "%~dp0chat.py"
echo.
echo ========================================
echo   Chat ended
echo ========================================
pause
