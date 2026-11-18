@echo off
cd /d "%~dp0\.."
if "%~1"=="" (
    echo Usage: run_fallback.bat path\to\video.mp4
    echo   or:  run_fallback.bat --test
    pause
    exit /b 1
)
.\.venv\Scripts\python.exe manual\fallback_clipper.py %*
pause
