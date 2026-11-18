@echo off
cd /d "%~dp0\.."
if "%~1"=="" (
    echo Usage: run_ai.bat "https://www.twitch.tv/channelname"
    echo   or:  run_ai.bat --test
    pause
    exit /b 1
)
.\.venv\Scripts\python.exe ai\auto_viral_clipper.py %*
pause
