@echo off
title ClipMore Studio
cd /d "%~dp0"
echo ===================================================
echo   ClipMore Studio - Autonomous Viral Stream Clipper
echo ===================================================
echo Starting web server on http://127.0.0.1:8501 ...
timeout /t 2 /nobreak >nul
start "" http://127.0.0.1:8501
.\.venv\Scripts\python.exe app.py
pause
