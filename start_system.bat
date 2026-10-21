@echo off
title DeepFake Detector Platform
echo ========================================================
echo Starting Explainable Deepfake Detection Platform
echo Multi-Agent Architecture (Primary + Secondary + Simple UI)
echo ========================================================
echo.

:: Start FastAPI Backend in background
echo [1/2] Starting FastAPI Backend on http://localhost:8000 ...
start "Deepfake Backend API" cmd /k "cd /d "%~dp0" && python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000"

:: Wait 3 seconds for backend to initialize
timeout /t 3 /nobreak >nul

:: Start Vite Frontend in background
echo [2/2] Starting React Frontend on http://localhost:5173 ...
start "Deepfake Frontend UI" cmd /k "cd /d "%~dp0frontend" && npm run dev"

echo.
echo ========================================================
echo Platform is launched!
echo Access your UI at: http://localhost:5173
echo Backend API at:    http://localhost:8000
echo ========================================================
pause
