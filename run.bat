@echo off
if not exist backend\venv (
    echo Setup hasn't been run yet. Run setup.bat first.
    pause
    exit /b 1
)

echo ==^> Starting backend on http://localhost:8000 ...
start "Vital Signs - Backend" cmd /k "cd backend\app && ..\venv\Scripts\python main.py"

timeout /t 3 /nobreak >nul

echo ==^> Starting frontend on http://localhost:5500 ...
start "Vital Signs - Frontend" cmd /k "cd frontend && python -m http.server 5500"

timeout /t 2 /nobreak >nul

start http://localhost:5500

echo.
echo Both servers are running in their own windows.
echo To stop everything, just close those two windows.
pause
