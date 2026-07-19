@echo off
echo ==============================================
echo   Insurance Fraud Detection - Startup Script
echo ==============================================
echo.

echo Starting FastAPI Backend (Port 8000)...
start "FastAPI Backend" cmd /k "cd backend && ..\venv\Scripts\uvicorn.exe app:app --host 127.0.0.1 --port 8000 --reload"

echo Starting Frontend Web Server (Port 8080)...
start "Frontend Web Server" cmd /k "cd frontend && ..\venv\Scripts\python.exe -m http.server 8080"

echo.
echo Both servers are starting up!
echo The frontend will be available at: http://localhost:8080
echo The backend API is running at: http://127.0.0.1:8000
echo.
echo (Press any key to close this window. The servers will remain running in their own windows.)
pause >nul
