@echo off
echo Starting ScoutIA Pro API Server...
echo.
echo API: http://localhost:8000
echo Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop the server
echo.
cd backend
..\venv\Scripts\python.exe -m uvicorn main:app --reload --port 8000
pause

