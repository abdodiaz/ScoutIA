@echo off
echo Running ScoutIA Pro Tests...
echo.
.\venv\Scripts\python.exe -m pytest tests/ -v
echo.
pause

