@echo off
echo Starting ScoutIA Pro Dashboard...
echo.
echo Opening: http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.
.\venv\Scripts\python.exe -m streamlit run frontend/streamlit_app.py --server.port 8501
pause

