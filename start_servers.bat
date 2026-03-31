@echo off
taskkill /f /im streamlit.exe 2>nul
timeout /t 2 /nobreak >nul
start "Admin Server (8501)" cmd /k "call C:\Users\koh\anaconda3\Scripts\activate.bat pppd && cd /d %~dp0 && streamlit run admin_app.py --server.port 8501 --server.headless true"
start "Student Server (8502)" cmd /k "call C:\Users\koh\anaconda3\Scripts\activate.bat pppd && cd /d %~dp0 && streamlit run student_app.py --server.port 8502 --server.headless true"
echo Admin : http://localhost:8501
echo Student: http://localhost:8502
pause
