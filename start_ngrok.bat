@echo off
echo.
echo =====================================================
echo  ngrok Tunnel Launcher (frontend:3000, backend:8000)
echo =====================================================
echo.
echo [1/2] Stopping existing ngrok processes...
taskkill /f /im ngrok.exe >nul 2>&1
timeout /t 1 /nobreak > nul
echo [2/2] Starting tunnels...
ngrok start --config "%LOCALAPPDATA%\ngrok\ngrok.yml" --config "%~dp0ngrok.yml" frontend backend
pause
