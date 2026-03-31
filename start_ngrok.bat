@echo off
taskkill /f /im ngrok.exe 2>nul
timeout /t 1 /nobreak >nul
ngrok start --config "%LOCALAPPDATA%\ngrok\ngrok.yml" --config "%~dp0ngrok.yml" admin student
pause
