@echo off
setlocal

set "ROOT=%~dp0"
set "BACKEND_DIR=%ROOT%backend"
set "FRONTEND_DIR=%ROOT%frontend"
set "NODE_PATH=C:\Program Files\nodejs"
set "CONDA_ACT=C:\Users\koh\anaconda3\Scripts\activate.bat"
set "CONDA_ENV=pppd"
set "BACKEND_RUN=%TEMP%\_pppd_be.bat"
set "FRONTEND_RUN=%TEMP%\_pppd_fe.bat"

echo.
echo =====================================================
echo  PPPD Dev Server Launcher
echo =====================================================
echo.

REM --- Check backend .env -------------------------------------------------------
if not exist "%BACKEND_DIR%\.env" (
    echo [ERROR] backend\.env not found.
    echo         Copy .env.example to .env and set GEMINI_API_KEY.
    echo.
    pause
    exit /b 1
)

REM --- Auto-create frontend .env.local ------------------------------------------
if not exist "%FRONTEND_DIR%\.env.local" (
    copy "%FRONTEND_DIR%\.env.local.example" "%FRONTEND_DIR%\.env.local" > nul
    echo [INFO] Created frontend\.env.local  ^(NEXT_PUBLIC_API_BASE_URL=http://localhost:8000^)
    echo.
)

REM --- npm install if node_modules missing --------------------------------------
if not exist "%FRONTEND_DIR%\node_modules" (
    echo [INFO] node_modules missing - running npm install...
    pushd "%FRONTEND_DIR%"
    set "PATH=%NODE_PATH%;%PATH%"
    call npm install
    popd
    if errorlevel 1 (
        echo [ERROR] npm install failed. Check Node.js at: %NODE_PATH%
        pause
        exit /b 1
    )
    echo.
)

REM --- Kill existing processes on ports 8000 / 3000 ----------------------------
echo Cleaning up ports 8000 and 3000...
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr ":8000 " ^| findstr "LISTENING"') do (
    taskkill /pid %%p /f >nul 2>&1
)
for /f "tokens=5" %%p in ('netstat -ano 2^>nul ^| findstr ":3000 " ^| findstr "LISTENING"') do (
    taskkill /pid %%p /f >nul 2>&1
)
timeout /t 1 /nobreak > nul

REM --- Write backend runner to TEMP (avoids Korean-path quote issues) -----------
(
    echo @echo off
    echo title Backend - FastAPI :8000
    echo call "%CONDA_ACT%" %CONDA_ENV%
    echo cd /d "%BACKEND_DIR%"
    echo uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    echo pause
) > "%BACKEND_RUN%"

REM --- Write frontend runner to TEMP -------------------------------------------
(
    echo @echo off
    echo title Frontend - Next.js :3000
    echo set PATH=%NODE_PATH%;%%PATH%%
    echo cd /d "%FRONTEND_DIR%"
    echo npm run dev
    echo pause
) > "%FRONTEND_RUN%"

REM --- Launch servers in separate windows ---------------------------------------
echo Starting Backend  (FastAPI  port 8000^)...
start "Backend - FastAPI :8000"  cmd /k "%BACKEND_RUN%"
timeout /t 3 /nobreak > nul

echo Starting Frontend (Next.js  port 3000^)...
start "Frontend - Next.js :3000"  cmd /k "%FRONTEND_RUN%"

echo.
echo =====================================================
echo  Backend  (FastAPI) : http://localhost:8000
echo  API Docs (Swagger) : http://localhost:8000/docs
echo  Frontend (Next.js) : http://localhost:3000
echo.
echo  Close each server window to stop that server.
echo  Run start_ngrok.bat for external access.
echo =====================================================
echo.
pause
endlocal
