@echo off
setlocal EnableDelayedExpansion
title Synapic Launcher
color 0A

pushd "%~dp0" || (
    echo [ERROR] Could not change to script directory.
    pause
    exit /b 1
)

echo ===============================================================================
echo                                SYNAPIC LAUNCHER                                
echo ===============================================================================
echo.

if not exist "main.py" (
    echo [!] Synapic core files make sure 'main.py' is in the folder.
    echo [*] If this is a new installation, we can download the latest version.
    echo.
    echo Press any key to download Synapic from GitHub...
    pause >nul
    echo [*] Downloading latest code from https://github.com/deanable/Synapic...
    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/deanable/Synapic/archive/refs/heads/main.zip' -OutFile 'synapic.zip'"
    if exist "synapic.zip" (
        echo [*] Extracting repository...
        powershell -Command "Expand-Archive -Path 'synapic.zip' -DestinationPath '.'"
        if !errorlevel! neq 0 (
            echo [ERROR] Failed to extract 'synapic.zip'. The file might be corrupted.
            del "synapic.zip"
            pause
            exit /b 1
        )
        if exist "Synapic-main" (
            echo [*] Moving files to root...
            robocopy "Synapic-main" "." /E /IS /MOVE /XF "start_synapic.bat" >nul
            if !errorlevel! geq 8 (
                echo [!] Error during file move. Cleanup skipped.
            ) else (
                rmdir /S /Q "Synapic-main"
                del "synapic.zip"
                echo [V] Download and setup complete.
            )
        ) else (
            echo [!] Failed to confirm extracted folder. structure might have changed.
            pause
            exit /b 1
        )
    ) else (
        echo [ERROR] Failed to download 'synapic.zip'. Check internet connection.
        pause
        exit /b 1
    )
    echo.
)

if exist ".git" (
    echo [*] Checking for updates...
    set "_SKIP_UPDATE="
    where gh >nul 2>&1
    if !errorlevel! neq 0 (
        echo [!] GitHub CLI not found. Attempting to install...
        where winget >nul 2>&1
        if !errorlevel! equ 0 (
            winget install -e --id GitHub.cli --source winget --accept-package-agreements --accept-source-agreements >nul 2>&1
        )
        where gh >nul 2>&1
        if !errorlevel! neq 0 (
            echo [WARN] GitHub CLI still not found. Skipping version check.
            set "_SKIP_UPDATE=1"
        ) else (
            echo [V] GitHub CLI installed.
            if exist "%ProgramFiles%\GitHub CLI\gh.exe" set "PATH=%ProgramFiles%\GitHub CLI;%PATH%"
            if exist "%LocalAppData%\GitHub CLI\gh.exe" set "PATH=%LocalAppData%\GitHub CLI;%PATH%"
        )
    )
    if not defined _SKIP_UPDATE (
        gh auth status >nul 2>&1
        if !errorlevel! neq 0 (
            echo [WARN] GitHub CLI not authenticated. Skipping version check.
            set "_SKIP_UPDATE=1"
        ) else (
            echo [V] GitHub CLI authenticated.
        )
    )
    if not defined _SKIP_UPDATE (
        git fetch origin main >nul 2>&1
        if !errorlevel! neq 0 (
            echo [WARN] Could not fetch latest version. Skipping update check.
            set "_SKIP_UPDATE=1"
        )
    )
    if not defined _SKIP_UPDATE (
        git rev-list --count HEAD..origin/main >"%TEMP%\synapic_behind.txt" 2>nul
        set /p _BEHIND=<"%TEMP%\synapic_behind.txt"
        del "%TEMP%\synapic_behind.txt" 2>nul
        if not defined _BEHIND set _BEHIND=0
        if !_BEHIND! gtr 0 (
            echo [!] You are !_BEHIND! commit^(s^) behind the latest version.
            echo.
            choice /c YN /t 10 /d Y /n /m "Update now (Y/N, auto-pull in 10s)? "
            if !errorlevel! equ 1 (
                echo [*] Pulling latest changes...
                git pull origin main
                if !errorlevel! neq 0 (
                    echo [ERROR] Failed to pull latest changes.
                ) else (
                    echo [V] Updated to latest version.
                )
            ) else (
                echo [*] Update skipped.
            )
        ) else (
            echo [V] You are running the latest version.
        )
    )
) else (
    echo [*] Not a git repository - skipping version check.
    echo [*] Clone the repo from https://github.com/deanable/Synapic to enable updates.
)
echo.

echo [*] Checking for Python installation...
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [!] Python is NOT found. Attempting to install...
    where winget >nul 2>&1
    if !errorlevel! neq 0 (
        color 0C
        echo [ERROR] Winget is not available to auto-install Python.
        echo Please install Python manually from https://www.python.org/
        pause
        exit /b 1
    )
    echo [*] Installing Python via Winget...
    winget install -e --id Python.Python.3 --source winget --accept-package-agreements --accept-source-agreements
    if !errorlevel! neq 0 (
        color 0C
        echo [ERROR] Automatic installation failed.
        echo Please install Python manually from https://www.python.org/
        pause
        exit /b 1
    )
    echo [*] Python installed.
    echo [!] IMPORTANT: You must restart this script to complete the setup.
    pause
    exit /b 0
) else (
    echo [V] Python is installed:
    python --version
)

echo.
echo ===============================================================================
echo                          SETTING UP ENVIRONMENT                                
echo ===============================================================================
echo.

if not exist ".venv" (
    echo [*] Creating virtual environment .venv...
    python -m venv .venv
    if !errorlevel! neq 0 (
        echo [!] Failed to create venv. Will attempt to use system Python...
    ) else (
        echo [V] Virtual environment created.
    )
)

if exist ".venv" (
    echo [*] Activating virtual environment...
    if exist ".venv\Scripts\activate.bat" (
        call .venv\Scripts\activate.bat
        echo [V] Virtual environment activated.
    ) else (
        echo [!] .venv folder exists but activate.bat not found. Using system Python.
    )
)

echo.
echo [*] Checking dependencies...
if exist "requirements.txt" (
    echo [*] Installing/Updating requirements...
    pip install -r requirements.txt
    if !errorlevel! neq 0 (
        color 0C
        echo.
        echo [ERROR] Failed to install dependencies.
        echo Please check your internet connection or requirements.txt file.
        pause
        exit /b 1
    )
    echo [V] Dependencies installed/updated.
) else (
    echo [!] requirements.txt not found! Skipping dependency installation.
)

echo.
echo ===============================================================================
echo                               LAUNCHING APP                                    
echo ===============================================================================
echo.

if exist "main.py" (
    echo [*] Launching Synapic...
    if exist ".venv\Scripts\pythonw.exe" (
        start /b "" .venv\Scripts\pythonw.exe "main.py"
    ) else (
        start /b "" pythonw "main.py"
    )
    echo [*] Application launched. Closing launcher window...
    timeout /t 2 /nobreak >nul
) else (
    color 0C
    echo [ERROR] main.py not found in current directory!
    pause
    exit /b 1
)

popd
endlocal
exit
