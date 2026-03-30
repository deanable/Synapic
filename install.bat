@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ------------------------------------------------------------
REM install.bat
REM Windows installer for Synapic with optional silent mode.
REM Usage: install.bat [-s]
REM  -s  Run in silent mode (no on-screen prompts)
REM ------------------------------------------------------------

set SILENT=0

:parse_args
if "%~1"=="" goto after_args
if /I "%~1"=="-s" set SILENT=1
shift
goto parse_args
:after_args

REM Helper log function (quiet in silent mode)
if %SILENT%==0 (
  echo [install] Starting Synapic setup...
)

REM Step 1: Ensure Python is available
where python.exe >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
  if %SILENT%==0 (
    echo [install] Python not found. Attempting silent install of Python 3.11.5 64-bit...
  )
  set PY_URL=https://www.python.org/ftp/python/3.11.5/python-3.11.5-amd64.exe
  set PY_INSTALLER=%TEMP%\python_installer.exe
  powershell -NoProfile -Command "Invoke-WebRequest -Uri '%PY_URL%' -OutFile '%PY_INSTALLER%'" >nul 2>&1
  if exist "%PY_INSTALLER%" (
    start /wait "Python Installer" "%PY_INSTALLER%" /quiet InstallAllUsers=1 PrependPath=1 TargetDir=C:\\Python311
    if exist "C:\\Python311\\python.exe" (
      set PATH=%PATH%;C:\\Python311;C:\\Python311\\Scripts
    )
  ) else (
    if %SILENT%==0 (
      echo [install] Failed to download Python; exiting.
    )
    exit /b 1
  )
) else (
  if %SILENT%==0 (
    echo [install] Python found on PATH.
  )
)

REM Refresh PATH for current session if Python was installed just now
set PYPATH=%PROGRAMFILES%\Python311\;C:\\Python311;C:\\Python311\\Scripts
set PATH=%PATH%;C:\\Python311;C:\\Python311\\Scripts

REM Step 2: Get source code (clone if missing, else update)
if exist ".git" (
  if %SILENT%==0 (
    echo [install] Updating Synapic repository...
  )
  git fetch --all
  git reset --hard origin/main
) else (
  if %SILENT%==0 (
    echo [install] Cloning Synapic repository...
  )
  git clone https://github.com/Dean-Kruger/Synapic.git .
)

REM Step 3: Setup Python virtual environment and install deps
if not exist ".venv" (
  if %SILENT%==0 (
    echo [install] Creating virtual environment...\n
  )
  call %SystemRoot%\\System32\\cmd.exe /c ""%LOCALAPPDATA%\\Programs\\Python\\Python39\\python.exe" -m venv .venv" 2>NUL
  if not exist ".venv" (
    REM Fallback to Python311 path if available
    if exist "C:\\Python311\\python.exe" (
      "C:\\Python311\\python.exe" -m venv .venv
    ) else (
      if %SILENT%==0 (
        echo [install] Failed to create virtual environment.
      )
      exit /b 1
    )
  )
)

REM Activate venv and install requirements
call .venv\\Scripts\\activate.bat
if %ERRORLEVEL% NEQ 0 (
  if %SILENT%==0 (
    echo [install] Failed to activate virtual environment.
  )
  exit /b 1
)

if exist requirements.txt (
  if %SILENT%==0 (
    echo [install] Installing dependencies from requirements.txt...
  )
  .venv\\Scripts\\pip.exe install -r requirements.txt >nul 2>&1
  if %ERRORLEVEL% NEQ 0 (
    if %SILENT%==0 (
      echo [install] Failed to install dependencies from requirements.txt. Attempting to install pytest directly...
    )
  )
  .venv\\Scripts\\pip.exe install pytest >nul 2>&1
  if %ERRORLEVEL% NEQ 0 (
    if %SILENT%==0 (
      echo [install] Dependency installation failed.
    )
    exit /b 1
  )
)

REM Step 4: Run unit tests
if exist tests\\unit (
  if %SILENT%==0 (
    echo [install] Running unit tests...
  )
  pytest tests\\unit -q
  if %ERRORLEVEL% NEQ 0 (
    if %SILENT%==0 (
      echo [install] Unit tests failed.
    )
    exit /b 1
  )
)

echo [install] Setup complete.
exit /b 0
