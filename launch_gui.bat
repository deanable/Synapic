@echo off
setlocal
cd /d "%~dp0"

REM Detect Python (prefer py launcher)
set "PY_CMD="
for %%P in (py.exe py python.exe python) do (
  where %%P >nul 2>nul
  if not errorlevel 1 set "PY_CMD=%%P"
  if defined PY_CMD goto :found_py
)
:found_py
if not defined PY_CMD (
  echo Python 3 is required but was not found on PATH.
  pause
  exit /b 1
)

REM Create venv if needed
if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  %PY_CMD% -m venv .venv
  if errorlevel 1 goto :fail
)

REM Activate venv (Windows)
call .venv\Scripts\activate.bat

REM Install dependencies if needed
if exist requirements.txt (
  .venv\Scripts\pip.exe install -r requirements.txt
  if errorlevel 1 goto :fail
)

REM Launch GUI
if exist ".venv\Scripts\pythonw.exe" (
  start "Synapic" ".venv\Scripts\pythonw.exe" main.py --gui
) else (
  start "Synapic" ".venv\Scripts\python.exe" main.py --gui
)
exit /b 0

:fail
echo. & echo Short Machine launcher failed. See logs for details.
pause
exit /b 1
