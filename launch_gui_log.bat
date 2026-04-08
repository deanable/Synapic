@echo off
setlocal
set "LOGFILE=%~dp0launch_gui.log"
echo Launch started: %DATE% %TIME% >> "%LOGFILE%"
call "%~dp0\launch_gui.bat" >> "%LOGFILE%" 2>&1
echo Launch exited with code %ERRORLEVEL% >> "%LOGFILE%"
endlocal
exit /b %ERRORLEVEL%
