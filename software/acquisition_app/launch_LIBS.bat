@echo off
REM ── Launch LIBS Application in conda base environment ──

REM Auto-detect Anaconda install path
set "CONDA_PATH="
if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    set "CONDA_PATH=%USERPROFILE%\anaconda3"
) else if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    set "CONDA_PATH=%USERPROFILE%\miniconda3"
) else if exist "C:\ProgramData\anaconda3\Scripts\activate.bat" (
    set "CONDA_PATH=C:\ProgramData\anaconda3"
) else if exist "C:\ProgramData\miniconda3\Scripts\activate.bat" (
    set "CONDA_PATH=C:\ProgramData\miniconda3"
)

if "%CONDA_PATH%"=="" (
    echo ERROR: Could not find Anaconda or Miniconda installation.
    echo Looked in:
    echo   %USERPROFILE%\anaconda3
    echo   %USERPROFILE%\miniconda3
    echo   C:\ProgramData\anaconda3
    echo   C:\ProgramData\miniconda3
    pause
    exit /b 1
)

REM Activate conda base environment
call "%CONDA_PATH%\Scripts\activate.bat" base

REM Change to the application directory
cd /d "%~dp0"

REM Try IPython first (interactive console stays open after app exits/crashes)
where ipython >nul 2>&1
if %errorlevel%==0 (
    echo Starting LIBS App via IPython...
    REM ipython -c "run Bruniquel_LIBS_improved.py" -i
    ipython -c "run Bruniquel_LIBS_improved.py"
) else (
    echo WARNING: IPython not found, falling back to plain python.
    echo Install it with: conda install ipython
    echo.
    python Bruniquel_LIBS_improved.py
    if errorlevel 1 (
        echo.
        echo ── Application exited with an error. Press any key to close. ──
        pause >nul
    )
)
