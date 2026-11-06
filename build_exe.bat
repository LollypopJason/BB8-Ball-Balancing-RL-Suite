@echo off
REM ============================================================================
REM  build_exe.bat - build "BB8_RL_Studio.exe" (Windows 10/11, 64-bit)
REM
REM  Usage (double-click, or from a terminal in this folder):
REM     build_exe.bat              folder build  -> dist\BB8_RL_Studio\BB8_RL_Studio.exe  (recommended, fast start)
REM     build_exe.bat onefile      single exe    -> dist\BB8_RL_Studio.exe  (big, slower start)
REM     build_exe.bat console      folder build with a console window for logs / debugging
REM     build_exe.bat onefile console
REM     build_exe.bat notest       skip the post-build self-test of the .exe
REM
REM  Requires 64-bit Python 3.10 - 3.12 from python.org (tick "Add python.exe to PATH").
REM  First run downloads ~1 GB of packages (PyTorch, MuJoCo, Qt) into .venv-build\
REM ============================================================================
setlocal EnableExtensions
cd /d "%~dp0"

set "BB8_ONEFILE=0"
set "BB8_CONSOLE=0"
set "BB8_TEST=1"
for %%A in (%*) do (
    if /I "%%~A"=="notest" set "BB8_TEST=0"
    if /I "%%~A"=="onefile" set "BB8_ONEFILE=1"
    if /I "%%~A"=="console" set "BB8_CONSOLE=1"
)

REM ---- find Python (prefer the py launcher) ----------------------------------
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY (
    where python >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo [ERROR] Python 3 not found. Install 64-bit Python 3.11 from https://www.python.org/downloads/
    echo         and tick "Add python.exe to PATH".
    goto :fail
)
%PY% -c "import sys,struct; assert sys.version_info[:2]>=(3,10) and struct.calcsize('P')==8" >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Need 64-bit Python 3.10 or newer. Found:
    %PY% --version
    goto :fail
)

REM ---- isolated build environment --------------------------------------------
if not exist ".venv-build\Scripts\python.exe" (
    echo [1/5] Creating build virtual environment .venv-build ...
    %PY% -m venv .venv-build || goto :fail
)
set "VPY=.venv-build\Scripts\python.exe"

echo [2/5] Installing dependencies (first run can take several minutes) ...
"%VPY%" -m pip install --upgrade pip wheel || goto :fail
"%VPY%" -m pip install -r requirements.txt || goto :fail
"%VPY%" -m pip install "pyinstaller>=6.6" || goto :fail

echo [3/5] Quick self-test (imports + physics + ONNX) ...
"%VPY%" -c "import mujoco, torch, stable_baselines3, onnx, onnxruntime, PySide6, pyqtgraph; from bb8_gym_env import BB8BalanceEnv; e=BB8BalanceEnv(); e.reset(seed=0); e.step(e.action_space.sample()); print('self-test OK, mujoco', mujoco.__version__)" || goto :fail

echo [4/5] Building executable (onefile=%BB8_ONEFILE%, console=%BB8_CONSOLE%) ...
"%VPY%" -m PyInstaller --noconfirm --clean bb8_studio.spec || goto :fail

if "%BB8_ONEFILE%"=="1" (set "EXE=dist\BB8_RL_Studio.exe") else (set "EXE=dist\BB8_RL_Studio\BB8_RL_Studio.exe")
if "%BB8_TEST%"=="1" (
    echo [5/5] Self-test of the packaged exe: physics, ONNX/C++ export, multi-process training ...
    start "" /wait "%EXE%" --self-test --workspace "%TEMP%\bb8_selftest"
    if errorlevel 1 (
        echo [ERROR] The packaged exe failed its self-test. Rebuild with "build_exe.bat console" and run
        echo         "%EXE% --self-test" in a terminal to see the traceback.
        goto :fail
    )
    echo       self-test passed.
) else (
    echo [5/5] Self-test skipped.
)

echo.
echo ============================================================================
if "%BB8_ONEFILE%"=="1" (
    echo  Done:  dist\BB8_RL_Studio.exe
) else (
    echo  Done:  dist\BB8_RL_Studio\BB8_RL_Studio.exe
    echo  Ship / zip the whole dist\BB8_RL_Studio folder.
)
echo  The app creates its "workspace" folder next to the .exe.
echo ============================================================================
endlocal
exit /b 0

:fail
echo.
echo [BUILD FAILED] See the messages above.
endlocal
pause
exit /b 1
