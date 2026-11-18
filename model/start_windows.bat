@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv
    if errorlevel 1 (
        echo Install 64-bit Python 3.11 first, or create the environment with Python 3.12.
        pause
        exit /b 1
    )
)
if not exist ".venv\.fencing_setup_complete" (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :failed
    ".venv\Scripts\python.exe" -m pip install -e .
    if errorlevel 1 goto :failed
    type nul > ".venv\.fencing_setup_complete"
)
if not exist "models\pose_landmarker_full.task" (
    ".venv\Scripts\python.exe" tools\download_pose_model.py
    if errorlevel 1 goto :failed
)
".venv\Scripts\python.exe" -m streamlit run app.py
if errorlevel 1 goto :failed
exit /b 0
:failed
echo The application could not start. Read the error above and README.md.
pause
exit /b 1
