@echo off
REM Check if virtual environment folder "env" exists, if not create one
if not exist env (
    echo Creating virtual environment...
    python -m venv env
)

REM Activate the virtual environment
call env\Scripts\activate

REM Upgrade pip
echo Upgrading pip...
pip install --upgrade pip

REM Install required dependencies
echo Installing dependencies...
pip install fastapi uvicorn jinja2 torch torchaudio soundfile transformers streamlit requests

REM Start uvicorn in a new command window
echo Starting FastAPI (uvicorn)...
start "Uvicorn" cmd /k "uvicorn app:app --host 0.0.0.0 --port 8000"

REM Start Streamlit in a new command window
echo Starting Streamlit app...
start "Streamlit" cmd /k "streamlit run streamlit_app.py"

echo All services started.
pause
