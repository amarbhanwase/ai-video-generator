@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo Starting AI Video Generator Server...
uvicorn main:app --host 127.0.0.1 --port 8000

pause
