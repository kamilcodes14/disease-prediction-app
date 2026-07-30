@echo off
echo ==^> Creating virtual environment...
python -m venv backend\venv
if errorlevel 1 (
    echo.
    echo Something went wrong creating the virtual environment.
    echo Make sure Python is installed and added to PATH, then try again.
    pause
    exit /b 1
)

echo ==^> Installing backend dependencies...
backend\venv\Scripts\pip install --upgrade pip -q
backend\venv\Scripts\pip install -r backend\requirements.txt -q

echo ==^> Training the structured-data models (Heart Disease / Diabetes / Breast Cancer)...
cd backend\app
..\venv\Scripts\python train_structured.py
echo ==^> Training the photo-screening model...
..\venv\Scripts\python train_image_model.py
cd ..\..

echo.
echo Setup complete. Now run: run.bat
pause
