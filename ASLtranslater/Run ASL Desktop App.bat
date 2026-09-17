@echo off
cd /d "%~dp0"
python "asl_desktop_app.py"
if errorlevel 1 pause
