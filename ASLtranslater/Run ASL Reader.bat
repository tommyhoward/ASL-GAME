@echo off
cd /d "%~dp0"
python "camera_reader.py"
if errorlevel 1 pause
