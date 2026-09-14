@echo off
cd /d "%~dp0"
python "new_project.py"
if errorlevel 1 pause
