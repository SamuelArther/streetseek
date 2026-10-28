@echo off
rem Starts the Streetseek web UI and opens it in your browser once it's ready.
cd /d "%~dp0"
python -m streetseek serve --open
rem Keep the window open if it crashed so the error can be read.
if errorlevel 1 pause
