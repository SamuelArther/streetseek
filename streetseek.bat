@echo off
rem Starts the Streetseek web UI and opens it in your browser.
cd /d "%~dp0"
start "" http://127.0.0.1:8765
python -m streetseek serve
