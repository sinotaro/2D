@echo off
rem Drag and drop an image (or a folder of images) onto this file.
if "%~1"=="" (
  echo Usage: drag an image onto run.bat, or: run.bat C:\path\to\character.png
  pause
  exit /b 1
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" -Image "%~1"
pause
