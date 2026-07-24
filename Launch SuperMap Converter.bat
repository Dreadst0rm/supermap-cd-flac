@echo off
title SuperMap Converter
cd /d "%~dp0"
python -m supermap_cd gui
if errorlevel 1 (
  echo.
  echo If that failed, install first:
  echo   python -m pip install -e .
  echo.
  pause
)
