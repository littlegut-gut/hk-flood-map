@echo off
cd /d D:\hk_flood_map
echo. >> data\task_scheduler.log
echo === %date% %time% === >> data\task_scheduler.log
venv\Scripts\python.exe accumulate_rainfall.py >> data\task_scheduler.log 2>&1
echo Exit code: %errorlevel% >> data\task_scheduler.log