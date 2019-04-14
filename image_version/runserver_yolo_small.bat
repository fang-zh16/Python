@echo off
cd %cd%\analyzers\image_analyzer
start python image_analyzer.py
cd ..
cd %cd%\record_analyzer
start python record_analyzer.py
cd ..
cd ..
start python intelligent_monitoring_platform/manage.py runserver

start http://127.0.0.1:8000/imp/