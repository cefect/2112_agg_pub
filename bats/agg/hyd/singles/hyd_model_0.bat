REM paramters
SET modelID=0
SET NAME=hyd4
@echo off

call C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\bats\hyd_setup.bat

REM execute
@echo on
cmd.exe /k python C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\agg\hyd\main.py %modelID% -n %NAME%