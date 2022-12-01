REM paramters
REM SET ids=(90,91,92)
SET START=110
SET COUNT=9

SET NAME=hyd7

call C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\bats\hyd_setup.bat

REM execute
SET /a "END=%START%+%COUNT%-1"
@echo on
FOR /l %%i IN (%START%,1,%END%) DO (
python -O C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\agg\hyd\main.py %%i -n %NAME% 
)
@echo off
FOR /l %%i IN (%START%,1,%END%) DO (
ECHO finished %NAME% %%i
)
pause

 


