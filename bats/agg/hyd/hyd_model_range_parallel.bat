REM paramters
SET ids=(1,2)
REM ,)

SET LAG=1
SET NAME=hyd7

call C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\bats\hyd_setup.bat

REM execute (using ping command to add some lag between calls)
@echo on
FOR %%i IN %ids% DO (
start cmd /k python -O C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\agg\hyd\main.py %%i -n %NAME%
ping 127.0.0.1 -n %LAG% > nul
)




 


