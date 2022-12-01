SET DS_L=(pre, post, preGW)
call C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\bats\setup.bat

REM execute
 
@echo on
FOR %%i IN %DS_L% DO (
python -O C:\LS\09_REPOS\02_JOBS\2112_agg\cef\agg\hydR\main.py -t %%i -n hr5 -i 8 -dsampStage %%i
)
ECHO finished 

pause

 


