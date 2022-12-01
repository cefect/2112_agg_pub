REM SET DS_L=(post, preGW, postFN, pre)
SET DS_L=(pre)
REM SET CAT_FP=C:\LS\10_OUT\2112_Agg\lib\hydR01\hydR01_run_index.csv -catalog_fp %CAT_FP%
SET LAG=5


REM execute
call C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\bats\setup.bat
@echo on
FOR %%i IN %DS_L% DO (
start cmd /k python -O C:\LS\09_REPOS\02_JOBS\2112_agg\cef\agg\hydR\main.py -t %%i -n hydR02 -i 8 -dsampStage %%i 
ping 127.0.0.1 -n %LAG% > nul
)
ECHO finished 

pause

 


