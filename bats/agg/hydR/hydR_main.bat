 
call C:\LS\09_REPOS\02_JOBS\2112_Agg\cef\bats\setup.bat

REM SET CAT_FP=C:\LS\10_OUT\2112_Agg\lib\hydR01\hydR01_run_index.csv -catalog_fp %CAT_FP%

REM execute
 
@echo on
 
python -O C:\LS\09_REPOS\02_JOBS\2112_agg\cef\agg\hydR\main.py -t post_Sins_nn -n hydR03 -i 8 -dsampStage post -downSampling nn -sequenceType inputs 
 
ECHO finished 

pause

 


