REM generic setup ufor hyd.model runs
@echo off
REM setup pyqgis
call "C:\LS\06_SOFT\conda\env\agg\activate.bat"

REM set the assocated projects
set PYTHONPATH=%PYTHONPATH%;C:\LS\09_REPOS\02_JOBS\2112_Agg\cef;C:\LS\09_REPOS\01_COMMON\coms