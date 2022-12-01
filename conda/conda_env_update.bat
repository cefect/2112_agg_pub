call C:\LS\06_SOFT\conda\env\agg\activate.bat
cd %REPO%
echo on
call conda env update --file conda_env.yml --prune

pause