call activate.bat

echo on
call conda env export >%REPO%\conda\conda_env.yml

pause