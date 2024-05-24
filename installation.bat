::
::
:: 
@echo off

set /p conda_env_name=Input the name of a conda virtual environment for test codes: 
echo The name of rhe conda virtual environment: %conda_env_name%

call conda update -n base conda -y
call conda update --all -y
call python -m pip install --upgrade pip 

call conda create --name %conda_env_name% python=3.11.4 -y

call conda activate %conda_env_name%

call pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
call conda install tensorboard matplotlib pyyaml tqdm scipy conda-forge::pandas conda-forge::torchinfo -y
call pip3 install wget

echo =========================
echo Installation Finished!!
echo =========================