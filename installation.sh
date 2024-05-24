#!/bin/bash
# Store the name of a conda virtual environment

echo "Input the name of a conda virtual environment for test codes: "
read conda_env_name

echo "The name of rhe conda virtual environment: $conda_env_name"

conda update -n base conda -y
conda update --all -y
python -m pip install --upgrade pip 

conda create --name $conda_env_name python=3.11.4 

conda activate $conda_env_name

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install tensorboard matplotlib pyyaml scipy tqdm conda-forge::pandas conda-forge::torchinfo -y
pip3 install wget
 
#conda deactivate
#source ~/anaconda3/etc/profile.d/conda.sh


