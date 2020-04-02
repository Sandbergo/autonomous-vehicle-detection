#!/bin/bash

# This script assumes that you have created a work directory with your default username
# ex: "/work/olavlp/", If another folder name was created pass it as the first argument
# when running this script. It also assumes the repo is in the top directory of your
# working folder.

# If a change is made to the TermProject environment, the environment on your machine can be run using:
# 'conda env update -f TermProject.yaml -n TermProject'

# Usage
# ./install_conda_env.sh [<directory-in-/work/..>]

# Setting work_dir folder
if [ -z "$1" ]; then
    user=$(whoami)
    echo "No argument passed, using default: $user"
else
    user=$1
    echo "using passed folder name: $user"
fi

# Checking to see if anaconda folder exist and if not will install anaconda
if [[ ! -d /work/$user/anaconda ]]; then
    echo "The directory anaconda doesn't exists, installing anaconda to /work/$user/anaconda"

    # Downloading Anaconda
    if [[ ! -f ~/Downloads/Anaconda3-2020.02-Linux-x86_64.sh ]]; then
        echo "### --------------------- Downloading Anaconda --------------------- ###"
        cd ~/Downloads
        wget https://repo.anaconda.com/archive/Anaconda3-2020.02-Linux-x86_64.sh
    fi

    # This installs anaconda in silent mode to the work directory
    #mkdir /work/$user/anaconda
    cd ~/Downloads
    echo "### --------------------- Installing Anaconda --------------------- ###"
    chmod +x Anaconda3-2020.02-Linux-x86_64.sh
    ./Anaconda3-2020.02-Linux-x86_64.sh -b -p /work/$user/anaconda
    source /work/$user/anaconda/etc/profile.d/conda.sh
    # eval "$(/work/$user/anaconda/bin/conda shell.bash hook)"

    # This adds conda path to your shells "--rc" file.
    conda init
else
    echo "The directory /work/$user/anaconda/ already exists, not installing anaconda"
fi

# Create a symbolic link from github repo .yaml file so that any changes to the yaml file can update environment
if [[ ! -d /work/$user/anaconda/envs/TermProject ]]; then
    mkdir -p /work/$user/anaconda/envs/TermProject
fi
cd /work/$user/anaconda/envs/TermProject
ln -s /work/$user/autonomous-vehicle-detection/anaconda_setup/TermProject.yaml
conda env create -f TermProject.yaml
conda activate TermProject

# Downloading the datasets
echo "### --------------------- Updating Datsets --------------------- ###"
cd /work/$user/autonomous-vehicle-detection/SSD/
python3 setup_waymo.py
python3 update_tdt4265_dataset.py
