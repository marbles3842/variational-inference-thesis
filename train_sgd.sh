#!/bin/sh 
#BSUB -q gpuv100
#BSUB -J bl_sgd
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB
#BSUB -W 24:00 
#BSUB -o /zhome/da/9/204020/variational-inference-thesis/out/%J.out 
#BSUB -e /zhome/da/9/204020/variational-inference-thesis/out/%J.err 

VENV_NAME=venv         
VENV_DIR=/zhome/da/9/204020            
PYTHON_VERSION=3.12.11  
PROJECT_DIR=/zhome/da/9/204020/variational-inference-thesis

rm -rf "${PROJECT_DIR}/out"
mkdir -p "${PROJECT_DIR}/out"

module purge
module load python3/$PYTHON_VERSION
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "numpy/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "scipy/")
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "matplotlib/")

source "${VENV_DIR}/${VENV_NAME}/bin/activate"

python -m baseline.train_sgd
