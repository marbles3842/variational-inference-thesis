#!/bin/sh 
#BSUB -q gpuv100
#BSUB -J bl_ivon
#BSUB -n 4
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -R "rusage[mem=4GB]"
#BSUB -M 5GB
#BSUB -W 24:00 
#BSUB -o /zhome/da/9/204020/variational-inference-thesis/out/jobs/ivon/%J.out 
#BSUB -e /zhome/da/9/204020/variational-inference-thesis/out/jobs/ivon/%J.err 

VENV_NAME=venv         
VENV_DIR=/zhome/da/9/204020            
PYTHON_VERSION=3.12.11  
PROJECT_DIR=/zhome/da/9/204020/variational-inference-thesis

MODEL_NAME=resnet20
SEED=0

CHECKPOINT_DIR="${PROJECT_DIR}/checkpoints/ivon/${LSB_JOBID}"

mkdir -p "${PROJECT_DIR}/out/jobs/ivon"

module purge
module load python3/$PYTHON_VERSION
module load $(module avail -o modulepath -t -C "python-${PYTHON_VERSION}" 2>&1 | grep "numpy/")

source "${VENV_DIR}/${VENV_NAME}/bin/activate"

python -m baseline.train_ivon --seed "$SEED"  --job-id "$LSB_JOBID" --model-name "$MODEL_NAME"

python -m baseline.test_ivon --seed "$SEED"  --last-checkpoint "$CHECKPOINT_DIR" --job-id "$LSB_JOBID" --model-name "$MODEL_NAME"