#!/bin/bash

#SBATCH --job-name=LundNet_training 
#SBATCH --account=gpu_gres               # to access gpu resources
#SBATCH --partition=gpu                                           
#SBATCH --nodes=1                        # request to run job on single node                                       
#SBATCH --ntasks=5                     # request 10 CPU's (t3gpu01/02: balance between CPU and GPU : 5CPU/1GPU)      
#SBATCH --gres=gpu:1                     # request  for two GPU's on machine, this is total  amount of GPUs for job        
#SBATCH --mem=12000M                     # memory (per job)
#SBATCH --time=00-3:00                   # time  in format DD-HH:MM
#SBATCH --gres-flags=disable-binding
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

mkdir -p /scratch/$USER/${SLURM_JOB_ID}
export TMPDIR=/scratch/$USER/${SLURM_JOB_ID}

python TrainLundNet.py --epochs 500 --batch 128 --name "LundNet5+_SVJ-QCD" --model "LN5p" --lr 0.001

rm  -rf /scratch/$USER/${SLURM_JOB_ID}
