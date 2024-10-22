#!/bin/bash

#SBATCH --job-name=Tagger
#SBATCH --partition=short
#SBATCH --account=t3
#SBATCH --nodes=1                        # request to run job on single node                                       
#SBATCH --ntasks=10                     # request 10 CPU's (t3gpu01/02: balance between CPU and GPU : 5CPU/1GPU)   
#SBATCH --mem=20000M                     # memory (per job)
#SBATCH --time=00-00:30                   # time  in format DD-HH:MM
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

mkdir -p /scratch/$USER/${SLURM_JOB_ID}
export TMPDIR=/scratch/$USER/${SLURM_JOB_ID}

python ProduceMTDistribution.py --name "LundNet5+_SVJ-QCD-TTJet-EW_vs12" --model "LN5p"

rm  -rf /scratch/$USER/${SLURM_JOB_ID}