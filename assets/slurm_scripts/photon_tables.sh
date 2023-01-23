#!/bin/bash

#SBATCH --job-name=photon_tables                
#SBATCH --ntasks=1                              
#SBATCH --time=12:00:00                         
#SBATCH --partition=lrz-v100x2
#SBATCH --gres=gpu:1

n_sims=1000
n_skip = ${SLURM_ARRAY_TASK_ID} * $n_sims
n_resample = 20

cd ~/repos/NeutrinoTelescopes/ && ~/.julia/juliaup/bin/julia --project=. scripts/photon_tables.jl --n_sims=$n_sims --n_skip=$n_skip--n_resample=$n_resample --output data/photon_table_extended_${SLURM_ARRAY_TASK_ID}.hd5 --dist_min=5 --dist_max=200 --mode extended --seed ${SLURM_ARRAY_TASK_ID}
