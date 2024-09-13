#!/bin/bash -l
#
#SBATCH --gres=gpu:a40:8
#SBATCH --time=8:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV

srun /home/hpc/capn/capn100h/.juliaup/bin/julia train_time_surrogate_hopt.jl
