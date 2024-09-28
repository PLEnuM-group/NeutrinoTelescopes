#!/bin/bash -l
#
#SBATCH -n 32 
#SBATCH --time=24:00:00
#SBATCH --export=NONE

unset SLURM_EXPORT_ENV
export JULIA_WORKER_TIMEOUT=180
julia /home/saturn/capn/capn100h/julia_dev/NeutrinoTelescopes/scripts/surrogate_models/symbolic_regression/symbolic_regression.jl --config /home/saturn/capn/capn100h/julia_dev/NeutrinoTelescopes/scripts/surrogate_models/symbolic_regression/sr_run_e_dist_phi.toml 