#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=photon_tables
#SBATCH --ntasks=1
#SBATCH --time=4:00:00
#SBATCH --partition=lrz-v100x2,lrz-dgx-a100-80x8,lrz-dgx-1-p100x8
#SBATCH --gres=gpu:1
#SBATCH -o bare_infinite.out
#SBATCH -e bare_infinite.err

n_sims=500
n_skip=$((SLURM_ARRAY_TASK_ID * n_sims))
n_resample=20

start=$(date +%s)
cd ~/repos/NeutrinoTelescopes/ && ~/.julia/juliaup/bin/julia --project=. scripts/photon_tables.jl --n_sims=${n_sims} --n_skip=${n_skip} --n_resample=${n_resample} --output data/photon_table_bare_infinite_${SLURM_ARRAY_TASK_ID}.hd5 --dist_min=1 --dist_max=200 --mode bare_infinite_track
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"