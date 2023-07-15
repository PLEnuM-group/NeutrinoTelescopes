#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=photon_tables
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH -o extended_%A-%a.out
#SBATCH -e extended_%A-%a.err

n_sims=1000
n_skip=$((SLURM_ARRAY_TASK_ID * n_sims))

start=$(date +%s)

out_folder=$WORK/photon_tables/extended/

cd ~/.julia/dev/NeutrinoTelescopes/ && /home/hpc/capn/capn100h/.juliaup/bin/julia --project=. scripts/photon_tables/photon_tables_photons.jl --n_sims=${n_sims} --n_skip=${n_skip} --output $out_folder/photon_table_extended_${SLURM_ARRAY_TASK_ID}.hd5 --dist_min=5 --dist_max=200 --mode extended || exit 1
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"