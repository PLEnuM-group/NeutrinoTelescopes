#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=photon_tables
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/extended_highE_%A-%a.out
#SBATCH -e logs/extended_highE_%A-%a.err

n_sims=500
n_skip=$((SLURM_ARRAY_TASK_ID * n_sims))
out_folder=$WORK/photon_tables/extended/

start=$(date +%s)
cd ~/.julia/dev/NeutrinoTelescopes/ && /home/hpc/capn/capn100h/.juliaup/bin/julia --project=. scripts/photon_tables/photon_tables_photons.jl --n_sims=${n_sims} --n_skip=${n_skip} --output $out_folder/photon_table_extended_${SLURM_ARRAY_TASK_ID}_highE.hd5 --dist_min=5 --dist_max=200 --mode extended || exit 1
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"