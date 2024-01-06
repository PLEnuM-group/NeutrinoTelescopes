#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=fisher
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH -o logs/fisher_%A-%a.out
#SBATCH -e logs/fisher_%A-%a.err

type=track
det=full
start=$(date +%s)
n_sims=20
out_folder=$WORK/fisher

cd ~/.julia/dev/NeutrinoTelescopes/ && /home/hpc/capn/capn100h/.juliaup/bin/julia --project=. scripts/fisher_information/calc_fisher_info.jl --outfile ${out_folder}/fisher_${type}_${det}_${SLURM_ARRAY_TASK_ID}.jld2 --type ${type} --det ${det} --nevents ${n_sims} || exit 1
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"