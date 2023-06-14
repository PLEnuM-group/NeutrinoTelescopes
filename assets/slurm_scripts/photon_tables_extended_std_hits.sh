#!/bin/bash
#SBATCH -N 1
#SBATCH --job-name=photon_tables_hits
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH -o extended_hits.out
#SBATCH -e extended_hits.err

n_resample=50

start=$(date +%s)

cd ~/.julia/dev/NeutrinoTelescopes/ && /home/hpc/capn/capn100h/.juliaup/bin/julia --project=. scripts/photon_tables/photon_tables_hits.jl --infile $WORK/photon_tables/extended/photon_table_extended_${SLURM_ARRAY_TASK_ID}.hd5 --outfile $WORK/photon_tables/extended/hits/photon_table_extended_${SLURM_ARRAY_TASK_ID}_hits.hd5 --resample ${n_resample}
end=$(date +%s)
echo "Elapsed Time: $(($end-$start)) seconds"