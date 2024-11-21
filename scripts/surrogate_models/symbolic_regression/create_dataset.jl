using JLD2
include("utils.jl")

glob("*.hd5", "/home/wecapstor3/capn/capn100h/snakemake/photon_tables_perturb/extended/hits/")

data = load_data(glob("*.hd5", "/home/wecapstor3/capn/capn100h/snakemake/photon_tables_perturb/extended/hits/"))

jldsave("/home/wecapstor3/capn/capn100h/symbolic_regression/sr_dataset_full.jld2", data=data)