using NeutrinoTelescopes
using HDF5
using DataFrames
using CairoMakie
using Base.Iterators
using CUDA
using BenchmarkTools
using Random
using TensorBoardLogger
using Glob

using Flux
using BSON: @save, @load
using ArgParse
using JSON


s = ArgParseSettings()
@add_arg_table s begin
    "-i"
    help = "Input files"
    nargs = '+'
    action => :store_arg
    "-o"
    help = "Output path"
    required = true
    "--hparam_config"
    help = "Output path"
    required = true
    "--model_name"
    help = "Model name"
    required = true
end
parsed_args = parse_args(ARGS, s; as_symbols=true)

fnames_casc = parsed_args[:i]
outpath = parsed_args[:o]
model_name = parsed_args[:model_name]


rng = MersenneTwister(31338)
nsel_frac = 0.9
tres, nhits, cond_labels, tf_dict = read_pmt_hits(fnames_casc, nsel_frac, rng)
data = (tres=tres, label=cond_labels, nhits=nhits)

hyperparams = JSON.parsefile(parsed_args[:hparam_config], dicttype=Dict{Symbol,Any})

@show hyperparams

kfold_train_model(data, outpath, model_name, tf_dict; hyperparams...)
