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


s = ArgParseSettings()
@add_arg_table s begin
    "-i"
    help = "Input files"
    nargs = '+'
    action => :store_arg
    "-o"
    help = "Output path"
    required = true
end
parsed_args = parse_args(ARGS, s; as_symbols=true)

fnames_casc = parsed_args[:i]
outpath = parsed_args[:o]

rng = MersenneTwister(31338)
nsel_frac = 0.9
tres, nhits, cond_labels, tf_dict = read_pmt_hits(fnames_casc, nsel_frac, rng)
data = (tres=tres, label=cond_labels, nhits=nhits)

hyperparams_default = Dict(
    :K => 12,
    :epochs => 100,
    :lr => 0.007,
    :mlp_layer_size => 768,
    :mlp_layers => 2,
    :dropout => 0.1,
    :non_linearity => :relu,
    :batch_size => 30000,
    :seed => 1,
    :l2_norm_alpha => 0,
    :adam_beta_1 => 0.9,
    :adam_beta_2 => 0.999
)

kfold_train_model(data, outpath, "full_kfold", tf_dict; hyperparams_default...)
