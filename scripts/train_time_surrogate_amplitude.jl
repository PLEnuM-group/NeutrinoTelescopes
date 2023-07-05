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
using JSON3


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
hits, features, tf_vec = read_pmt_number_of_hits(fnames_casc, nsel_frac, rng)

data = (nhits=hits, labels=features)


hparams = PoissonExpModel(
    batch_size=5000,
    mlp_layers = 2,
    mlp_layer_size = 512,
    lr = 0.001,
    lr_min = 1E-5,
    epochs = 100,
    dropout = 0.2,
    non_linearity = "relu",
    seed = 31338,
    l2_norm_alpha = 0.0,
    adam_beta_1 = 0.9,
    adam_beta_2 = 0.999,
    resnet = false
)



kfold_train_model(data, outpath, model_name, tf_vec, 5, hparams)
