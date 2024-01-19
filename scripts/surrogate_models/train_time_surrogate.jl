using NeutrinoTelescopes
using HDF5
using DataFrames
using CairoMakie
using Base.Iterators
using CUDA
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
    "-s"
    help = "Timing uncert (sigma)"
    required = false
    default = 1.5
    arg_type = Float64
    "--hparam_config"
    help = "Hyper parameter config file"
    required = true
    "--model_name"
    help = "Model name"
    required = true
    "--perturb_medium"
    help = "Train a model with medium perturbation"
    action = :store_true
end
parsed_args = parse_args(ARGS, s; as_symbols=true)

fnames_casc = parsed_args[:i]
outpath = parsed_args[:o]
model_name = parsed_args[:model_name]


hit_buffer = Vector{Float64}(undef, Int64(1E8))
pmt_ixs_buffer = Vector{Int64}(undef, Int64(1E8))

feature_length = parsed_args[:perturb_medium] ? 24 + 2 : 24

features_buffer = Matrix{Float64}(undef, feature_length, Int64(1E8))

rng = MersenneTwister(31338)
nsel_frac = 0.9
hits, features, tf_vec = read_pmt_hits!(fnames_casc, hit_buffer, pmt_ixs_buffer, features_buffer, nsel_frac, rng)

if parsed_args[:s] > 0
    @inbounds for i in eachindex(hits)
        hits[i] = randn() * parsed_args[:s]
    end
end

data = (tres=hits, label=features)

model_type = parsed_args[:perturb_medium] ? AbsScaRQNormFlowHParams : RQNormFlowHParams

hparams = model_type(
    K=12,
    batch_size=5000,
    mlp_layers = 2,
    mlp_layer_size = 512,
    lr = 0.001,
    lr_min = 1E-7,
    epochs = 100,
    dropout = 0.1,
    non_linearity = "relu",
    seed = 31338,
    l2_norm_alpha = 0.0,
    adam_beta_1 = 0.9,
    adam_beta_2 = 0.999,
    resnet = false
)

@show model_type


kfold_train_model(data, outpath, model_name, tf_vec, 5, hparams)
