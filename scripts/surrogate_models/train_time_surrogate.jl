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
using JLD2


s = ArgParseSettings()
@add_arg_table s begin
    "-i"
    help = "Input file"
    "-o"
    help = "Output path"
    required = true
    "-s"
    help = "Timing uncert (sigma)"
    required = false
    default = 1.5
    arg_type = Float64
    "--model_name"
    help = "Model name"
    required = true
    "--perturb_medium"
    help = "Train a model with medium perturbation"
    action = :store_true
end
parsed_args = parse_args(ARGS, s; as_symbols=true)

#=
parsed_args = Dict(
    :i => "/home/wecapstor3/capn/capn100h/snakemake/training_inputs/time_input__perturb_extended.jld2",
    :o => "/tmp/",
    :s => 1.5,
    :model_name => "test",
    :perturb_medium => true)
=#
fname = parsed_args[:i]
outpath = parsed_args[:o]
model_name = parsed_args[:model_name]

rng = MersenneTwister(31338)
fid = jldopen(fname) 
hits = fid["hits"][:]
features = fid["features"][:, :]
close(fid)

feature_length = parsed_args[:perturb_medium] ? 24 + 2 : 24

tf_vec = Vector{Normalizer}(undef, 0)

@views for row in eachrow(features[1:feature_length-16, :])
    _, tf = fit_normalizer!(row)
    push!(tf_vec, tf)
end


if parsed_args[:s] > 0
    @inbounds for i in eachindex(hits)
        hits[i] += randn() * parsed_args[:s]
    end
end

data = (tres=hits, label=features)

model_type = parsed_args[:perturb_medium] ? AbsScaRQNormFlowHParams : RQNormFlowHParams

hparams = model_type(
    K=10,
    batch_size=32768,
    mlp_layers = 3,
    mlp_layer_size = 862,
    lr = 0.0005,
    lr_min = 1E-7,
    epochs = 100,
    dropout = 0.2,
    non_linearity = "relu",
    seed = 31338,
    l2_norm_alpha = 2E-5,
    adam_beta_1 = 0.9,
    adam_beta_2 = 0.999,
    resnet = false
)

ptm_flag = parsed_args[:perturb_medium] ? "perturb" : "const_medium"

logdir = joinpath(ENV["ECAPSTOR"], "tensorboard/time_model_kfold_$(model_name)_$(ptm_flag)")

kfold_train_model(data, outpath, model_name, tf_vec, 3, hparams, logdir)
