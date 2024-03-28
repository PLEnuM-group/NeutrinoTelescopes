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
using MLUtils


s = ArgParseSettings()

event_choices = ["extended", "lightsabre"]

@add_arg_table s begin
    "-i"
    help = "Input files"
    nargs = '+'
    action => :store_arg
    "-o"
    help = "Output path"
    required = true
    "--model_name"
    help = "Model name"
    required = true
    "--perturb_medium"
    help = "Train a model with medium perturbation"
    action = :store_true
    "--event_type"
    help = "Event type; must be one of " * join(event_choices, ", ", " or ")
    range_tester = (x -> x in event_choices)
    default = "extended"
end
parsed_args = parse_args(ARGS, s; as_symbols=true)

fnames_casc = parsed_args[:i]
outpath = parsed_args[:o]
model_name = parsed_args[:model_name]


rng = MersenneTwister(31338)
nsel_frac = 0.9

feature_length = parsed_args[:perturb_medium] ? 8 + 2 : 8


hit_buffer = Matrix{Float64}(undef, 16, Int64(1E8))
features_buffer = Matrix{Float64}(undef, feature_length, Int64(1E8))


hits, features = read_amplitudes_from_hdf5!(fnames_casc, hit_buffer, features_buffer, nsel_frac, nothing)

tf_vec = Vector{Normalizer}(undef, 0)

@views for row in eachrow(features[1:feature_length, :])
    _, tf = fit_normalizer!(row)
    push!(tf_vec, tf)
end

data = (nhits=hits, labels=features)

model_type = parsed_args[:perturb_medium] ? AbsScaPoissonExpModelParams : PoissonExpModelParams

if parsed_args[:event_type] == "lightsabre"
    hparams = model_type(
        batch_size=8192,
        mlp_layers = 3,
        mlp_layer_size = 768,
        lr = 0.0035,
        lr_min = 1E-8,
        epochs = 100,
        dropout = 0.0015,
        non_linearity = "relu",
        seed = 31338,
        l2_norm_alpha = 0.000658,
        adam_beta_1 = 0.9,
        adam_beta_2 = 0.999,
        resnet = false
    )
elseif parsed_args[:event_type] == "extended"
    hparams = model_type(
        batch_size=8192,
        mlp_layers = 3,
        mlp_layer_size = 916,
        lr = 0.001,
        lr_min = 1E-8,
        epochs = 150,
        dropout = 0.05,
        non_linearity = "relu",
        seed = 31338,
        l2_norm_alpha = 0,
        adam_beta_1 = 0.9,
        adam_beta_2 = 0.999,
        resnet = false
    )
else
    error("Unknown event type")
end


ptm_flag = parsed_args[:perturb_medium] ? "perturb" : "const_medium"

logdir = joinpath(ENV["ECAPSTOR"], "tensorboard/kfold_$(model_name)_$(ptm_flag)")

flds = kfolds(data; k=3)

kfold_train_model(data, outpath, model_name, tf_vec, 3, hparams, logdir)
