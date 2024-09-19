using PhotonSurrogateModel
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


event_choices = ["extended", "lightsabre", "hadronic"]

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
    "--event_type"
    help = "Event type; must be one of " * join(event_choices, ", ", " or ")
    range_tester = (x -> x in event_choices)
    default = "extended"
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

tf_vec = Vector{UnitRangeScaler}(undef, 0)
@views for row in eachrow(features[1:feature_length, :])
    _, tf = fit_transformation!(UnitRangeScaler, row)
    push!(tf_vec, tf)
end


if parsed_args[:s] > 0
    @inbounds for i in eachindex(hits)
        hits[i] += randn() * parsed_args[:s]
    end
end

rand_mat = randn(rng, (hparams.fourier_mapping_size, feature_length))
data = (nhits=hits, labels=fourier_input_mapping(features, rand_mat*hparams.fourier_gaussian_scale))


model_type = parsed_args[:perturb_medium] ? AbsScaRQNormFlowFourierHParams : RQNormFlowFourierHParams
if parsed_args[:event_type] == "extended"
    hparams = model_type(
        K=10,
        batch_size=65536,
        mlp_layers = 2,
        mlp_layer_size = 768,
        lr = 0.0024,
        lr_min = 1E-8,
        epochs = 80,
        dropout = 0.05,
        non_linearity = "gelu",
        seed = 31338,
        l2_norm_alpha = 0.0045,
        adam_beta_1 = 0.9,
        adam_beta_2 = 0.999,
        resnet = false,
        fourier_gaussian_scale=fourier_feature_scale,
        fourier_mapping_size=fourier_mapping_size,
    )
elseif parsed_args[:event_type] == "hadronic"
    hparams = model_type(
        K=10,
        batch_size=65536,
        mlp_layers = 3,
        mlp_layer_size = 1024,
        lr = 0.0015,
        lr_min = 1E-8,
        epochs = 100,
        dropout = 0.006,
        non_linearity = "gelu",
        seed = 31338,
        l2_norm_alpha = 0.000004,
        adam_beta_1 = 0.9,
        adam_beta_2 = 0.999,
        resnet = false,
        fourier_gaussian_scale=fourier_feature_scale,
        fourier_mapping_size=fourier_mapping_size,
    )
elseif parsed_args[:event_type] == "lightsabre"
    hparams = model_type(
        K=10,
        batch_size=65536,
        mlp_layers = 2,
        mlp_layer_size = 768,
        lr = 0.002,
        lr_min = 1E-8,
        epochs = 80,
        dropout = 0.14,
        non_linearity = "gelu",
        seed = 31338,
        l2_norm_alpha = 0.00005,
        adam_beta_1 = 0.9,
        adam_beta_2 = 0.999,
        resnet = false,
        fourier_gaussian_scale=fourier_feature_scale,
        fourier_mapping_size=fourier_mapping_size,
    )
else
    error("Unknown type")
end

ptm_flag = parsed_args[:perturb_medium] ? "perturb" : "const_medium"

logdir = joinpath(ENV["ECAPSTOR"], "tensorboard/time_model_kfold_$(model_name)_$(ptm_flag)")

n_folds = 3

for (model_num, (train_data, val_data)) in enumerate(kfolds(shuffleobs(data); k=n_folds))

    model, loss_f = setup_model(hparams, tf_vec, rand_mat)
    model = gpu(model)

    opt_state, train_loader, test_loader, lg, schedule = setup_training(model, 
        train_data, val_data, hparams, logdir
    )

    device = gpu
    model, final_test_loss, best_test_loss, best_test_epoch, time_elapsed = train_model!(
        optimizer=opt_state,
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        loss_function=loss_f,
        hparams=hparams,
        logger=lg,
        device=device,
        use_early_stopping=false,
        checkpoint_path=nothing,
        schedule=schedule)

    model_path = joinpath(outpath, "$(model_name)_$(model_num)_FNL.bson")
    model = cpu(model)
    @save model_path model hparams
end
