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
using MLUtils


s = ArgParseSettings()

event_choices = ["extended", "lightsabre", "hadronic"]

@add_arg_table s begin
    "-i"
    help = "Input file"
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

fnames = parsed_args[:i]
outpath = parsed_args[:o]
model_name = parsed_args[:model_name]


rng = MersenneTwister(31338)

feature_length = parsed_args[:perturb_medium] ? 8 + 2 : 8

fid = jldopen(fname) 
hits = fid["hits"][:]
features = fid["features"][:, :]
close(fid)


tf_vec = Vector{UnitRangeScaler}(undef, 0)
@views for row in eachrow(features[1:feature_length, :])
    _, tf = fit_transformation!(UnitRangeScaler, row)
    push!(tf_vec, tf)
end



model_type = parsed_args[:perturb_medium] ? AbsScaPoissonExpFourierModelParams : PoissonExpFourierModelParams

if parsed_args[:event_type] == "lightsabre"
    hparams = model_type(
        batch_size=16384,
        mlp_layers = 3,
        mlp_layer_size = 768,
        lr = 0.002,
        lr_min = 1E-8,
        epochs = 100,
        dropout = 0.0046,
        non_linearity = "relu",
        seed = 31338,
        l2_norm_alpha = 0.039,
        adam_beta_1 = 0.9,
        adam_beta_2 = 0.999,
        resnet = false,
        fourier_gaussian_scale=0.0977,
        fourier_mapping_size=64,
    )
elseif parsed_args[:event_type] == "extended"
    hparams = model_type(
        batch_size = 16384,
        mlp_layers = 3,
        mlp_layer_size = 768,
        lr = 0.0023,
        lr_min = 1E-8,
        epochs = 150,
        dropout = 0.0044,
        non_linearity = "gelu",
        seed = 31338,
        l2_norm_alpha = 0.1,
        adam_beta_1 = 0.9,
        adam_beta_2 = 0.999,
        resnet = false,
        fourier_gaussian_scale = 0.1,
        fourier_mapping_size = 64
    )
elseif parsed_args[:event_type] == "hadronic"
    hparams = model_type(
        batch_size = 16384,
        mlp_layers = 3,
        mlp_layer_size = 768,
        lr = 0.00058,
        lr_min = 1E-8,
        epochs = 150,
        dropout = 0.0044,
        non_linearity = "gelu",
        seed = 31338,
        l2_norm_alpha = 0.29,
        adam_beta_1 = 0.9,
        adam_beta_2 = 0.999,
        resnet = false,
        fourier_gaussian_scale = 0.116,
        fourier_mapping_size = 64
    )
else
else
    error("Unknown event type")
end

rand_mat = randn(rng, (hparams.fourier_mapping_size, feature_length))

data = (nhits=hits, labels=fourier_input_mapping(features, rand_mat*fourier_feature_scale))


ptm_flag = parsed_args[:perturb_medium] ? "perturb" : "const_medium"

logdir = joinpath(ENV["ECAPSTOR"], "tensorboard/kfold_$(model_name)_$(ptm_flag)")

flds = kfolds(data; k=3)

for (model_num, (train_data, val_data)) in enumerate(kfolds(shuffleobs(data); k=n_folds))

    model, loss_f = setup_model(hparams, tf_vec, rand_mat)
    model = gpu(model)

    opt_state, train_loader, test_loader, lg, schedule = setup_training(
        train_data, val_data, tf_vec, hparams, logdir
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




