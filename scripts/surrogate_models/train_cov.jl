using NeutrinoTelescopes
using PhotonPropagation
using PhysicsTools
using PreallocationTools
using Flux
using Random
using StaticArrays
using Distributions
using LinearAlgebra
using CairoMakie
using JLD2
using ProgressBars
using ParameterHandling
using Zygote
using StatsBase
using StructArrays
using MLUtils
using Glob
using DataStructures
using PairPlots
using BSON
using HyperTuning
using TensorBoardLogger
using Logging
using ParameterSchedulers

include("train_cov_utils.jl")

(data, tf_in, tf_out) = load_data_from_dir(joinpath(ENV["ECAPSTOR"], "snakemake/training_data_cov"), "extended")
train_loader, test_loader = FisherSurrogate.make_dataloaders(data)
logdir = joinpath(ENV["ECAPSTOR"], "tensorboard/cov_model_prod")
 
hparams = FisherSurrogateModelParams(
    mlp_layers = 4,
    mlp_layer_size = 1024,
    lr = 0.0002,
    l2_norm_alpha = 0.0001,
    dropout = 0.1,
    non_linearity=relu
)

function training_loop!(model, optim, train_loader, test_loader, logger)

    test_loss = Inf64
    for epoch in ProgressBar(1:130)
        train_loss = 0.
        Flux.trainmode!(model)
        for (x, y) in train_loader
            loss, grads = Flux.withgradient(model) do m
                # Evaluate model and loss inside gradient context:
                y_hat = m(x |> gpu)
                Flux.mse(y_hat, y |> gpu)
            end
            Flux.update!(optim, model, grads[1])
            
            train_loss += loss / length(train_loader)
        end

        Flux.testmode!(model)
        test_loss = 0.
        for (x, y) in test_loader
            loss = Flux.mse(model(x |> gpu), y |> gpu)
            test_loss += loss / length(test_loader)
        end

        if !isnothing(logger)
            with_logger(logger) do
                @info "loss" train = train_loss test = test_loss
                #@info "model" params = param_dict log_step_increment = 0
            end
        end
    end
end


cov_model, optim = FisherSurrogate.setup_training(hparams)

logger = TBLogger(logdir)
training_loop!(cov_model, optim, train_loader, test_loader, logger)
model_fname = joinpath(ENV["ECAPSTOR"], "snakemake/fisher_surrogates/fisher_surrogate_extended.bson")
save(model_fname,
    Dict(:model => cpu(cov_model),
         :tf_in => tf_in,
         :tf_out => tf_out,)
)

#schedule = Interpolator(CosAnneal(λ0=1E-5, λ1=5E-3, period=10), length(train_loader))
#optim2 = Scheduler(schedule, optim)


  
fisher_model = gpu(FisherSurrogateModel(model_fname))

targets_line = make_detector_line(@SVector[0., 0.0, 0.0], 20, 50, 1)
medium = make_cascadia_medium_properties(0.95f0)
detector_line = Detector(targets_line, medium)
cylinder = get_bounding_cylinder(detector_line)
pdist = CategoricalSetDistribution(OrderedSet([PEMinus, PEPlus]), [0.5, 0.5])
ang_dist = UniformAngularDistribution()
length_dist = Dirac(0.)
edist = Dirac(5E4)
time_dist = Dirac(0.)
inj = VolumeInjector(cylinder, edist, pdist, ang_dist, length_dist, time_dist)

input_buffer = create_input_buffer(16*20, 1)
output_buffer = create_output_buffer(16*20, 100)
diff_cache = FixedSizeDiffCache(input_buffer, 6)

model = PhotonSurrogate(
    joinpath(ENV["ECAPSTOR"], "snakemake/time_surrogate/extended/amplitude_1_FNL.bson"),
    joinpath(ENV["ECAPSTOR"], "snakemake/time_surrogate/extended/time_uncert_2.5_1_FNL.bson"))
hit_generator = SurrogateModelHitGenerator(gpu(model), 200.0, input_buffer, output_buffer)

rng = Random.default_rng()

event = rand(inj)
m, = calc_fisher_matrix(event, detector_line, hit_generator, use_grad=true, rng=rng, cache=diff_cache)
cov = inv(m)

@time cov_pred_sum = predict_cov(event[:particles], targets_line, fisher_model)

fig = Figure()
ax = Axis(fig[1, 1], title="Fisher")
ax2 = Axis(fig[1, 2], title="Surrogate")

hm = heatmap!(ax, cov, colorrange=(minimum(cov), maximum(cov)))
heatmap!(ax2, cov_pred_sum, colorrange=(minimum(cov), maximum(cov)))
Colorbar(fig[1,3 ], hm)
fig


sqrt.(diag(cov_pred_sum))
sqrt.(diag(cov))

#cov = 0.5* (cov + cov')

=#

