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
using MultivariateStats
using NeutrinoSurrogateModelData


include("train_cov_utils.jl")



function training_loop!(model, optim, train_loader, test_loader, logger, epochs=150)

    test_loss = Inf64
    for epoch in ProgressBar(1:epochs)
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
                @info "loss" train = train_loss test = test_loss log_step_increment = length(train_loader)
                #@info "model" params = param_dict log_step_increment = 0
            end
        end
    end
end


### Cascades
type = "per_string_extended"
(data, tf_in, tf_out) = load_data_from_dir(joinpath(ENV["ECAPSTOR"], "snakemake/training_data_cov"), type)

train_loader, test_loader = FisherSurrogate.make_dataloaders(data, 2^15)
logdir = joinpath(ENV["ECAPSTOR"], "tensorboard/cov_model_prod_per_string")

hparams = FisherSurrogateModelParams(
    mlp_layers = 3,
    mlp_layer_size = 1024,
    lr = 0.001,
    lr_min = 1E-6,
    l2_norm_alpha =4E-6,
    dropout = 0.1,
    non_linearity=relu,
    epochs = 100
)

cov_model, optim = FisherSurrogate.setup_training(hparams, length(train_loader))

logger = TBLogger(logdir)
write_hparams!(logger, Dict(hparams), ["loss/train", "loss/test"])
training_loop!(cov_model, optim, train_loader, test_loader, logger, hparams.epochs)
model_fname = joinpath(ENV["ECAPSTOR"], "snakemake/fisher_surrogates/fisher_surrogate_$type.bson")
save(model_fname,
    Dict(:model => cpu(cov_model),
        :tf_in => tf_in,
        :tf_out => tf_out,
        :range_cylinder => Cylinder(SA[0., 0., -475.], 1200., 150.))
)

### Tracks
type = "per_string_lightsabre"
(data, tf_in, tf_out) = load_data_from_dir(joinpath(ENV["ECAPSTOR"], "snakemake/training_data_cov"), type)

train_loader, test_loader = FisherSurrogate.make_dataloaders(data, 2^13)
logdir = joinpath(ENV["ECAPSTOR"], "tensorboard/cov_model_prod_per_string")

hparams = FisherSurrogateModelParams(
    mlp_layers = 3,
    mlp_layer_size = 968,
    lr = 0.0008,
    lr_min = 1E-6,
    l2_norm_alpha =4E-7,
    dropout = 0.15,
    non_linearity=relu,
    epochs = 150
)

cov_model, optim = FisherSurrogate.setup_training(hparams, length(train_loader))

logger = TBLogger(logdir)
write_hparams!(logger, Dict(hparams), ["loss/train", "loss/test"])
training_loop!(cov_model, optim, train_loader, test_loader, logger, hparams.epochs)
model_fname = joinpath(ENV["ECAPSTOR"], "snakemake/fisher_surrogates/fisher_surrogate_$type.bson")
save(model_fname,
    Dict(:model => cpu(cov_model),
        :tf_in => tf_in,
        :tf_out => tf_out,
        :range_cylinder => Cylinder(SA[0., 0., -475.], 1200., 150.))
)




model_fname = joinpath(ENV["ECAPSTOR"], "snakemake/fisher_surrogates/fisher_surrogate_per_string_lightsabre.bson") 

fisher_model = gpu(FisherSurrogateModelPerLine(model_fname))

targets_line = make_detector_line(@SVector[0., 0.0, 0.0], 20, 50, 1)
medium = make_cascadia_medium_properties(0.95f0)
detector_line = LineDetector([targets_line], medium)
#cylinder = get_bounding_cylinder(detector_line)

cylinder = Cylinder(SA[0., 0., -475.], 1100., 100.)

pdist = CategoricalSetDistribution(OrderedSet([PMuPlus, PMuMinus]), [0.5, 0.5])
ang_dist = LowerHalfSphere()
length_dist = Dirac(1E4)
edist = Pareto(1, 1E4)
time_dist = Dirac(0.)
inj = SurfaceInjector(CylinderSurface(cylinder), edist, pdist, ang_dist, length_dist, time_dist)

model = PhotonSurrogate(lightsabre_time_model(2)...)


input_buffer = create_input_buffer(model, 16*20, 1)
output_buffer = create_output_buffer(16*20, 100)
diff_cache = DiffCache(input_buffer, 13)

hit_generator = SurrogateModelHitGenerator(gpu(model), 200.0, input_buffer, output_buffer)

rng = Random.default_rng()

event = rand(inj)

p = first(event[:particles])
p_shift = shift_to_closest_approach(p, [0, 0, -475])


m, = calc_fisher_matrix(p_shift, detector_line, hit_generator, use_grad=true, rng=rng, cache=diff_cache, n_samples=200)
covm = inv(m)
cov_pred_sum,_ = predict_cov([event], detector_line, fisher_model, abs_scale=1, sca_scale=1)
diag(covm)
diag(cov_pred_sum[1])

begin

fig = Figure()
ax = Axis(fig[1, 1], title="Fisher")
ax2 = Axis(fig[1, 2], title="Surrogate")

hm = heatmap!(ax, covm, colorrange=(minimum(covm), maximum(covm)))
heatmap!(ax2, cov_pred_sum[1], colorrange=(minimum(covm), maximum(covm)))
Colorbar(fig[1,3 ], hm)
fig
end



lines = get_detector_lines(detector_line)
all_particles::Vector{Particle{Float32}} = [(first(event[:particles])) for event in [event]]
eval_targets = [DummyTarget(SA_F32[first(l).shape.position[1], first(l).shape.position[2], -475f0], 1) for l in lines]
n_events = 1
normalizers::Vector{Normalizer{Float32}} = fisher_model.tf_out
inv_y_tf = inv.(normalizers)
inp = @view NeuralFlowSurrogate._calc_flow_input!(all_particles, eval_targets, fisher_model.tf_in, fisher_model.input_buffer)[:, 1:(length(all_particles) * length(eval_targets))]

triu_pred = cpu(NeuralFlowSurrogate.apply_feature_transform(fisher_model.model(gpu(inp)), inv_y_tf).^3)

cholesky(m).U


all_fishers = FisherSurrogate.predict_fisher([event], lines, fisher_model)





sqrt.(diag(cov_pred_sum))
sqrt.(diag(cov))

#cov = 0.5* (cov + cov')

=#

