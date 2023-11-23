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
import Base.GC: gc
using TensorBoardLogger
using Logging

include("train_cov_utils.jl")


function make_objective()
    data = load_data_from_dir(joinpath(ENV["ECAPSTOR"], "snakemake/training_data_cov"), "extended")
    train_loader, test_loader = FisherSurrogate.make_dataloaders(data)
    logdir = joinpath(ENV["ECAPSTOR"], "tensorboard/cov_model")
    

    function objective(trial)
        logger = TBLogger(logdir)
        # fix seed for the RNG
        seed = get_seed(trial)
        Random.seed!(seed)
       
        # get suggested hyperparameters
        @suggest mlp_layer_size in trial
        @suggest mlp_layers in trial

        # hyperparameters for the optimizer
        @suggest lr in trial
        @suggest l2_norm_alpha in trial
        @suggest dropout in trial

        hparams = FisherSurrogateModelParams(
            mlp_layers = mlp_layers,
            mlp_layer_size = mlp_layer_size,
            lr = lr,
            l2_norm_alpha = l2_norm_alpha,
            dropout = dropout,
            non_linearity=relu
        )

        write_hparams!(logger, Dict(hparams), ["loss/train", "loss/test"])

        cov_model, optim = FisherSurrogate.setup_training(hparams)
        test_loss = Inf64

        for epoch in ProgressBar(1:80)
            train_loss = 0.
            Flux.trainmode!(cov_model)
            for (x, y) in train_loader
                loss, grads = Flux.withgradient(cov_model) do m
                    # Evaluate model and loss inside gradient context:
                    y_hat = m(x |> gpu)
                    Flux.mse(y_hat, y |> gpu)
                end
                Flux.update!(optim, cov_model, grads[1])
                
                train_loss += loss / length(train_loader)
            end
            #push!(train_losses, train_loss)

            Flux.testmode!(cov_model)
            test_loss = 0.
            for (x, y) in test_loader
                loss = Flux.mse(cov_model(x |> gpu), y |> gpu)
                test_loss += loss / length(test_loader)
            end
            #push!(test_losses, test_loss)

            if !isnothing(logger)
                with_logger(logger) do
                    @info "loss" train = train_loss test = test_loss
                    #@info "model" params = param_dict log_step_increment = 0
                end
    
            end
            # println("Epoch: $epoch, Train: $train_loss Test: $test_loss")

            report_value!(trial, test_loss)
            # check if pruning is necessary
            should_prune(trial) && (return)
        end

        report_success!(trial)
        gc()
        
        return test_loss
    end

    return objective

end

obj = make_objective()

scenario = Scenario(### hyperparameters
                    # learning rates
                    lr = BoxConstrainedSpace(0., 0.01),
                    l2_norm_alpha = BoxConstrainedSpace(0., 0.001),
                    # number of dense layers
                    mlp_layers = 3:4,
                    # number of neurons for each dense layer
                    mlp_layer_size = Bounds(650, 1024),
                    ### Common settings
                    pruner= MedianPruner(start_after = 5#=trials=#, prune_after = 5#=epochs=#),
                    verbose = true, # show the log
                    max_trials = 40, # maximum number of hyperparameters computed
                    dropout = BoxConstrainedSpace(0., 0.5)
                   )

display(scenario)

# minimize accuracy error
scen_opt = HyperTuning.optimize(obj, scenario)

display(scenario)