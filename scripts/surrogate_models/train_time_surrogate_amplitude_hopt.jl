using NeutrinoTelescopes
using PhotonPropagation
using PhysicsTools
using Glob
import Base.GC: gc
using StructTypes
using ProgressLogging
using JLD2
using CUDA
using Distributed
using Base.Iterators


addprocs(2*length(devices()))

@everywhere using CUDA

# assign devices
asyncmap((zip(workers(), cycle(devices())))) do (p, d)
    remotecall_wait(p) do
        @info "Worker $p uses $d"
        device!(d)
    end
end


@everywhere using HyperTuning, MLUtils, Flux, Random, ParameterHandling, TensorBoardLogger, Logging
@everywhere using ParameterSchedulers, JLD2, PhotonSurrogateModel, Zygote
@everywhere using ParameterSchedulers: next!
@everywhere using SharedArrays

function load_data()

    fname = "/home/wecapstor3/capn/capn100h/snakemake/training_inputs/amplitude_input__perturb_hadronic.jld2"
    fid = jldopen(fname) 
    hits = fid["hits"][:, :]
    features = fid["features"][:, :]
    close(fid)

    feature_length = 10 

    tf_vec = Vector{UnitRangeScaler}(undef, 0)
    @views for row in eachrow(features[1:feature_length, :])
        _, tf = fit_transformation!(UnitRangeScaler, row)
        push!(tf_vec, tf)
    end

    return hits, features, tf_vec
end

hits, features, tf_vec = load_data()

hits_s = SharedArray{Float32}(hits)
features_s = SharedArray{Float32}(features)

@everywhere hits_s = $hits_s
@everywhere features_s = $features_s
@everywhere tf_vec = $tf_vec

rng = MersenneTwister(31338)
@everywhere fourier_mapping_size = 64
@everywhere rand_mat = randn($rng, (fourier_mapping_size, 10))

@everywhere function objective(trial)

    logdir = joinpath(ENV["ECAPSTOR"], "tensorboard/hopt_amplitude_hadronic/time_surrogate_amplitude_hadronic")

    logger = TBLogger(logdir)
    # fix seed for the RNG
    seed = get_seed(trial)
    Random.seed!(seed)
    
    batch_size = 16384

    # get suggested hyperparameters
    @suggest mlp_layer_size in trial
    #@suggest mlp_layers in trial

    mlp_layers = 3
    #mlp_layer_size = 1024

    #@suggests epochs in trial

    non_lin = "gelu"

    # hyperparameters for the optimizer
    @suggest lr in trial
    @suggest l2_norm_alpha in trial
    #l2_norm_alpha = 0.03
    @suggest dropout in trial
    #dropout = 0.05

    #@suggest epochs in trial
    epochs = 150
    @suggest fourier_feature_scale in trial

    
    data = (nhits=hits_s, labels=fourier_input_mapping(features_s, rand_mat*fourier_feature_scale))
    train_loader, test_loader = PhotonSurrogateModel.setup_dataloaders(data, 31338, batch_size)

    hparams = AbsScaPoissonExpFourierModelParams(
        batch_size=batch_size,
        mlp_layers = mlp_layers,
        mlp_layer_size = mlp_layer_size,
        lr = lr,
        lr_min = 1E-8,
        epochs = epochs,
        dropout = dropout,
        non_linearity = non_lin,
        seed = 31338,
        l2_norm_alpha = l2_norm_alpha,
        adam_beta_1 = 0.9,
        adam_beta_2 = 0.999,
        resnet = false,
        fourier_gaussian_scale=fourier_feature_scale,
        fourier_mapping_size=fourier_mapping_size,
    )

    model, loss_f = setup_model(hparams, tf_vec)
    model = gpu(model)

    opt = PhotonSurrogateModel.setup_optimizer(hparams)

    schedule = CosAnneal(λ0=hparams.lr_min, λ1=hparams.lr, period=hparams.epochs)

    #schedule = ParameterSchedulers.Constant(hparams.lr)
    opt_state = Flux.setup(opt, model)
    write_hparams!(logger, Dict(string(key)=>getfield(hparams, key) for key ∈ fieldnames(typeof(hparams))), ["loss/train", "loss/test"])

    test_loss = Inf64

    for (eta, epoch) in zip(schedule, 1:hparams.epochs)
        train_loss = 0.
        Flux.trainmode!(model)
        #optim.eta = next!(schedule)
        for d in train_loader
            d = d |> gpu
            loss, grads = Flux.withgradient(model) do m
                # Evaluate model and loss inside gradient context:
                loss_f(d, m)
            end
            Flux.update!(opt_state, model, grads[1])
            
            train_loss += cpu(loss) / length(train_loader)
        end
        #push!(train_losses, train_loss)

        Flux.adjust!(opt_state, eta)

        Flux.testmode!(model)
        test_loss = 0.
        for d in test_loader
            d = d |> gpu
            loss = loss_f(d, model)
            test_loss += cpu(loss)
        end
        test_loss /= length(test_loader)

        if !isnothing(logger)
            with_logger(logger) do
                @info "loss" train = train_loss test = test_loss log_step_increment = length(train_loader)
            end

        end
        println("Epoch: $epoch, Train: $train_loss Test: $test_loss, Eta: $eta")
    end

    return test_loss
end



scenario = Scenario(### hyperparameters
                    lr = BoxConstrainedSpace(0.0005, 0.003),
                    l2_norm_alpha = BoxConstrainedSpace(0.009, 1),
                    fourier_feature_scale = BoxConstrainedSpace(0.001, 0.5),
                    dropout = BoxConstrainedSpace(0, 0.1),
                    #epochs = [80, 100, 150, 200, 250],
                    #mlp_layers = [2, 3],
                    mlp_layer_size = [768, 1024],
                    pruner= NeverPrune(),
                    verbose = true, # show the log
                    max_trials = 100, # maximum number of hyperparameters computed
                   )

display(scenario)

# minimize accuracy error
scen_opt = HyperTuning.optimize(objective, scenario)

display(scenario)
display(top_parameters(scenario))

@info "History"
# display all evaluated trials
display(history(scenario))

 