using NeutrinoTelescopes
using PhotonPropagation
using PhysicsTools
using Flux
using Random
using ParameterHandling
using MLUtils
using Glob
using HyperTuning
import Base.GC: gc
using TensorBoardLogger
using Logging
using ParameterSchedulers
using ParameterSchedulers: next!
using Zygote
using StructTypes
using ProgressLogging

function make_dataloaders(data, batch_size=8192)
    train_data, test_data = splitobs(data, at=0.8, shuffle=true)
    train_loader = Flux.DataLoader(train_data, batchsize=batch_size, shuffle=true)
    test_loader = Flux.DataLoader(test_data, batchsize=size(test_data[1], 2), shuffle=false)
    return train_loader, test_loader
end


function make_objective(fnames, perturb_medium=true)

    rng = MersenneTwister(31338)
    nsel_frac = 0.9


    feature_length = perturb_medium ? 8 + 2 : 8
    hit_buffer = Matrix{Float64}(undef, 16, Int64(1E8))
    features_buffer = Matrix{Float64}(undef, feature_length, Int64(1E8))
    hits, features = read_amplitudes_from_hdf5!(fnames, hit_buffer, features_buffer, nsel_frac, rng)
    tf_vec = Vector{Normalizer}(undef, 0)

    @show size(hits), size(features)

    @views for row in eachrow(features[1:feature_length, :])
        _, tf = fit_normalizer!(row)
        push!(tf_vec, tf)
    end

    data = (nhits=hits, labels=features)

    train_loader, test_loader = make_dataloaders(data)

    model_type = perturb_medium ? AbsScaPoissonExpModelParams : PoissonExpModelParams
   
    logdir = joinpath(ENV["ECAPSTOR"], "tensorboard/time_surrogate_amplitude")
    

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

        hparams = model_type(
            batch_size=8192,
            mlp_layers = mlp_layers,
            mlp_layer_size = mlp_layer_size,
            lr = lr,
            lr_min = 1E-8,
            epochs = 80,
            dropout = dropout,
            non_linearity = "relu",
            seed = 31338,
            l2_norm_alpha = l2_norm_alpha,
            adam_beta_1 = 0.9,
            adam_beta_2 = 0.999,
            resnet = false
        )
    
        model, loss_f = setup_model(hparams, tf_vec)
        model = gpu(model)
    
        opt = PhotonSurrogates.setup_optimizer(hparams)

        schedule = CosAnneal(λ0=hparams.lr_min, λ1=hparams.lr, period=hparams.epochs)

        opt_state = Flux.setup(opt, model)
        write_hparams!(logger, Dict(string(key)=>getfield(hparams, key) for key ∈ fieldnames(typeof(hparams))), ["loss/train", "loss/test"])

        test_loss = Inf64

        @progress for (eta, epoch) in zip(schedule, 1:hparams.epochs)
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


fnames = [
    "/home/wecapstor3/capn/capn100h/snakemake/photon_tables_perturb/extended/hits/photon_table_hits_extended_dmin_1_dmax_200_emin_100_emax_100000.0_0.hd5",
    "/home/wecapstor3/capn/capn100h/snakemake/photon_tables_perturb/extended/hits/photon_table_hits_extended_dmin_1_dmax_200_emin_100_emax_100000.0_1.hd5",
    "/home/wecapstor3/capn/capn100h/snakemake/photon_tables_perturb/extended/hits/photon_table_hits_extended_dmin_1_dmax_200_emin_100_emax_100000.0_3.hd5",
    "/home/wecapstor3/capn/capn100h/snakemake/photon_tables_perturb/extended/hits/photon_table_hits_extended_dmin_1_dmax_200_emin_100000.0_emax_5000000.0_0.hd5",
    "/home/wecapstor3/capn/capn100h/snakemake/photon_tables_perturb/extended/hits/photon_table_hits_extended_dmin_1_dmax_200_emin_100000.0_emax_5000000.0_1.hd5",
   ]

obj = make_objective(fnames, true)

scenario = Scenario(### hyperparameters
                    # learning rates
                    lr = BoxConstrainedSpace(0.001, 0.01),
                    l2_norm_alpha = BoxConstrainedSpace(0., 0.0001),
                    # number of dense layers
                    mlp_layers = 2:3,
                    # number of neurons for each dense layer
                    mlp_layer_size = Bounds(400, 1024),
                    ### Common settings
                    pruner= MedianPruner(start_after = 5#=trials=#, prune_after = 5#=epochs=#),
                    verbose = true, # show the log
                    max_trials = 70, # maximum number of hyperparameters computed
                    dropout = BoxConstrainedSpace(0., 0.01)
                   )

display(scenario)

# minimize accuracy error
scen_opt = HyperTuning.optimize(obj, scenario)

display(scenario)