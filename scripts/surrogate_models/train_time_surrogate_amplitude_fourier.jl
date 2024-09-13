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
using JSON3
using MLUtils
using Sobol
using ParameterSchedulers
using SearchSpaces

using Surrogates
using AbstractGPs
using SurrogatesAbstractGPs

fnames = [
   "/home/wecapstor3/capn/capn100h/snakemake/photon_tables_perturb/extended/hits/photon_table_hits_extended_dmin_1_dmax_200_emin_100_emax_100000.0_0.hd5",
   "/home/wecapstor3/capn/capn100h/snakemake/photon_tables_perturb/extended/hits/photon_table_hits_extended_dmin_1_dmax_200_emin_100_emax_100000.0_1.hd5",
   "/home/wecapstor3/capn/capn100h/snakemake/photon_tables_perturb/extended/hits/photon_table_hits_extended_dmin_1_dmax_200_emin_100_emax_100000.0_2.hd5",
   "/home/wecapstor3/capn/capn100h/snakemake/photon_tables_perturb/extended/hits/photon_table_hits_extended_dmin_1_dmax_200_emin_100_emax_100000.0_3.hd5"
    ]



nsel_frac = 0.9
feature_length = 10

hit_buffer = Matrix{Float32}(undef, 16, Int64(1E8))
features_buffer = Matrix{Float32}(undef, feature_length, Int64(1E8))
hits, features = read_amplitudes_from_hdf5!(fnames, hit_buffer, features_buffer, nsel_frac, nothing)
tf_vec = Vector{UnitRangeScaler}(undef, 0)

@views for row in eachrow(features[1:feature_length, :])
    _, tf = fit_transformation!(UnitRangeScaler, row)
    push!(tf_vec, tf)
end

fourier_mapping_size = 64
rng = MersenneTwister(31338)
rand_mat = randn(rng, Float32, (fourier_mapping_size, 10)) 

function fit_model(params)

    log10_gaussian_scale = params

    gaussian_scale = 10^log10_gaussian_scale

    epochs = 80

    hparams = AbsScaPoissonExpFourierModelParams(
        batch_size=8192,
        mlp_layers = 3,
        mlp_layer_size = 1024,
        lr = 0.0025,
        lr_min = 1E-6,
        epochs = epochs,
        dropout = 0.05,
        non_linearity = "gelu",
        seed = 31338,
        l2_norm_alpha = 0.03,
        adam_beta_1 = 0.9,
        adam_beta_2 = 0.999,
        resnet = false,
        fourier_gaussian_scale=gaussian_scale,
        fourier_mapping_size=fourier_mapping_size,
        )

        data = (nhits=hits, labels=fourier_input_mapping(features, rand_mat* gaussian_scale))


    train_data, test_data = splitobs(data, at=0.8, shuffle=true)

    logdir = joinpath(ENV["ECAPSTOR"], "tensorboard/lr_test")
    
    opt_state, train_loader, test_loader, model, loss_f, lg, schedule = setup_training(
        train_data, test_data, tf_vec, hparams, logdir
    )


    model, final_test_loss, best_test_loss, best_test_epoch, time_elapsed = PhotonSurrogateModel.train_model!(
        optimizer=opt_state,
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        loss_function=loss_f,
        hparams=hparams,
        logger=lg,
        device=gpu,
        use_early_stopping=false,
        checkpoint_path=nothing,
        schedule=schedule)

    write_hparams!(lg, Dict(string(key)=>getfield(hparams, key) for key ∈ fieldnames(typeof(hparams))), ["loss/train", "loss/test"])
    
    return log10(final_test_loss)
end


fit_model(log10(0.12))

sched = Sequence(
        [OneCycle(15, 0.003, percent_start = 0.4),
         Sin(0.001, 0.001, 20),
         Sin(0.0001, 0.0003, 10)],
         [15, 40, 30])

plot([_ for _ in zip(1:150, sched, )])


n_samples = 8
lower_bound = -2.5
upper_bound = 1.

params = sample(n_samples, lower_bound, upper_bound, SobolSample())

test_losses = fit_model.(params)

mask = isfinite.(test_losses)
gp_surrogate = AbstractGPSurrogate(params[mask], (test_losses[mask]))

surrogate_optimize(fit_model, SRBF(), lower_bound, upper_bound, gp_surrogate, SobolSample(), maxiters=15)

10^-0.927

fig = Figure()
ax = Axis(fig[1, 1])

#surrogate_optimize(f, SRBF(), lower_bound, upper_bound, gp_surrogate, SobolSample())

scatter!(ax, gp_surrogate.x, gp_surrogate.y)
x = -2:0.01:1
lines!(ax, x, gp_surrogate.(x))
fig



#p2 = contour(x, y, (x, y) -> gp_surrogate([x y]))
#scatter!(xs, ys, marker_z=zs)
#plot(p1, p2, title="Surrogate")