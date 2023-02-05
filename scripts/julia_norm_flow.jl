using NeutrinoTelescopes
using HDF5
using DataFrames
using CairoMakie
using Base.Iterators
using CUDA
using BenchmarkTools
using Random
using StaticArrays
using CairoMakie
using OneHotArrays
using MLUtils
using Glob
using StatsBase
using Hyperopt
using Flux
using Plots
using BSON: @save, @load
using Parquet
using Combinatorics
using TensorBoardLogger

function train_models(files, model_name, batch_size)
    rng = MersenneTwister(31338)

    tres, nhits, cond_labels, tf_dict = read_pmt_hits(files, nsel_frac, rng)

    for model_num in 1:5
        nsel_frac = 0.5
        tres, nhits, cond_labels, tf_dict = read_pmt_hits(files, nsel_frac, rng)

        println("Dataset size: $(length(tres))")

        chk_path = joinpath(@__DIR__, "../data/$(model_name)_$model_num")

        model, model_loss, hparams, opt = train_time_expectation_model(
            (tres=gpu(tres), label=gpu(cond_labels), nhits=gpu(nhits)),
            true,
            true,
            chk_path,
            K=12,
            epochs=100,
            lr=0.005,
            mlp_layer_size=768,
            mlp_layers=2,
            dropout=0.1,
            non_linearity=:relu,
            batch_size=batch_size,
            seed=1,
            l2_norm_alpha=0)

        model_path = joinpath(@__DIR__, "../data/$(model_name)_$(model_num)_FNL.bson")
        model = cpu(model)

        @save model_path model hparams opt tf_dict
    end

    nsel_frac = 1
    tres, nhits, cond_labels, tf_dict = read_pmt_hits(files, nsel_frac, rng)

    chk_path = joinpath(@__DIR__, "../data/$(model_name)_FULL")

    model, model_loss, hparams, opt = train_time_expectation_model(
        (tres=tres, label=cond_labels, nhits=nhits),
        true,
        true,
        chk_path,
        K=12,
        epochs=100,
        lr=0.001,
        mlp_layer_size=768,
        mlp_layers=2,
        dropout=0.1,
        non_linearity=:relu,
        batch_size=batch_size,
        seed=1,
        l2_norm_alpha=0)

    model_path = joinpath(@__DIR__, "../data/$(model_name)_FULL_FNL.bson")
    model = cpu(model)

    @save model_path model hparams opt tf_dict
end


function train_one_model(data, model_name; hyperparams...)
    chk_path = joinpath(@__DIR__, "../data/$(model_name)_FULL")
    model, model_loss, best_test_loss, best_test_epoch, hparams, opt, time_elapsed = train_time_expectation_model(
        data,
        true,
        true,
        chk_path,
        model_name;
        hyperparams...)
    return model_loss, best_test_loss, best_test_epoch, hparams, time_elapsed
end

function kfold_model(data, model_name, tf_vec, k=5; hyperparams...)
    hparams = RQNormFlowHParams(; hyperparams...)
   
    logdir = joinpath(@__DIR__, "../../tensorboard_logs/$model_name")
   
    for (model_num, (train_data, val_data)) in enumerate(kfolds(data; k=k))
        lg = TBLogger(logdir)
        model = setup_time_expectation_model(hparams)
        chk_path = joinpath(@__DIR__, "../data/$(model_name)_$(model_num)")

        train_loader, test_loader = setup_dataloaders(train_data, val_data, hparams)
        opt = setup_optimizer(hparams, length(train_loader))   
        device = gpu
        model, final_test_loss, best_test_loss, best_test_epoch, time_elapsed = train_model!(
            optimizer=opt,
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            loss_function=log_likelihood_with_poisson,
            hparams=hparams,
            logger=lg,
            device=device,
            use_early_stopping=false,
            checkpoint_path=chk_path)    

        model_path = joinpath(@__DIR__, "../data/$(model_name)_$(model_num)_FNL.bson")
        model = cpu(model)
        @save model_path model hparams opt tf_vec
    end
end



fnames_casc = glob("photon_table_extended_*", joinpath(@__DIR__, "../data/"))

rng = MersenneTwister(31338)
nsel_frac = 0.9
tres, nhits, cond_labels, tf_dict = read_pmt_hits(fnames_casc, nsel_frac, rng)
length(tres)
data = (tres=tres, label=cond_labels, nhits=nhits)

hyperparams_default = Dict(
        :K => 12,
        :epochs => 100,
        :lr => 0.007,
        :mlp_layer_size => 768,
        :mlp_layers => 2,
        :dropout => 0.1,
        :non_linearity => :relu,
        :batch_size => 30000,
        :seed => 1,
        :l2_norm_alpha => 0,
        :adam_beta_1 => 0.9,
        :adam_beta_2 => 0.999
)

kfold_model(data, "full_kfold", tf_dict; hyperparams_default...)




train_one_model(data, "full", hyperparams_default...)





iterators = Dict(
    #:K => [5, 7, 9, 11, 13],
    :epochs => [50, 75, 100],
    :mlp_layer_size => [256, 512, 768],
    :lr => [0.01, 0.005, 0.001]
)

function run_experiment(iterators)
    exp_name = "exp_$(join(keys(iterators), "_"))"

    all_results = []
    for (i, hvals) in enumerate(product(values(iterators)...))
        hp = Dict(hyperparams_default)

        par_dict = Dict(zip(keys(iterators), hvals))
        hp = merge(hp, par_dict)
        model_loss, best_test_loss, best_test_epoch, hparams, time_elapsed = train_one_model(data, "$(exp_name)_$i"; hp...)

        results = Dict(par_dict)
        results[:best_test_loss] = best_test_loss
        results[:best_test_epoch] = best_test_epoch
        results[:time_elapsed] = time_elapsed
        results[:final_loss] = model_loss
        push!(all_results, results)
    end
    all_results_df = DataFrame(all_results)

    write_parquet(joinpath(@__DIR__, "../data/model_opt/$(exp_name).parquet"), all_results_df)
    return all_results_df
end


function plot_experiment(iterators, results)
    fig = Figure()
    for (i, key) in enumerate(keys(iterators))

        ax = Axis(fig[i, 1], xlabel=String(key))

        mask = isfinite.(results[:, :best_test_loss])

        CairoMakie.scatter!(ax, results[mask, key], results[mask, :best_test_loss], color=results[mask, :lr])
        CairoMakie.ylims!(ax, low=2.7, high = 3.2)

    end

    fig2 = Figure()
    combs = combinations(collect(keys(iterators)), 2)
    
    @show first(combs)

    for (i, (key1, key2)) in enumerate(combs)
    
        row, cols = divrem(i-1, 3)
        ax = Axis(fig2[row+1, cols+1], xlabel=String(key1), ylabel=String(key2))

        mask = isfinite.(results[:, :best_test_loss])

        CairoMakie.scatter!(ax, results[mask, key1], results[mask, key2], color=log10.(results[mask, :best_test_loss]))
    
    end



    return fig, fig2
end

iterators = Dict(
    :K => [5, 7, 9, 11, 13],
    :mlp_layer_size => [256, 512, 768],
    :lr => [0.01, 0.005, 0.001]
)

results = run_experiment(iterators)

iterators = Dict(
    :epochs => [50, 75, 100],
    :mlp_layer_size => [256, 512, 768],
    :lr => [0.01, 0.005, 0.001]
)
results = run_experiment(iterators)
plot_experiment(iterators, results)



iterators = Dict(
    :mlp_layer_size => [256, 512, 768, 1024],
    :mlp_layers => [2, 3],
    :lr => [0.007, 0.01, 0.02, 0.03],
)
results = run_experiment(iterators)
fig1, fig2 = plot_experiment(iterators, results)

fig1

hyperparams_default = Dict(
        :K => 12,
        :epochs => 50,
        :lr => 0.001,
        :mlp_layer_size => 512,
        :mlp_layers => 2,
        :dropout => 0,
        :non_linearity => :relu,
        :batch_size => 30000,
        :seed => 1,
        :l2_norm_alpha => 0,
        :adam_beta_1 => 0.9,
        :adam_beta_2 => 0.999
)


iterators = Dict(
    :K => [5, 7, 9, 11, 13],
    :dropout => [0, 0.1, 0.3],
    :lr => [0.007, 0.01, 0.02, 0.03],
)
results = run_experiment(iterators)
fig1, fig2 = plot_experiment(iterators, results)

fig1

hyperparams_default = Dict(
        :K => 12,
        :epochs => 70,
        :lr => 0.007,
        :mlp_layer_size => 512,
        :mlp_layers => 2,
        :dropout => 0.1,
        :non_linearity => :relu,
        :batch_size => 30000,
        :seed => 1,
        :l2_norm_alpha => 0,
        :adam_beta_1 => 0.9,
        :adam_beta_2 => 0.999
)


iterators = Dict(
    :K => [13, 15, 17],
    :lr => [0.005, 0.007, 0.01],
    :epochs=> [ 70, 100]
)
results = run_experiment(iterators)
fig1, fig2 = plot_experiment(iterators, results)

fig1

train_one_model(data, "full", hyperparams_default...)


begin



fig
end

begin
fig = Figure()
ax = Axis(fig[1,1])
CairoMakie.scatter!(ax, all_results_df[:, :K], all_results_df[:, :mlp_layer_size], color=log10.(all_results_df[:, :final_loss]))
ax = Axis(fig[1,2])
CairoMakie.scatter!(ax, all_results_df[:, :K], all_results_df[:, :lr], color=all_results_df[:, :final_loss])
x = Axis(fig[1,3])
CairoMakie.scatter!(ax, all_results_df[:, :mlp_layer_size], all_results_df[:, :lr], color=all_results_df[:, :final_loss])
fig
end


hyperparams_default

bsizes = [10000, 20000, 30000, 40000, 80000]
times = []
for bsize in bsizes
    hyperparams = Dict(
        :K => 12,
        :epochs => 15,
        :lr => 0.001,
        :mlp_layer_size => 768,
        :mlp_layers => 2,
        :dropout => 0,
        :non_linearity => :relu,
        :batch_size => bsize,
        :seed => 1,
        :l2_norm_alpha => 0
    )

    _, _, _, t = train_one_model(data, "exp_1"; hyperparams...)
    push!(times, t)
end
times


train_models(fnames_casc, "rq_spline_model_casc_l2_0", 30000)





train_model(fnames_casc, "rq_spline_model_casc_l2_0", 7000)

fnames_muon = [
    joinpath(@__DIR__, "../data/photon_table_bare_muon.hd5"),
]
train_model(fnames_muon, "rq_spline_model_muon_l2_0", 100)



begin
    @load joinpath(@__DIR__, "../assets/rq_spline_model_l2_0_5_FNL.bson") model hparams opt tf_dict
    @load joinpath(@__DIR__, "../assets/rq_spline_model_l2_0_5_BEST.bson") model
end

h5open(fnames[2], "r") do fid
    df = create_pmt_table(fid["pmt_hits"]["dataset_4150"])
    plot_tres = df[:, :tres]
    plot_labels, _ = preproc_labels(df, 16, tf_dict)


    n_per_pmt = combine(groupby(df, :pmt_id), nrow => :n)
    max_i = argmax(n_per_pmt[:, :n])


    pmt_plot = n_per_pmt[max_i, :pmt_id]
    mask = df.pmt_id .== pmt_plot

    plot_tres_m = plot_tres[mask]
    plot_labels_m = plot_labels[:, mask]


    t_plot = -5:0.1:50
    l_plot = repeat(Vector(plot_labels[:, 1]), 1, length(t_plot))


    fig = Figure()

    ax1 = Axis(fig[1, 1], title="Shape", xlabel="Time Residual (800nm) (ns)", ylabel="PDF")
    ax2 = Axis(fig[1, 2], title="Shape + Counts", xlabel="Time Residual (800nm) (ns)", ylabel="Counts")

    log_pdf_eval, log_expec = cpu(model)(t_plot, l_plot, true)

    lines!(ax1, t_plot, exp.(log_pdf_eval))
    hist!(ax1, plot_tres[mask], bins=-5:1:50, normalization=:pdf)

    lines!(ax2, t_plot, exp.(log_pdf_eval) .* exp.(log_expec))
    hist!(ax2, plot_tres[mask], bins=-5:1:50)

    println("$(exp(log_expec[1])), $(sum(mask))")

    fig
end
