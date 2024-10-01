using JLD2
using FHist
using PhysicsTools
using PhotonPropagation
using Distributions
using StaticArrays
using Rotations
using LinearAlgebra
using CairoMakie
import SymbolicRegression: calculate_pareto_frontier, compute_complexity, string_tree
using SymbolicUtils
using SymbolicRegression
using MLUtils
using HDF5
using DataFrames
using StatsBase
using PoissonRandom
using Dates
using PairPlots
using LoopVectorization
using Bumper
using SpecialFunctions
using Glob
using Latexify
using TOML
using SHA
using CSV
using LossFunctions
using Random
using Symbolics

include("utils.jl")

#=data = load_data(glob("*.hd5", "/home/wecapstor3/capn/capn100h/snakemake/photon_tables_perturb/extended/hits/"))

e_sel = (data.energy .> 1E4) .&& (data.energy .< 3E4) .&& ((data.distance .> 15) .&& (data.distance .< 20))
dsel = data[e_sel, :]

h = Hist2D((dsel.delta_phi, dsel.theta_dir), binedges=(-2π:0.2:2π, 0:0.2:π), weights=dsel.nhits)
plot(h, colorscale=log10)

h = Hist2D((dsel.delta_phi, dsel.theta_pos ), binedges=(-2π:0.2:2π, 0:0.2:π), weights=dsel.nhits)
plot(h, colorscale=log10)

h = Hist2D((dsel.theta_dir, dsel.theta_pos), binedges=(0:0.2:π, 0:0.2:π), weights=dsel.nhits)
plot(h, colorscale=log10)

#jldsave("/home/wecapstor3/capn/capn100h/symbolic_regression/sr_dataset_full.jld2", data=data)
=#
runs = glob("sr_e*", "/home/wecapstor3/capn/capn100h/symbolic_regression/")

finished_runs = []

for run in runs
    if isfile(joinpath(run, "sr_state.jld2"))
        push!(finished_runs, run)
    end
end

show(collect(zip(eachindex(finished_runs), basename.(finished_runs))))

data = load("/home/wecapstor3/capn/capn100h/symbolic_regression/sr_dataset_full.jld2")["data"]

any(data.variance[data.nhits .> 0] .==0 )

(data.nhits ./ sqrt.(data.variance))[(data.nhits .<= 1E-10) .&& (data.nhits .> 0)]


variable_configs = ["e_dist_abs_sca_True_logl2_400_True", "e_dist_abs_sca_2_True_logl2_400_True", "e_dist_abs_sca_3_True_logl2_400_True"]
fig = Figure(size=(400 * length(variable_configs), 1600))
for (vix, variable_config) in enumerate(variable_configs)
    sel_run = "/home/wecapstor3/capn/capn100h/symbolic_regression/sr_$(variable_config)/"
    if !isfile(joinpath(sel_run, "sr_state.jld2"))
        continue
    end

    X, y, w, sr_summary = read_results(sel_run)

    six = searchsortedfirst(sr_summary.complexity, 15)

    eq_sel = sr_summary.equation[six]
    eq_text_simpl = string(simplify(sr_summary.eq_sym[six]))
    println("Equation at index $six (complexity: $(sr_summary.complexity[six]) for $variable_config: $(eq_text_simpl)")


    ax = Axis(fig[1, vix], yscale=log10, xlabel="Complexity", ylabel="Loss", title="Variable Config: $variable_config")
    lines!(ax, sr_summary.complexity, sr_summary.train_loss)
    lines!(ax, sr_summary.complexity, sr_summary.val_loss)
    ax = Axis(fig[2, vix], yscale=log10, xscale=log10, xlabel="Truth", ylabel="Prediction")
    plot_pred_target(X, y, eq_sel, ax)
    ax = Axis(fig[3, vix], yscale=log10, xscale=log10, xlabel="Distance", ylabel="Prediction")
    plot_distance_energy(X, y, eq_sel, [(3, 1.0), (4, 1.0)], ax)
    ax = Axis(fig[4, vix], yscale=log10, xscale=log10, xlabel="Energy", ylabel="Number of Hits")
    plot_energy_distance(X, y, eq_sel, [(3, 1.0), (4, 1.0)], ax)


end
fig




const VARIABLE_MAPPING = Dict(
    :energy => 1,
    :distance => 2,
    :abs_scale => 3,
    :sca_scale => 4,
    :delta_phi => 5,
    :theta_pos => 6,
    :theta_dir => 7,
)

const VARIABLE_NAMES = Dict(
    :energy => "Energy",
    :distance => "Distance",
    :abs_scale => "Abs Scale",
    :sca_scale => "Sca Scale",
    :delta_phi => "Delta Phi",
    :theta_pos => "Theta Pos",
    :theta_dir => "Theta Dir",
)


function make_slice_plot(config_dict::Dict, grid_location, X, y, eq)
    plot_variable = config_dict[:plot_variable]
    plot_range = config_dict[:plot_range]
    xscale = config_dict[:xscale]
    yscale = config_dict[:yscale]
    fixed_vars = config_dict[:fixed_vars]
    slice_variable = config_dict[:slice_variable]
    slice_points = config_dict[:slice_points]
    slice_log = config_dict[:slice_log]
    variable_cuts = config_dict[:variable_cuts]

    fixed_vars_parsed = [(VARIABLE_MAPPING[var], val) for (var, val) in fixed_vars]
    plot_var_parsed = [VARIABLE_MAPPING[plot_variable], plot_range]
    slice_var_parsed = [VARIABLE_MAPPING[slice_variable], slice_points, slice_log]


    ax = Axis(
        grid_location,
        yscale=yscale,
        xscale=xscale,
        xlabel=VARIABLE_NAMES[plot_variable],
        ylabel="Prediction")

    selection = ones(Bool, size(X, 2))

    for (vname, vcut) in variable_cuts
        vix = VARIABLE_MAPPING[vname]

        if vix > size(X, 1)
            continue
        end

        selection = selection .& (X[vix, :] .> vcut[1]) .& (X[vix, :] .< vcut[2])
        push!(fixed_vars_parsed, (vix, mean(vcut)))
    end

    plot_variable_slice(X[:, selection], y[selection], eq, fixed_vars_parsed, plot_var_parsed, slice_var_parsed, ax)

    ylims!(ax, config_dict[:ylims])

    return ax
end


function make_ratio_plot(config_dict::Dict, grid_location, X, y, eq)
    plot_variable_x = config_dict[:plot_variable_x]
    plot_range_x = config_dict[:plot_range_x]
    plot_variable_y = config_dict[:plot_variable_y]
    plot_range_y = config_dict[:plot_range_y]
    xscale = config_dict[:xscale]
    yscale = config_dict[:yscale]
    fixed_vars = config_dict[:fixed_vars]
    variable_cuts = config_dict[:variable_cuts]


    fixed_vars_parsed = [(VARIABLE_MAPPING[var], val) for (var, val) in fixed_vars]
    plot_var_parsed_x = [VARIABLE_MAPPING[plot_variable_x], plot_range_x]
    plot_var_parsed_y = [VARIABLE_MAPPING[plot_variable_y], plot_range_y]


    ax = Axis(
        grid_location[1, 1],
        yscale=yscale,
        xscale=xscale,
        xlabel=VARIABLE_NAMES[plot_variable_x],
        ylabel=VARIABLE_NAMES[plot_variable_y],)

    selection = ones(Bool, size(X, 2))

    for (vname, vcut) in variable_cuts
        vix = VARIABLE_MAPPING[vname]

        if vix > size(X, 1)
            continue
        end

        selection = selection .& (X[vix, :] .> vcut[1]) .& (X[vix, :] .< vcut[2])
        push!(fixed_vars_parsed, (vix, mean(vcut)))
    end

    _, hm = plot_ratio(X[:, selection], y[selection], eq, fixed_vars_parsed, plot_var_parsed_x, plot_var_parsed_y, ax)

    Colorbar(grid_location[1, 2], hm)

    return ax
end



function parse_plot_config(config_dict::Dict, grid_location, X, y, w, eq_sel)
    plot_type = config_dict[:plot_type]
   
    if plot_type == :slice
        return make_slice_plot(config_dict, grid_location, X, y, eq_sel)
    elseif plot_type == :ratio
        return make_ratio_plot(config_dict, grid_location, X, y, eq_sel)
    else
        error("Unknown plot type: $plot_type")
    end
end

plot_configs = [
    Dict(
        :plot_type => :slice,
        :plot_variable => :distance,
        :plot_range => 1:0.1:200,
        :xscale => log10,
        :yscale => log10,
        :ylims => (1E-2, 1E7),
        :fixed_vars => [:abs_scale => 1.0, :sca_scale => 1.0],
        :slice_variable => :energy,
        :slice_points => range(2, 7.5, 10),
        :slice_log => true,
        :variable_cuts => Dict(:delta_phi => (0.4, 0.6), :theta_pos => (0.9, 1.1), :theta_dir => (1.3, 1.5))
        ),

    Dict(
        :plot_type => :ratio,
        :plot_variable_x => :distance,
        :plot_range_x => 10 .^range(0, log10(200), 7),
        :plot_variable_y => :energy,
        :plot_range_y => 10 .^range(2, 7, 7),
        :xscale => log10,
        :yscale => log10,
        :fixed_vars => [:abs_scale => 1.0, :sca_scale => 1.0],
        :variable_cuts => Dict(:delta_phi => (0.4, 0.6), :theta_pos => (0.9, 1.1), :theta_dir => (1.3, 1.5))
        ),

    Dict(
        :plot_type => :slice,
        :plot_variable => :delta_phi,
        :plot_range => -2π:0.01:2π,
        :xscale => identity,
        :yscale => log10,
        :ylims => (1E-2, 1E7),
        :fixed_vars => [:abs_scale => 1.0, :sca_scale => 1.0],
        :slice_variable => :energy,
        :slice_points => range(2, 7, 10),
        :slice_log => true,
        :variable_cuts => Dict(:distance => (10, 15), :theta_pos => (0.9, 1.1), :theta_dir => (1.3, 1.5))
        ),

    Dict(
        :plot_type => :ratio,
        :plot_variable_x => :delta_phi,
        :plot_range_x => range(-2π, 2π, 10),
        :plot_variable_y => :energy,
        :plot_range_y => 10 .^range(2, 7, 7),
        :xscale => identity,
        :yscale => log10,
        :fixed_vars => [:abs_scale => 1.0, :sca_scale => 1.0],
        :variable_cuts => Dict(:distance => (10, 15), :theta_pos => (0.9, 1.1), :theta_dir => (1.3, 1.5))
        ),
]



sel_run = "/home/wecapstor3/capn/capn100h/symbolic_regression/sr_e_dist_phi_abs_sca_True_logl1_400_True/"
X, y, w, sr_summary = read_results(sel_run)
six = searchsortedfirst(sr_summary.complexity, 45)
eq_sel = sr_summary.equation[six]

sr_summary


  # finally walk and apply

@show sr_summary.eq_str[six]

begin
    fig = Figure(size=(400*3, 400*4))
    ax = Axis(fig[1, 1], yscale=log10, xlabel="Complexity", ylabel="Loss")
    lines!(ax, sr_summary.complexity, sr_summary.train_loss)
    lines!(ax, sr_summary.complexity, sr_summary.val_loss)
    scatter!(ax, sr_summary.complexity[six], sr_summary.train_loss[six], color=:red)
    ax = Axis(fig[1, 2], yscale=log10, xscale=log10, xlabel="Truth", ylabel="Prediction")
    plot_pred_target(X, y, eq_sel, ax)

    parse_plot_config(plot_configs[1], fig[2, 1], X, y, w, eq_sel)
    parse_plot_config(plot_configs[2], fig[2, 2], X, y, w, eq_sel)
    parse_plot_config(plot_configs[3], fig[3, 1], X, y, w, eq_sel)
    parse_plot_config(plot_configs[4], fig[3, 2], X, y, w, eq_sel)

end
fig



vars = Symbolics.get_variables(sr_summary.eq_sym[six])

eq_rw = Rewriters.Postwalk(literaltoreal)(sr_summary.eq_sym[six])
erw = Rewriters.Postwalk(literaltoreal).(vars)
D = Differential(erw)




f_expr = build_function(Symbolics.toexpr(Symbolics.gradient(eq_rw, erw)), erw)
fev = eval(f_expr[1])

fev([1, 2, 3])

D(sr_summary.eq_sym[six])





begin
fig = Figure(size=(400*3, 400*4))
ax = Axis(fig[1, 1], yscale=log10, xlabel="Complexity", ylabel="Loss")
lines!(ax, sr_summary.complexity, sr_summary.train_loss)
lines!(ax, sr_summary.complexity, sr_summary.val_loss)
scatter!(ax, sr_summary.complexity[six], sr_summary.train_loss[six], color=:red)
ax = Axis(fig[1, 2], yscale=log10, xscale=log10, xlabel="Truth", ylabel="Prediction")
plot_pred_target(X, y, eq_sel, ax)


yscaling = Makie.Symlog10(1E1)
yscaling = log10



a = Figure()


if size(X, 1) == 7
    ax = Axis(fig[2, 1], yscale=yscaling, xscale=log10, xlabel="Distance", ylabel="Prediction")
    sel = (X[5, :] .< 0.1) .&& (X[5, :] .> -0.1) .&& (X[6, :] .< 1.3) .&& (X[6, :] .> -1.5) .&& (X[7, :] .< 1.1) .&& (X[7, :] .> 0.9)
    plot_variable_slice(X[:, sel], y[sel], eq_sel, [(3, 1.0), (4, 1.0), (5, 0.0), (6, 1.4),(7, 1.0)], [2, 1:1:200], [1, range(2, 7, 7), true], ax)
    ax = Axis(fig[2, 2], yscale=yscaling, xscale=log10, xlabel="Energy", ylabel="Number of Hits")
    plot_variable_slice(X[:, sel], y[sel], eq_sel, [(3, 1.0), (4, 1.0), (5, 0.0), (6, 1.4),(7, 1.0)], [1, 10 .^(2:0.1:7.5)], [2, range(0, log10(200), 7), true], ax)
    
    ax = Axis(fig[1, 3], yscale=yscaling, xlabel="Abs Scale", ylabel="Number of Hits")
    sel = (X[2, :] .> 20) .&& (X[2, :] .< 30) .&& (X[5, :] .< 0.1) .&& (X[5, :] .> -0.1) .&& (X[6, :] .< 1.3) .&& (X[6, :] .> -1.5) .&& (X[7, :] .< 1.1) .&& (X[7, :] .> 0.9)
    plot_variable_slice(X[:, sel], y[sel], eq_sel, [(2, 25), (4, 1.0), (5, 0.0), (6, 1.4), (7, 1.0)], [3, 0.8:0.01:1.2], [1, range(2, 7, 7), true], ax)
    ax = Axis(fig[2, 3], yscale=yscaling, xlabel="Sca Scale", ylabel="Number of Hits")
    sel = (X[2, :] .> 20) .&& (X[2, :] .< 30) .&& (X[5, :] .< 0.1) .&& (X[5, :] .> -0.1) .&& (X[6, :] .< 1.3) .&& (X[6, :] .> -1.5) .&& (X[7, :] .< 1.1) .&& (X[7, :] .> 0.9)
    plot_variable_slice(X[:, sel], y[sel], eq_sel, [(2, 25), (3, 1.0), (5, 0.0), (6, 1.4), (7, 1.0)], [4, 0.8:0.01:1.2], [1, range(2, 7, 7), true], ax)

    ax = Axis(fig[3, 1], yscale=yscaling, xlabel="Delta Phi", ylabel="Number of Hits")
    sel = (X[2, :] .> 20) .&& (X[2, :] .< 30) .&& (X[6, :] .< 1.3) .&& (X[6, :] .> 1.1) .&& (X[7, :] .< 1.1) .&& (X[7, :] .> 0.9)
    plot_variable_slice(X[:, sel], y[sel], eq_sel, [(2, 25), (3, 1.0), (4, 1.0), (6, 1.2), (7, 1.0)], [5, -2π:0.01:2π], [1, range(2, 7, 7), true], ax)

    ax = Axis(fig[3, 2], yscale=yscaling, xlabel="Theta Dir", ylabel="Number of Hits")
    sel = (X[2, :] .> 20) .&& (X[2, :] .< 30) .&& (X[5, :] .< 0.1) .&& (X[5, :] .> -0.1)
    plot_variable_slice(X[:, sel], y[sel], eq_sel, [(2, 25), (3, 1.0), (4, 1.0), (5, 0.0)], [6, 0:0.01:π], [1, range(2, 7, 7), true], ax)


    ax = Axis(fig[4, 1], yscale=yscaling, xlabel="Theta Pos", ylabel="Number of Hits")
    sel = (X[2, :] .> 20) .&& (X[2, :] .< 30) .&& (X[5, :] .< 0.1) .&& (X[5, :] .> -0.1) .&& (X[6, :] .< 1.1) .&& (X[6, :] .> 0.9)
    plot_variable_slice(X[:, sel], y[sel], eq_sel, [(2, 25), (3, 1.0), (4, 1.0), (5, 0.0)], [7, 0:0.01:π], [1, range(2, 7, 7), true], ax)

    ax = Axis(fig[4, 2], yscale=yscaling, xlabel="Theta Pos", ylabel="Number of Hits")
    sel = (X[2, :] .> 20) .&& (X[2, :] .< 30) .&& (X[5, :] .< 0.1) .&& (X[5, :] .> -0.1)  .&& (X[1, :] .< 5E4) .&& (X[1, :] .> 3E4)
    plot_variable_slice(X[:, sel], y[sel], eq_sel, [(1, 4E4), (2, 25), (3, 1.0), (4, 1.0), (5, 0.0)], [7, 0:0.01:π], [6, range(0, 2π, 8), true], ax)

    

elseif size(X, 1) == 6
    ax = Axis(fig[2, 1], yscale=yscaling, xscale=log10, xlabel="Distance", ylabel="Prediction")
    sel = (X[5, :] .< 3.3) .&& (X[5, :] .> 3.1) .&& (X[6, :] .< 1.3) .&& (X[6, :] .> 1.1)
    plot_variable_slice(X[:, sel], y[sel], eq_sel, [(3, 1.0), (4, 1.0), (5, 3.2), (6, 1.2)], [2, 1:1:200], [1, range(2, 7, 7), true], ax)
    ax = Axis(fig[2, 2], yscale=yscaling, xscale=log10, xlabel="Energy", ylabel="Number of Hits")
    plot_variable_slice(X[:, sel], y[sel], eq_sel, [(3, 1.0), (4, 1.0), (5, 0.0), (6, 1.2)], [1, 10 .^(2:0.1:7.5)], [2, range(0, log10(200), 7), true], ax)
    
    ax = Axis(fig[1, 3], yscale=yscaling, xlabel="Abs Scale", ylabel="Number of Hits")
    sel = (X[2, :] .> 20) .&& (X[2, :] .< 30) .&& (X[5, :] .< 0.1) .&& (X[5, :] .> -0.1) .&& (X[6, :] .< 1.3) .&& (X[6, :] .> 1.1)
    plot_variable_slice(X[:, sel], y[sel], eq_sel, [(2, 25), (4, 1.0), (5, 0.0), (6, 1.2)], [3, 0.8:0.01:1.2], [1, range(2, 7, 7), true], ax)
    ax = Axis(fig[2, 3], yscale=yscaling, xlabel="Sca Scale", ylabel="Number of Hits")
    sel = (X[2, :] .> 20) .&& (X[2, :] .< 30) .&& (X[5, :] .< 0.1) .&& (X[5, :] .> -0.1) .&& (X[6, :] .< 1.3) .&& (X[6, :] .> 1.1)
    plot_variable_slice(X[:, sel], y[sel], eq_sel, [(2, 25), (3, 1.0), (5, 0.0), (6, 1.2)], [4, 0.8:0.01:1.2], [1, range(2, 7, 7), true], ax)

    ax = Axis(fig[3, 1], yscale=yscaling, xlabel="Delta Phi", ylabel="Number of Hits")
    sel = (X[2, :] .> 20) .&& (X[2, :] .< 30) .&& (X[6, :] .< 1.3) .&& (X[6, :] .> 1.1)
    plot_variable_slice(X[:, sel], y[sel], eq_sel, [(2, 25), (3, 1.0), (4, 1.0), (6, 1.2)], [5, -2π:0.01:2π], [1, range(2, 7, 7), true], ax)

    ax = Axis(fig[3, 2], yscale=yscaling, xlabel="Theta Dir", ylabel="Number of Hits")
    sel = (X[2, :] .> 20) .&& (X[2, :] .< 30) .&& (X[5, :] .< 0.1) .&& (X[5, :] .> -0.1)
    plot_variable_slice(X[:, sel], y[sel], eq_sel, [(2, 25), (3, 1.0), (4, 1.0), (5, 0.0)], [6, 0:0.01:π], [1, range(2, 7, 7), true], ax)
elseif size(X, 1) == 5
    ax = Axis(fig[2, 1], yscale=yscaling, xscale=log10, xlabel="Distance", ylabel="Prediction")
    sel = (X[5, :] .< 0.1) .&& (X[5, :] .> -0.1)
    plot_variable_slice(X[:, sel], y[sel], eq_sel, [(3, 1.0), (4, 1.0), (5, 0.0)], [2, 1:1:200], [1, range(2, 7, 7), true], ax)
    ax = Axis(fig[2, 2], yscale=yscaling, xscale=log10, xlabel="Energy", ylabel="Number of Hits")
    plot_variable_slice(X[:, sel], y[sel], eq_sel, [(3, 1.0), (4, 1.0), (5, 0.0)], [1, 10 .^(2:0.1:7.5)], [2, range(0, log10(200), 7), true], ax)
    
    ax = Axis(fig[1, 3], yscale=yscaling, xlabel="Abs Scale", ylabel="Number of Hits")
    sel = (X[2, :] .> 20) .&& (X[2, :] .< 30) .&& (X[5, :] .< 0.1) .&& (X[5, :] .> -0.1)
    plot_variable_slice(X[:, sel], y[sel], eq_sel, [(2, 25), (4, 1.0), (5, 0.0)], [3, 0.8:0.01:1.2], [1, range(2, 7, 7), true], ax)
    ax = Axis(fig[2, 3], yscale=yscaling, xlabel="Sca Scale", ylabel="Number of Hits")
    sel = (X[2, :] .> 20) .&& (X[2, :] .< 30) .&& (X[5, :] .< 0.1) .&& (X[5, :] .> -0.1)
    plot_variable_slice(X[:, sel], y[sel], eq_sel, [(2, 25), (3, 1.0), (5, 0.0)], [4, 0.8:0.01:1.2], [1, range(2, 7, 7), true], ax)

    ax = Axis(fig[3, 1], yscale=yscaling, xlabel="Delta Phi", ylabel="Number of Hits")
    sel = (X[2, :] .> 20) .&& (X[2, :] .< 30)
    plot_variable_slice(X[:, sel], y[sel], eq_sel, [(2, 25), (3, 1.0), (4, 1.0)], [5, -2π:0.01:2π], [1, range(2, 7, 7), true], ax)

    sel = (X[1, :] .> 1E3) .&& (X[1, :] .< 3E3)
    ax = Axis(fig[3, 2][1, 1], xlabel="Distance", ylabel="Delta Phi")
    _, hm = plot_ratio(X[:, sel], y[sel], eq_sel, [(1, 2E3), (3, 1.0), (4, 1.0), ], [2, 1:5:50], [5, range(-2π, 2π, 15)], ax)
    Colorbar(fig[3, 2][1, 2], hm)
else

    fig = Figure(size=(400*3, 400*2))
    ax = Axis(fig[1, 1], yscale=log10, xlabel="Complexity", ylabel="Loss")
    lines!(ax, sr_summary.complexity, sr_summary.train_loss)
    lines!(ax, sr_summary.complexity, sr_summary.val_loss)
    scatter!(ax, sr_summary.complexity[six], sr_summary.train_loss[six], color=:red)
    ax = Axis(fig[1, 2], yscale=log10, xscale=log10, xlabel="Truth", ylabel="Prediction")
    plot_pred_target(X, y, eq_sel, ax)
    ax = Axis(fig[2, 1], yscale=log10, xscale=log10, xlabel="Distance", ylabel="Prediction")
    plot_variable_slice(X, y, eq_sel, [(3, 1.0), (4, 1.0)], [2, 1:1:200], [1, range(2, 7, 7), true], ax)
    ax = Axis(fig[2, 2], yscale=log10, xscale=log10, xlabel="Energy", ylabel="Number of Hits")
    plot_variable_slice(X, y, eq_sel, [(3, 1.0), (4, 1.0)], [1, 10 .^(2:0.1:7.5)], [2, range(0, log10(200), 7), true], ax)

    ax = Axis(fig[1, 3], yscale=log10, xscale=log10, xlabel="Abs Scale", ylabel="Number of Hits")
    sel = (X[2, :] .> 20) .&& (X[2, :] .< 30)
    plot_variable_slice(X[:, sel], y[sel], eq_sel, [(2, 25), (4, 1.0)], [3, 0.8:0.01:1.2], [1, range(2, 7, 7), true], ax)

    ax = Axis(fig[2, 3], yscale=log10, xscale=log10, xlabel="Sca Scale", ylabel="Number of Hits")
    sel = (X[2, :] .> 20) .&& (X[2, :] .< 30)
    plot_variable_slice(X[:, sel], y[sel], eq_sel, [(2, 25), (3, 1.0)], [4, 0.8:0.01:1.2], [1, range(2, 7, 7), true], ax)
end
fig
end

X, y, w, sr_summary = read_results("/home/wecapstor3/capn/capn100h/symbolic_regression/sr_e_dist_abs_sca_False_logl2_400_True")


hist(w)



using PhotonSurrogateModel
using PhotonPropagation
using BSON
medium = CascadiaMediumProperties(0.95, 1.0, 1.0)
rng = MersenneTwister(31338)

target = POM(SA_F32[0., 0., 0.], 1);

model_path = "/home/wecapstor3/capn/capn100h/snakemake/time_surrogate_perturb/extended/amplitude_2_FNL.bson"
model = load(model_path)[:model]

hparams = load(model_path)[:hparams]

input_size = size(model.embedding.layers[1].weight, 2)
feat_buffer = create_input_buffer(input_size, 16, 1);


pmt_positions = get_pmt_positions(target, RotMatrix3(I))

function predict_sr(target_pos, pmt_pos, pos, dir, energy)

    rpos = pos - target_pos
    distance = norm(rpos)
    theta_pos, theta_dir, delta_phi = transform_input_old(pmt_pos, rpos, dir)
    abs_scale = 1.0
    sca_scale = 1.0

    return sqrt(theta_pos) * ((((((((energy / 10.27729310492359) + 2.281475231675802) / ((((cos(delta_phi * 1.0037058977062525) - -1.2592755464425704) * (theta_pos ^ (theta_dir - 0.6116821727100822))) ^ ((sqrt(theta_pos) * 1.3273037859357315) ^ 1.4293860003709848)) ^ theta_dir)) ^ 0.8775619398688311) / (sqrt((((3.3892622706077162 / distance) ^ (((0.8035604305233338 / 1.1218453402020847) - ((theta_pos ^ theta_pos) * 0.7477722525620631)) + theta_dir)) + 0.2792340028894907) / abs_scale) ^ theta_dir)) + ((1.329906979942511 * 1.2530095663885246) / theta_pos)) / 0.27935327069921206) / exp_minus(sqrt(distance) / -0.8192681986376636))
end

particle_azimuth = 0.9
particle_zenith = 0.7

particle_pos = SA[-10.0, 15., 25.]
particle_dir = sph_to_cart(particle_zenith,particle_azimuth)
particle_energy = 7e4
particle_type = PEPlus
azimuths = 0:0.05:2π


amps = Float64[]
amps_sr = Float64[]
for azimuth in azimuths

    particle_dir = sph_to_cart(particle_zenith,azimuth)
    create_model_input!(
        particle_pos,
        particle_dir,
        particle_energy,
        target.shape.position, 
        feat_buffer,
        model.transformations,
        abs_scale=1.0,
        sca_scale=1.0)

    feat_buffer[:, 1] .= fourier_input_mapping(feat_buffer[1:10, 1], model.embedding_matrix * hparams.fourier_gaussian_scale)


    push!(amps, sum(exp.(model(feat_buffer[1:128, 1]))))

    amp_sr = 0
    for ppos in pmt_positions
        amp_sr += predict_sr(target.shape.position, ppos, particle_pos, particle_dir, particle_energy)
    end

    push!(amps_sr, amp_sr)
end

fig = Figure()
ax = Axis(fig[1, 1], yscale=log10)
lines!(ax, azimuths, amps)
lines!(ax, azimuths, amps_sr)
fig


distances = 1:0.1:200

particle_pos_theta = 0.94
particle_pos_phi = 1.3
particle_azimuth = 0.9
particle_zenith = 1.7
particle_dir = sph_to_cart(particle_zenith,particle_azimuth)

amps = Float64[]
amps_sr = Float64[]

for dist in distances
    particle_pos = sph_to_cart(particle_pos_theta, particle_pos_phi) * dist

    create_model_input!(
        particle_pos,
        particle_dir,
        particle_energy,
        target.shape.position, 
        feat_buffer,
        model.transformations,
        abs_scale=1.0,
        sca_scale=1.0)

    feat_buffer[:, 1] .= fourier_input_mapping(feat_buffer[1:10, 1], model.embedding_matrix* hparams.fourier_gaussian_scale)


    push!(amps, sum(exp.(model(feat_buffer[1:128, 1]))))

    amp_sr = 0
    for ppos in pmt_positions
        amp_sr += predict_sr(target.shape.position, ppos, particle_pos, particle_dir, particle_energy)
    end

    push!(amps_sr, amp_sr)
end



fig = Figure()
ax = Axis(fig[1, 1], yscale=log10)
lines!(ax, distances, amps)
lines!(ax, distances, amps_sr)
fig




#=
#func(energy, distance, abs_scale, delta_phi) = exp_minus(cos(delta_phi)) * (energy / ((531.98652551418 / distance) + square((cos(delta_phi) - -1.7549204269503753) * distance)))
# ((energy ^ 0.86009) / (square(2.6646 - distance) + 45.157)) * square(abs_scale)
#y = (energy / (square(5.0384 - distance) + 41.002)) ^ 0.87882

hof_files = glob("*.csv", "/home/wecapstor3/capn/capn100h/symbolic_regression/sr_e_dist_abs_sca_False_logl1_100/")
hof_file = sort(hof_files, by=filename -> mtime(filename))[end]
sr_result_csv = CSV.read(hof_file, DataFrame)

cfile = "/home/saturn/capn/capn100h/julia_dev/NeutrinoTelescopes/scripts/surrogate_models/symbolic_regression/configs/sr_e_dist_abs_sca.toml"

config = TOML.parsefile(cfile)

Xs, ys, weight = prepare_data(cfile)

Xs
#X, y, weights = prepare_data(config)

X
=#



#jldsave("/home/wecapstor3/capn/capn100h/sr_dataset_e_dist.jld2", X=X, y=y_resampled)


 selected_data = apply_selection(data, selection)

    # Split the data into training and testing sets
    if nrow(selected_data) > 40000
        wfunc = make_weight_func(selected_data)
        selected_data = weighted_subsample(selected_data, 40000, wfunc)
        train_data = selected_data[1:20000, :]
        test_data = selected_data[20001:end, :]


sr_result = load("/home/wecapstor3/capn/capn100h/sr_state_2024-09-19T10:07:01.488.jld2")

hof = sr_result["hof"]
opt = sr_result["options"]

dominating = calculate_pareto_frontier(hof)
for (i, member) in enumerate(dominating)
    complexity = compute_complexity(member, opt)
    loss = member.loss
    string = string_tree(member.tree, opt, variable_names=["energy", "distance", "abs_scale"])

    println("$i\t$(complexity)\t$(loss)\t$(string)")
end

eq_sel = dominating[13].tree

latexify(string(simplify(node_to_symbolic(eq_sel, opt)))) |> render

latexify(string(node_to_symbolic(eq_sel, opt))) |> render


plot_result(x -> eval_tree_array(eq_sel, x, opt)[1], X, y)

#func(energy, distance) = ((energy / ((distance * distance) + 33.99)) ^ 0.92311) * 0.38275
#yeval = func.(dsel.energy, dsel.distance)







yeval = func(X)


h2d = Hist2D((y, yeval), binedges=(10 .^ (-2:0.1:6), 10 .^ (-2:0.1:6)))
h2d.bincounts ./= sum(h2d.bincounts, dims=2)

fig = Figure()


fig



xsel = ones(size(X, 2), Bool)

if !isnothing(fixed_vars)
    for (var, (minval, maxval)) in fixed_vars
        xsel .&= (X[var, :] .> minval) .& (X[var, :] .< maxval)
    end
end 









#pairplot((nhits=asinh.(data[:, :nhits]), logE=log10.(data[:, :energy]), theta_dir=data[:, :theta_dir], theta_pos=data[:, :theta_pos], delta_phi=abs.(data[:, :delta_phi]), abs_scale=data[:, :abs_scale], sca_scale=data[:, :sca_scale], distance=log10.(data[:, :distance])))



dsub = weighted_subsample(data, 20000, weight_func)
e_hist = Hist1D(log10.(dsub[:, :energy]), binedges=2:0.1:7)
plot(e_hist)

hist(dsub[:, :energy], bins=50)

plot(e_hist)

d_hist = Hist1D((dsub[:, :distance]))
plot(d_hist)

X = hcat(
    dsub.energy,
    dsub.distance,
    dsub.theta_pos,
    dsub.theta_dir,
    dsub.delta_phi,
    dsub.abs_scale,
    dsub.sca_scale,
)'
y = dsub.nhits

y_resampled = pois_rand.(y)

scatter(X[2, :], asinh.(y_resampled), color=log10.(X[1, :]))


hist(X[2, :], bins=100)


fig,ax, s = scatter(dsub.distance, asinh.(dsub.nhits), color=log10.(dsub.energy))
Colorbar(fig[1, 2], s)
fig



jldsave("/home/wecapstor3/capn/capn100h/sr_dataset_flat.jld2", X=X, y=y_resampled)


sr_result = load("/home/wecapstor3/capn/capn100h/sr_state_")

hof = sr_result["hof"]
opt = sr_result["options"]

dominating = calculate_pareto_frontier(hof)

println("Complexity\tMSE\tEquation")



for member in dominating
    complexity = compute_complexity(member, opt)
    loss = member.loss
    string = string_tree(member.tree, opt, variable_names=["energy", "distance", "theta_pos", "theta_dir", "delta_phi"])

    println("$(complexity)\t$(loss)\t$(string)")
end

trees = [member.tree for member in dominating]

fig = Figure()
ax = Axis(fig[1, 1], xscale=log10)
xs = 10 .^(-7:0.1:1)
lines!(ax, xs, poisson_loss.(xs, 0.01))
lines!(ax, xs, poisson_loss.(xs, 0.1))
lines!(ax, xs, poisson_loss.(xs, 0.8))
lines!(ax, xs, poisson_loss.(xs, 1))
#lines!(ax, xs, poisson_loss.(xs, 1.5))
lines!(ax, xs, poisson_loss.(xs, 2))
fig


poisson_loss.(2, 1)

y = data.nhits

func(energy, distance, theta_pos, theta_dir, delta_phi, abs_scale, sca_scale) =  ((abs((1.9988 ^ (log(energy) + (((-1.0944 - cos(delta_phi)) - cos(delta_phi)) + (((distance * -0.15111) + -0.37165) / 1.1237)))) + tan(-4.8558 / sca_scale)) ^ 1.1185) / (theta_pos ^ (theta_pos ^ theta_dir))) + theta_dir
yeval = func.(data.energy, data.distance, data.theta_pos, data.theta_dir, data.delta_phi, data.abs_scale, data.sca_scale)


h2d = Hist2D((y, yeval), binedges=(10 .^ (-2:0.1:6), 10 .^ (-2:0.1:6)))
h2d.bincounts ./= sum(h2d.bincounts, dims=2)

fig = Figure()
ax = Axis(fig[1, 1], yscale=log10, xscale=log10, xlabel="Truth", ylabel="Prediction")
plot!(ax, h2d)

xs = 10 .^ (-2:0.1:5)
lines!(ax, xs, xs, color=:white)
lower_q = quantile.(Poisson.(xs), 0.16)
upper_q = quantile.(Poisson.(xs), 0.84)
#lines!(ax, xs, lower_q, color=:white)
#lines!(ax, xs, upper_q, color=:white)
fig




fig = Figure()
ax = Axis(fig[1, 1], yscale=log10, xscale=log10, xlabel="Distance", ylabel="Number of Hits")
scatter!(ax, dsel.distance, dsel.nhits, color=dsel.delta_phi)

xs = 10 .^ (0:0.01:3)
lines!(ax, xs, func.(mean(e_sel), xs, mean(theta_pos_sel), mean(theta_dir_sel), mean(delta_phi_sel), mean(abs_scale_sel), mean(sca_scale_sel)), color=:red)

fig

X_eval = zeros(7, length(xs))
X_eval[1, :] .= mean(e_sel)
X_eval[2, :] .= xs
X_eval[3, :] .= mean(theta_pos_sel)
X_eval[4, :] .= mean(theta_dir_sel)
X_eval[5, :] .= mean(delta_phi_sel)
X_eval[6, :] .= mean(abs_scale_sel)
X_eval[7, :] .= mean(sca_scale_sel)


output, did_succeed = eval_tree_array(trees[end], X_eval, opt)

lines!(ax, xs, output)


poisson_loss(3, 2)

exps = 10 .^ (-2:0.1:6)

mean_lhs = [mean(poisson_loss.(e, [pois_rand(e) for _ in 1:10000])) for e in exps]






hist(poisson_loss.(1.4, [pois_rand(1.4) for _ in 1:10000]))


