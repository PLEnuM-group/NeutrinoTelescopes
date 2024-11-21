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

glob("*.hd5", "/home/wecapstor3/capn/capn100h/snakemake/photon_tables_perturb/extended/hits/")
#=
data = load_data(glob("*.hd5", "/home/wecapstor3/capn/capn100h/snakemake/photon_tables_perturb/extended/hits/"))



e_sel = (data.energy .> 1E4) .&& (data.energy .< 3E4) .&& ((data.distance .> 15) .&& (data.distance .< 20))
dsel = data[e_sel, :]

h = Hist2D((dsel.phi_dir, dsel.theta_dir), binedges=(0:0.2:2π, 0:0.2:π), weights=dsel.nhits)
plot(h, colorscale=log10)

h = Hist2D((dsel.phi_dir, dsel.theta_pos ), binedges=(0:0.2:2π, 0:0.2:π), weights=dsel.nhits)
plot(h, colorscale=log10)

h = Hist2D((dsel.theta_dir, dsel.theta_pos), binedges=(0:0.2:π, 0:0.2:π), weights=dsel.nhits)
plot(h, colorscale=log10)

jldsave("/home/wecapstor3/capn/capn100h/symbolic_regression/sr_dataset_full.jld2", data=data)
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


variable_configs = ["e_dist_abs_sca_True_logl2_600_True", "e_dist_abs_sca_2_True_logl2_600_True", "e_dist_abs_sca_3_True_logl2_600_True"]
fig = Figure(size=(400 * length(variable_configs), 1600))
for (vix, variable_config) in enumerate(variable_configs)
    sel_run = "/home/wecapstor3/capn/capn100h/symbolic_regression/sr_$(variable_config)/"
    if !isfile(joinpath(sel_run, "sr_state.jld2"))
        continue
    end

    X, y, w, sr_summary = read_results(sel_run)

    six = searchsortedfirst(sr_summary.complexity, 20)

    eq_sel = sr_summary.equation[six]
    eq_text_simpl = string(simplify(sr_summary.eq_sym[six]))
    println("Equation at index $six (complexity: $(sr_summary.complexity[six]) for $variable_config: $(eq_text_simpl)")


    ax = Axis(fig[1, vix], yscale=log10, xlabel="Complexity", ylabel="Loss", title="Variable Config: $variable_config")
    lines!(ax, sr_summary.complexity, sr_summary.train_loss)
    lines!(ax, sr_summary.complexity, sr_summary.val_loss)
    scatter!(ax, sr_summary.complexity[six], sr_summary.train_loss[six], color=:red)
    ax = Axis(fig[2, vix], yscale=log10, xscale=log10, xlabel="Truth", ylabel="Prediction")
    plot_pred_target(X, y, eq_sel, ax)
    ax = Axis(fig[3, vix], yscale=log10, xscale=log10, xlabel="Distance", ylabel="Prediction")

    plot_variable_slice(X, y, eq_sel, [(3, 1.0), (4, 1.9)], [2, 1:1:200], [1, range(2, 7.5, 10), true], ax)
    ax = Axis(fig[4, vix], yscale=log10, xscale=log10, xlabel="Energy", ylabel="Number of Hits")
    plot_variable_slice(X, y, eq_sel, [(3, 1.0), (4, 1.9)], [1, 10 .^(range(2, 7.5, 100))], [2, range(0, log10(200), 10), true], ax)


end
fig


const VARIABLE_MAPPING = Dict(
    :energy => 1,
    :distance => 2,
    :abs_scale => 3,
    :sca_scale => 4,
    :theta_pos => 5,
    :theta_dir => 6,
    :phi_dir => 7,

)

const VARIABLE_NAMES = Dict(
    :energy => "Energy",
    :distance => "Distance",
    :abs_scale => "Abs Scale",
    :sca_scale => "Sca Scale",
    :phi_dir => "Phi Dir",
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
        :variable_cuts => Dict(:phi_dir => (0.4, 0.6), :theta_pos => (0.9, 1.1), :theta_dir => (1.3, 1.5))
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
        :variable_cuts => Dict(:phi_dir => (3.0, 3.4), :theta_pos => (0.8, 1.2), :theta_dir => (1.3, 1.7))
        ),

    Dict(
        :plot_type => :slice,
        :plot_variable => :phi_dir,
        :plot_range => 0:0.01:2π,
        :xscale => identity,
        :yscale => log10,
        :ylims => (1E-2, 1E7),
        :fixed_vars => [:abs_scale => 1.0, :sca_scale => 1.0],
        :slice_variable => :theta_dir,
        :slice_points => range(0.1, π, 10),
        :slice_log => true,
        :variable_cuts => Dict(:distance => (40, 50), :theta_pos => (0.4, 0.6), :energy => (1E4, 3E4))
        ),

    Dict(
        :plot_type => :ratio,
        :plot_variable_x => :phi_dir,
        :plot_range_x => range(0, 2π, 10),
        :plot_variable_y => :energy,
        :plot_range_y => 10 .^range(2, 7, 7),
        :xscale => identity,
        :yscale => log10,
        :fixed_vars => [:abs_scale => 1.0, :sca_scale => 1.0],
        :variable_cuts => Dict(:distance => (40, 50), :theta_pos => (0.4, 0.6), :theta_dir => (1.8, 2.0))
        ),

    Dict(
        :plot_type => :slice,
        :plot_variable => :theta_dir,
        :plot_range => 0.01:0.01:π,
        :xscale => identity,
        :yscale => log10,
        :ylims => (1E-2, 1E7),
        :fixed_vars => [:abs_scale => 1.0, :sca_scale => 1.0],
        :slice_variable => :energy,
        :slice_points => range(2, 7.5, 10),
        :slice_log => true,
        :variable_cuts => Dict(:distance => (10, 15), :theta_pos => (2.1, 2.3), :phi_dir => (3.3, 3.5))
        ),

    Dict(
        :plot_type => :ratio,
        :plot_variable_x => :phi_dir,
        :plot_range_x => range(0, 2π, 10),
        :plot_variable_y => :theta_dir,
        :plot_range_y => range(0, π, 10),
        :xscale => identity,
        :yscale => identity,
        :fixed_vars => [:abs_scale => 1.0, :sca_scale => 1.0],
        :variable_cuts => Dict(:energy => (1E4, 3E4), :distance => (20, 30), :theta_pos => (0.9, 1.1), )
        ),

    Dict(
        :plot_type => :slice,
        :plot_variable => :theta_pos,
        :plot_range => 0.01:0.01:π,
        :xscale => identity,
        :yscale => log10,
        :ylims => (1E-2, 1E7),
        :fixed_vars => [:abs_scale => 1.0, :sca_scale => 1.0],
        :slice_variable => :phi_dir,
        :slice_points => range(0, 2π, 10),
        :slice_log => false,
        :variable_cuts => Dict(:energy => (3E4, 5E4), :distance => (20, 30), :theta_dir => (2.2, 2.5))
        ),

    Dict(
        :plot_type => :slice,
        :plot_variable => :theta_pos,
        :plot_range => 0.01:0.01:π,
        :xscale => identity,
        :yscale => log10,
        :ylims => (1E-2, 1E7),
        :fixed_vars => [:abs_scale => 1.0, :sca_scale => 1.0],
        :slice_variable => :theta_dir,
        :slice_points => range(0.01, π, 10),
        :slice_log => false,
        :variable_cuts => Dict(:energy => (3E4, 7E4), :distance => (20, 30), :phi_dir => (2.8, 3.0))
        ),

    Dict(
        :plot_type => :ratio,
        :plot_variable_x => :phi_dir,
        :plot_range_x => range(0.01, 2π, 10),
        :plot_variable_y => :theta_pos,
        :plot_range_y => range(0.01, π, 10),
        :xscale => identity,
        :yscale => identity,
        :fixed_vars => [:abs_scale => 1.0, :sca_scale => 1.0],
        :variable_cuts => Dict(:energy => (1E4, 3E4), :distance => (20, 30), :theta_dir => (1.9, 2.1), )
        ),
]


sel_run = "/home/wecapstor3/capn/capn100h/symbolic_regression/sr_e_dist_abs_sca_theta_pos_theta_dir_phi_dir_True_logl1_1000_True/"
X, y, w, sr_summary = read_results(sel_run)
six = searchsortedfirst(sr_summary.complexity, 70)
eq_sel = sr_summary.equation[six]


@show sr_summary.eq_str[six]

@show sr_summary.eq_sym[six]

begin
    fig = Figure(size=(400*3, 400*4))
    ax = Axis(fig[1, 1], yscale=log10, xscale=log10, xlabel="Complexity", ylabel="Loss")
    lines!(ax, sr_summary.complexity, sr_summary.train_loss)
    lines!(ax, sr_summary.complexity, sr_summary.val_loss)
    scatter!(ax, sr_summary.complexity[six], sr_summary.train_loss[six], color=:red)
    ax = Axis(fig[1, 2], yscale=log10, xscale=log10, xlabel="Truth", ylabel="Prediction")
    plot_pred_target(X, y, eq_sel, ax)

    parse_plot_config(plot_configs[1], fig[2, 1], X, y, w, eq_sel)
    parse_plot_config(plot_configs[2], fig[2, 2], X, y, w, eq_sel)
    parse_plot_config(plot_configs[3], fig[2, 3], X, y, w, eq_sel)
    parse_plot_config(plot_configs[4], fig[3, 1], X, y, w, eq_sel)
    parse_plot_config(plot_configs[5], fig[3, 2], X, y, w, eq_sel)
    parse_plot_config(plot_configs[6], fig[3, 3], X, y, w, eq_sel)
    parse_plot_config(plot_configs[7], fig[4, 1], X, y, w, eq_sel)
    parse_plot_config(plot_configs[8], fig[4, 2], X, y, w, eq_sel)
    parse_plot_config(plot_configs[9], fig[4, 3], X, y, w, eq_sel)

end

fig

save("sr_demo.png", fig)

sum((X[1, :] .> 3E4) .&& (X[1, :] .< 5E4) .&& (X[2, :] .> 25) .&& (X[1, :] .< 30) .&& (X[5, :] .> 2.8) .&& (X[5, :] .< 3.0))

X[5, :]

eq_sel(permutedims([100.0, 12.5, 1.0, 1.0, 0.5, 0.12])')


vars = Symbolics.get_variables(sr_summary.eq_sym[six])

eq_rw = Rewriters.Postwalk(literaltoreal)(sr_summary.eq_sym[six])
erw = Rewriters.Postwalk(literaltoreal).(vars)
D = Differential(erw)

sr_summary.eq_sym[six]

erw


f_expr = build_function(Symbolics.toexpr(Symbolics.gradient(eq_rw, erw)), erw)
fev = eval(f_expr[1])

fev([1, 2, 3])

Symbolics.derivative(eq_rw, erw[1])

D(sr_summary.eq_sym[six])

expr = build_function(Symbolics.derivative(eq_rw, erw[1]), erw...)


expr = build_function(Symbolics.gradient(eq_rw, erw), erw...)

fev = eval(expr)

@time fev([1, 1, 1, 1, 1])


#expr = build_function(Symbolics.gradient(eq_rw, erw), erw)

buf = open("test.jl", "w")
Base.show_unquoted(buf, expr)
close(buf)

String(take!(buf))

function _sr_amplitude(energy, distance, theta_pos, theta_dir, phi_dir, abs_scale, sca_scale)
    return sqrt(energy / (((((theta_dir ^ 5.428181991340079) * 9.247915244382002) + cosh(sqrt(distance * 5.401046278233757))) / (((energy * 0.003483322443704482) ^ 0.8087638193983807) * ((abs(tan((-0.39288049458807456 - phi_dir) * 0.44508497163027766) / (theta_pos ^ (((theta_dir * 1.5318010730821618) ^ (1.1520196591834815 ^ (theta_pos * theta_pos))) - cos(-0.265860206865594 / theta_pos)))) + 0.5236238819674259) ^ (theta_dir / (0.3405908311200451 / theta_pos))))) + (abs((((energy / distance) / distance) + (cosh(sqrt((distance * 7.77668330855216) / (0.7949963740162783 ^ (theta_pos * theta_pos)))) * -0.30241315158471527)) * 0.0037544609869570376) / energy)))
end

using ForwardDiff
@time ForwardDiff.derivative(e -> _sr_amplitude(e, 1, 1, 1, 1, 1, 1), 1)