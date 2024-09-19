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
include("utils.jl")



function load_data()

    files = glob("*.hd5", "/home/wecapstor3/capn/capn100h/snakemake/photon_tables_perturb/extended/hits/")

    target = POM(SA_F32[0, 0, 0], 1)
    r = RotMatrix3(I)
    pmt_positions = get_pmt_positions(target, r)

    params = []

    for fname in files
        fid = h5open(fname)

        datasets = keys( fid["pmt_hits"])
        
        for ds in datasets
            hits = fid["pmt_hits/$ds"][:, :]
            attributes = attrs(fid["pmt_hits/$ds"])

            pos = attributes["distance"] * sph_to_cart(attributes["pos_theta"], attributes["pos_phi"])
            dir = sph_to_cart(attributes["dir_theta"], attributes["dir_phi"])
            particle = Particle(pos, dir, 0., attributes["energy"], 0, PEMinus)

            if length(hits) == 0
                continue
            end

            if size(hits, 2) != 3
                @show fname, ds
                continue
            end

            df = DataFrame(hits, [:time, :pmt_ix, :weight])
            df[!, :time] .+= randn(length(df.time))*2

            for pmt_ix in 1:16

                pmt_pos = pmt_positions[pmt_ix]

                rot = calc_rot_matrix(pmt_pos, [0, 0, 1])

                part_pos_rot = rot * pos
                part_dir_rot = rot * dir

                part_pos_rot_sph = cart_to_sph(part_pos_rot ./ norm(part_pos_rot))
                part_dir_rot_sph = cart_to_sph(part_dir_rot)
                delta_phi = part_pos_rot_sph[2] - part_dir_rot_sph[2]

                sel = df[df.pmt_ix .== pmt_ix, :]

                push!(
                        params,
                        (
                            nhits = sum(sel.weight, init=0),
                            variance = sum(sel.weight .^2, init=0),
                            pmt_ix=pmt_ix,
                            pos=pos,
                            dir=dir,
                            tshift=NaN,
                            gfit_alpha=NaN,
                            gfit_theta=NaN,
                            energy=particle.energy,
                            delta_phi=delta_phi,
                            theta_pos=part_pos_rot_sph[1],
                            theta_dir=part_dir_rot_sph[1],
                            abs_scale=attributes["abs_scale"],
                            sca_scale=attributes["sca_scale"],
                            distance=attributes["distance"],
                        )
                    )
                    

                #=
                if nrow(sel) < 10
                    push!(
                        params,
                        (
                            nhits = sum(sel.weight, init=0),
                            variance = sum(sel.weight .^2, init=0),
                            pmt_ix=pmt_ix,
                            pos=pos,
                            dir=dir,
                            tshift=NaN,
                            gfit_alpha=NaN,
                            gfit_theta=NaN,
                            energy=particle.energy,
                            delta_phi=delta_phi,
                            theta_pos=part_pos_rot_sph[1],
                            theta_dir=part_dir_rot_sph[1],
                            abs_scale=attributes["abs_scale"],
                            sca_scale=attributes["sca_scale"],
                            distance=attributes["distance"],))
                    continue
                end

                
                tshift = minimum(sel[:, :time])
                #@show sel.time .- tshift .+ 1E-10
                gfit = nothing
                gfit = fit_mle(Gamma, sel.time .- tshift .+ 1E-10, sel.weight)
            

                push!(
                    params,
                    (
                        nhits = sum(sel.weight, init=0),
                        variance = sum(sel.weight .^2, init=0),
                        pmt_ix=pmt_ix,
                        pos=pos,
                        dir=dir,
                        tshift=tshift,
                        gfit_alpha=gfit.α,
                        gfit_theta=gfit.θ,
                        energy=particle.energy,
                        delta_phi=delta_phi,
                        theta_pos=part_pos_rot_sph[1],
                        theta_dir=part_dir_rot_sph[1],
                        abs_scale=attributes["abs_scale"],
                        sca_scale=attributes["sca_scale"],
                        distance=attributes["distance"],
                        )
                )
            =#
            end
        end
    end
    params = DataFrame(params)
    return params
end


function weighted_subsample(df, n, weighting_func)

    weights = weighting_func(df)
    ixs = 1:nrow(df)

    selected_ixs = StatsBase.sample(ixs, Weights(weights), n, replace=false)

    return df[selected_ixs, :]
    
end

function weight_func(df) 

    cnt_1 = sum(df.energy .< 1E5)
    cnt_2 = sum(df.energy .>= 1E5)

    weight = zeros(nrow(df))

    weight[df.energy .< 1E5] .= (log10(1E5)-log10(1E2)) / cnt_1
    weight[df.energy .>= 1E5] .= (log10(5E6)-log10(1E5)) / cnt_2

    weight .*= df.energy .* df.distance

    return weight
end

function plot_result(func, X, y)
    
    log_ebins =1:0.5:8
    ebins = 10 .^(log_ebins)

    xs = 10 .^ (0:0.01:log10(200))

    cmap = cgrad(:viridis)

    get_color_e(val) = cmap[(val-log10(minimum(ebins))) / (log10(maximum(ebins)) - log10(minimum(ebins)))]

    fig = Figure(size=(1000, 1000))
    ax = Axis(fig[1, 1], yscale=log10, xscale=log10, xlabel="Distance", ylabel="Number of Hits")
    scatter!(ax, X[2, :], y, color=get_color_e.(log10.(X[1, :])))

    bin_centers = 10 .^ (0.5 * (log_ebins[1:end-1] + log_ebins[2:end]))

    xeval = zeros(3, length(xs))

    for ebin in bin_centers
        xeval[1, :] .= ebin
        xeval[2, :] .= xs
        xeval[3, :] .= 1
        #ys, _ = eval_tree_array(eq_sel, xeval, opt)
        
        ys = func(xeval)
        
        lines!(ax, xs, ys, color=get_color_e(log10(ebin)))
    end
    ylims!(ax, 1E-2, 1E5)
    
    ax2 = Axis(fig[1, 2], yscale=log10, xscale=log10, xlabel="Distance", ylabel="Number of Hits")
    xeval[1, :] .= 1E4
    ys = func(xeval)
    sel = (X[1, :] .> 9E3) .&& (X[1, :] .< 2E4)
    scatter!(ax2, X[2, sel], y[sel], color=get_color_e.(log10.(X[1, sel])))        
    lines!(ax2, xs, ys, color=get_color_e(log10(1E4)))

    xeval[1, :] .= 1E3
    ys = func(xeval)
    sel = (X[1, :] .> 9E2) .&& (X[1, :] .< 2E3)
    scatter!(ax2, X[2, sel], y[sel], color=get_color_e.(log10.(X[1, sel])))        
    lines!(ax2, xs, ys, color=get_color_e(log10(1E3)))

    ylims!(ax2, 1E-2, 1E5, )


    ax3 = Axis(fig[2, 1], yscale=log10, xscale=log10, xlabel="Distance", ylabel="Number of Hits")
    get_color_a(val) = cmap[(val-0.9) / (1.1 - 0.9)]

    xeval[1, :] .= 1E3
    xeval[3, :] .= 0.9
    ys = func(xeval)
    sel = (X[1, :] .> 9E2) .&& (X[1, :] .< 2E3)
    scatter!(ax3, X[2, sel], y[sel], color=get_color_a.(X[3, sel]))        
    lines!(ax3, xs, ys, color=get_color_a(0.9))

    xeval[3, :] .= 1.0
    ys = func(xeval)
    lines!(ax3, xs, ys, color=get_color_a(1.0))

    xeval[3, :] .= 1.1
    ys = func(xeval)
    lines!(ax3, xs, ys, color=get_color_a(1.0))

    ylims!(ax3, 1E-2, 1E5, )
    fig

end




#=data = load_data()
jldsave("/home/wecapstor3/capn/capn100h/sr_dataset_full.jld2", data=data)
=#

data = load("/home/wecapstor3/capn/capn100h/sr_dataset_full.jld2")["data"]

config = TOML.parsefile("/home/saturn/capn/capn100h/julia_dev/NeutrinoTelescopes/scripts/surrogate_models/symbolic_regression/sr_run.toml")
data = JLD2.load(config["input"]["filename"])["data"]

dsel = apply_selection(data, config["selection"])
X = Matrix(dsel[:, Symbol.(config["run"]["variables"])])'
y = dsel.nhits
y_resampled = pois_rand.(y)

#jldsave("/home/wecapstor3/capn/capn100h/sr_dataset_e_dist.jld2", X=X, y=y_resampled)


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

func(energy, distance, abs_scale) =(energy / (square(square(distance * 0.21846)) + 55.481)) ^ 0.89596
# ((energy ^ 0.86009) / (square(2.6646 - distance) + 45.157)) * square(abs_scale)
#y = (energy / (square(5.0384 - distance) + 41.002)) ^ 0.87882

(energy / (square(distance) + 85.966)) ^ 0.94623

func(X) = func.(X[1, :], X[2, :], X[3, :])
plot_result(func, X, y)


h2d = Hist2D((y, yeval), binedges=(10 .^ (-2:0.1:6), 10 .^ (-2:0.1:6)))
h2d.bincounts ./= sum(h2d.bincounts, dims=2)

fig = Figure()
ax = Axis(fig[1, 1], yscale=log10, xscale=log10, xlabel="Truth", ylabel="Prediction")
plot!(ax, h2d)

xs = 10 .^ (-2:0.1:5)
lines!(ax, xs, xs, color=:black)
fig












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


fig


