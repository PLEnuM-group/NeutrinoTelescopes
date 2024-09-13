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
import SymbolicUtils
import SymbolicRegression


function load_data()


    fname = "/home/wecapstor3/capn/capn100h/snakemake/photon_tables_perturb/hadronic/hits/photon_table_hits_hadronic_dmin_1_dmax_200_emin_100_emax_100000.0_9.hd5"


    fid = h5open(fname)

    datasets = keys( fid["pmt_hits"])
    target = POM(SA_F32[0, 0, 0], 1)
    r = RotMatrix3(I)
    pmt_positions = get_pmt_positions(target, r)

    params = []
    for ds in datasets
        hits = fid["pmt_hits/$ds"][:, :]
        attributes = attrs(fid["pmt_hits/$ds"])

        pos = attributes["distance"] * sph_to_cart(attributes["pos_theta"], attributes["pos_phi"])
        dir = sph_to_cart(attributes["dir_theta"], attributes["dir_phi"])
        particle = Particle(pos, dir, 0., attributes["energy"], 0, PHadronShower)
        
        medium = CascadiaMediumProperties(0.95, attributes["abs_scale"], attributes["sca_scale"])

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
            if nrow(sel) < 10
                push!(
                    params,
                    (
                        nhits = sum(sel.weight, init=0),
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
                    distance=attributes["distance"],
                    )
            )
        end
    end
    params = DataFrame(params)
    return params
end

params = load_data()

X = hcat(
    params.energy,
    params.distance,
    params.theta_pos,
    params.theta_dir,
    params.delta_phi
)'
y = params.nhits


jldsave("/home/wecapstor3/capn/capn100h/sr_dataset.jld2", X=X, y=y)


dominating = calculate_pareto_frontier(hof)

println("Complexity\tMSE\tEquation")

for member in dominating
    complexity = compute_complexity(member, options)
    loss = member.loss
    string = string_tree(member.tree, options)

    println("$(complexity)\t$(loss)\t$(string)")
end

save("symbolic_regression.jld2", hof=hof)



func(energy, distance, delta_phi, theta_dir, theta_pos) = ((abs(((energy / (1.1551 ^ (distance - -3.4184))) * (tan(-0.5616 + cos(delta_phi)) * theta_dir)) - delta_phi) / (exp(theta_dir + (0.65039 + -0.96265)) ^ (((theta_pos * theta_pos) * theta_dir) + -2.9214))) ^ 0.57105) * theta_pos
yeval = func.(Xtest.energy, Xtest.distance, Xtest.delta_phi, Xtest.theta_dir, Xtest.theta_pos)
ytest



h2d = Hist2D((ytest, yeval), binedges=(10 .^ (-2:0.2:5), 10 .^ (-2:0.2:5)))
h2d.bincounts ./= sum(h2d.bincounts, dims=2)

fig = Figure()
ax = Axis(fig[1, 1], yscale=log10, xscale=log10, xlabel="Truth", ylabel="Prediction")
plot!(ax, h2d)

xs = 10 .^ (-2:0.2:5)
lines!(ax, xs, xs)
ig


sel = (Xtest.distance .> 20) .& (Xtest.distance .< 30) .& (Xtest.delta_phi .> 0) .& (Xtest.delta_phi .< 0.1)

Xtest_sel = (energy=Xtest.energy[sel], distance=Xtest.distance[sel], delta_phi=Xtest.delta_phi[sel])


yeval = func.(Xtest_sel.energy, Xtest_sel.distance, Xtest_sel.delta_phi)
ytest_sel = ytest[sel]

scatter(yeval, ytest_sel)



h2d = Hist2D((log10.(Xtest.energy), (ytest-yeval) ./ yeval), binedges = (2:0.2:5, -5:0.1:1))

plot(h2d)


fig, ax, s = hist((log10.(Xtest.energy), (ytest./yeval)))

ylims!(ax, -0.5, 2)
fig


#=
fig, ax, s = scatter(ytest, predict(mach, (data=Xtest, idx=10)))
xlims!(-10, 10)
ylims!(-10, 10)
fig


sel = (Xtest.distance .> 20) .& (Xtest.distance .< 30)

import SymbolicRegression: compute_complexity, string_tree

println("Complexity\tMSE\tEquation")

for member in dominating
    complexity = compute_complexity(member, options)
    loss = member.loss
    string = string_tree(member.tree, options)

    println("$(complexity)\t$(loss)\t$(string)")
end
=#