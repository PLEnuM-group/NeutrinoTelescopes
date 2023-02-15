using NeutrinoTelescopes
using PhysicsTools
using PhotonPropagation
using StaticArrays
using CUDA
using Flux
using Random
using BSON: @load
using LinearAlgebra
using Optim
using CairoMakie
using Distributions

models = Dict(
    "1" => joinpath(@__DIR__, "../data/full_kfold_1_FNL.bson"),
    "2" => joinpath(@__DIR__, "../data/full_kfold_2_FNL.bson"),
    "3" => joinpath(@__DIR__, "../data/full_kfold_3_FNL.bson"),
    "4" => joinpath(@__DIR__, "../data/full_kfold_4_FNL.bson"),
    "5" => joinpath(@__DIR__, "../data/full_kfold_5_FNL.bson"),
    #"FULL" => joinpath(@__DIR__, "../assets/rq_spline_model_l2_0_FULL_FNL.bson")
)
    
targets_single = [make_pone_module(@SVector[-25., 0., -450.], 1)]
targets_line = make_detector_line(@SVector[-25., 0.0, 0.0], 20, 50, 1)
targets_three_l = [
    make_detector_line(@SVector[-25., 0.0, 0.0], 20, 50, 1)
    make_detector_line(@SVector[25., 0.0, 0.0], 20, 50, 21)
    make_detector_line(@SVector[0., sqrt(50^2-25^2), 0.0], 20, 50, 41)]
targets_hex = make_hex_detector(3, 50, 20, 50, truncate=1)

detectors = Dict("Single" => targets_single, "Line" =>targets_line, "Tri" => targets_three_l, "Hex" => targets_hex)
medium = make_cascadia_medium_properties(0.99f0)

@load models["4"] model hparams tf_vec
gpu_model = gpu(model)

pos = SA[8., -5., -450]
theta = 0.5
phi = 0.4
rng = MersenneTwister(31338)
targets_range = [t for t in targets_hex if norm(t.position .- pos) < 200]

c_n = c_at_wl(800f0, medium)

samples = sample_cascade_event(5E4, theta, phi, pos, 0.; targets=targets_range, model=model, tf_vec=tf_vec, rng=rng, c_n=c_n)

pos_seed = calc_center_of_mass(targets_range, samples)

hypo = make_event_fit_model(seed_x=pos_seed[1], seed_y=pos_seed[2], seed_z=pos_seed[3], seed_time=0.)#seed_x=pos[1], seed_y=pos[2], seed_z=pos[3], seed_time=0.)
# Fix time
set_inactive!(hypo, "time")

obj_func = make_obj_func_cascade(hypo; data=samples, targets=targets_range, model=gpu_model, tf_vec=tf_vec, c_n=c_n, use_feat_buffer=false)
res = minimize_model(hypo, obj_func, strategy=:cg)

# Maximum likelihood
bestfit = Optim.minimizer(res)

fig = Figure(resolution=(1000, 1000))
les = 2:0.1:6
zens = 0:0.1:π
azis = 0:0.1:2*π


for (i, (key, targ)) in enumerate(detectors)

    targets_range = [t for t in targ if norm(t.position .- pos) < 200]
    samples = sample_cascade_event(5E4, theta, phi, pos, 0.; targets=targets_range, model=model, tf_vec=tf_vec, rng=rng, c_n=c_n)
    hypo = make_event_fit_model(seed_x=pos[1], seed_y=pos[2], seed_z=pos[3], seed_time=0.)
    set_inactive!(hypo, "time")
    obj_func = make_obj_func_cascade(hypo; data=samples, targets=targets_range, model=gpu_model, tf_vec=tf_vec, c_n=c_n, use_feat_buffer=false)

    res = minimize_model(hypo, obj_func, strategy=:cg)
    minvals = Optim.minimizer(res)


    ax1 = Axis(fig[i, 1])
    llhs = [single_cascade_likelihood(le, theta, phi, pos, 0.; data=samples, targets=targets_range, model=gpu_model, tf_vec=tf_vec, c_n=c_n) for le in les]
    lines!(ax1, les, llhs)
    vlines!(ax1, [log10(5E4), minvals[1]], color=[:red, :black])


    ax2 = Axis(fig[i, 2])
    llhs = [single_cascade_likelihood(log10(5E4), zen, phi, pos, 0.; data=samples, targets=targets_range, model=gpu_model, tf_vec=tf_vec, c_n=c_n) for zen in zens]
    lines!(ax2, zens, llhs)
    vlines!(ax2, [theta, minvals[2]], color=[:red, :black])

    ax3 = Axis(fig[i, 3])
    llhs = [single_cascade_likelihood(log10(5E4), theta, azi, pos, 0.; data=samples, targets=targets_range, model=gpu_model, tf_vec=tf_vec, c_n=c_n) for azi in azis]
    lines!(ax3, azis, llhs)
    vlines!(ax3, [phi, minvals[3]], color=[:red, :black])
end

fig

pos = SA[8., -5., -450]
theta = 0.5
phi = 0.4
rng = MersenneTwister(31338)
#targets_range = [target]

axis_ranges = Dict("Tri" => (26, 31, 15, 35, 0.1), "Single" => (0, 90, -90, 90, 1), "Line" => (20, 40, -50, 60, 0.5), "Hex" => (26, 30, 19, 28, 0.1))

levels = map(sigma -> invlogcdf(Chisq(2), log(1-2*ccdf(Normal(), sigma))), [1, 2, 3])


fig = Figure()
ga = fig[1, 1] = GridLayout()

for (i, (dkey, targets)) in enumerate(detectors)

    ranges = axis_ranges[dkey]
    thetas = deg2rad.(ranges[1]:ranges[end]:ranges[2])
    phis = deg2rad.(ranges[3]:ranges[end]:ranges[4])


    targets_range = [t for t in targets if norm(t.position .- pos) < 200]
    data = sample_cascade_event(5E4, theta, phi, pos, 0.; targets=targets_range, model=model, tf_vec=tf_vec, rng=rng, c_n=c_n)
    
    llh2d = single_cascade_likelihood.(log10(5E4), thetas, permutedims(phis), Ref(pos), 0.; data=data, targets=targets_range, model=gpu_model, tf_vec=tf_vec, c_n=c_n)
    rellh = maximum(llh2d) .- llh2d
    row, col = divrem(i-1, 2)
    ax = Axis(ga[row+1, col+1], title=dkey, xlabel="Zenith (deg)", ylabel="Azimuth (deg)", limits=ranges[1:4])
    hm = heatmap!(ax, rad2deg.(thetas), rad2deg.(phis), rellh)
    contour!(ax,  rad2deg.(thetas), rad2deg.(phis), rellh, levels=levels, color=:white)
    scatter!(ax, rad2deg(theta), rad2deg(phi))
end
#Colorbar(fig[1, 2], hm)
fig