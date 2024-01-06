using NeutrinoTelescopes
using Flux
using ParameterSchedulers
using CUDA
using Random
using StaticArrays
using BSON: @save, @load
using BSON
using CairoMakie
using DataFrames
using PhysicsTools
using PhotonPropagation
using LinearAlgebra
using Formatting

# Setup
models = Dict(
    "1" => joinpath(@__DIR__, "../data/full_kfold_1_FNL.bson"),
    "2" => joinpath(@__DIR__, "../data/full_kfold_2_FNL.bson"),
    "3" => joinpath(@__DIR__, "../data/full_kfold_3_FNL.bson"),
    "4" => joinpath(@__DIR__, "../data/full_kfold_4_FNL.bson"),
    "5" => joinpath(@__DIR__, "../data/full_kfold_5_FNL.bson"),
    #"FULL" => joinpath(@__DIR__, "../assets/rq_spline_model_l2_0_FULL_FNL.bson")
)

targets_single = [make_pone_module(@SVector[-25.0, 0.0, -450.0], 1)]
medium = make_cascadia_medium_properties(0.99f0)
n_pmt = get_pmt_count(eltype(targets_single))
c_n = c_at_wl(800.0f0, medium)

times = -50:1:150
@load models["2"] model hparams tf_vec

fig = Figure(resolution=(1500, 1000))
title = Observable("")
ga = fig[1, 1] = GridLayout(4, 4)
lab = Label(ga[1, :, Top()], title, padding=(0, 0, 5, 0))

points = []
for i in 1:16
    row, col = divrem(i - 1, 4)
    ax = Axis(ga[col+1, row+1], xlabel="Time Residual(ns)", ylabel="Photons / time", title="PMT $i",
        limits=(-60, 160, 1E-1, 1E2), yscale=log10)
    p = Observable(Point2f[])
    lines!(ax, p)
    push!(points, p)
end


fname = joinpath(@__DIR__, "../data/model_anim.mkv")

CairoMakie.record(fig, fname, 0.001:0.05:π) do zen

    # Check model performance
    pos = SA[-25.0, 5.0, -460]
    dir_theta = zen
    dir_phi = 0.3
    dir = sph_to_cart(dir_theta, dir_phi)

    particles = [
        Particle(pos, dir, 0.0, 5E4, 0.0, PEMinus),
    ]
    t_geos = repeat([calc_tgeo(norm(particles[1].position - t.position) - t.radius, c_n) for t in targets_single], n_pmt)
    t0 = particles[1].time
    input = calc_flow_input(particles, targets_single, tf_vec)

    shape_lhs = []
    log_expec = 0
    for t in times
        _, shape_lh, log_expec = SurrogateModels.evaluate_model(particles, Vector.(eachrow(t .+ t_geos .+ t0)), targets_single, gpu(model), tf_vec, c_n)
        push!(shape_lhs, collect(shape_lh))
    end
    shape_lh = reduce(hcat, shape_lhs)

    for i in 1:16
        points[i][] = Point2f.(times, exp.(shape_lh[i, :] .+ log_expec[i]))
    end
    title[] = format("Zenith: {:.2f}", rad2deg(zen))
    notify.(points)
    notify(title)
end




hits = mc_expectation(particles, targets_single, 1);
compare_mc_model(particles, targets_single, models, medium, hits)
