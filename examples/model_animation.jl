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




function compare_mc_model(
    particles::AbstractVector{<:Particle},
    targets::AbstractVector{<:PhotonTarget},
    models::Dict,
    medium::MediumProperties,
    hits)

    c_n = c_at_wl(800.0f0, medium)

    fig = Figure(resolution=(1500, 1000))
    ga = fig[1, 1] = GridLayout(4, 4)

    for i in 1:16
        row, col = divrem(i - 1, 4)
        mask = hits[:, :pmt_id] .== i
        ax = Axis(ga[col+1, row+1], xlabel="Time Residual(ns)", ylabel="Photons / time", title="PMT $i",
        )
        hist!(ax, hits[mask, :tres], bins=-50:3:150, weights=hits[mask, :total_weight], color=:orange, normalization=:density,)
    end

    n_pmt = get_pmt_count(eltype(targets))

    t_geos = repeat([calc_tgeo(norm(particles[1].position - t.position) - t.radius, c_n) for t in targets], n_pmt)
    t0 = particles[1].time

    times = -50:1:150
    for (mname, model_path) in models
        @load model_path model hparams opt tf_vec
        input = calc_flow_input(particles, targets, tf_vec)


        shape_lhs = []
        local log_expec
        for t in times
            _, shape_lh, log_expec = SurrogateModels.evaluate_model(particles, Vector.(eachrow(t .+ t_geos .+ t0)), targets, gpu(model), tf_vec, c_n)
            push!(shape_lhs, collect(shape_lh))
        end

        shape_lh = reduce(hcat, shape_lhs)

        for i in 1:16
            row, col = divrem(i - 1, 4)
            lines!(ga[col+1, row+1], times, exp.(shape_lh[i, :] .+ log_expec[i]), label=mname)

        end
    end

    fig
end



# Setup
models = Dict(
    "1" => joinpath(@__DIR__, "../assets/full_kfold_1_FNL.bson"),
    "2" => joinpath(@__DIR__, "../assets/full_kfold_2_FNL.bson"),
    "3" => joinpath(@__DIR__, "../assets/full_kfold_3_FNL.bson"),
    "4" => joinpath(@__DIR__, "../assets/full_kfold_4_FNL.bson"),
    "5" => joinpath(@__DIR__, "../assets/full_kfold_5_FNL.bson"),
    #"FULL" => joinpath(@__DIR__, "../assets/rq_spline_model_l2_0_FULL_FNL.bson")
)

targets_single = [make_pone_module(@SVector[-25.0, 0.0, -450.0], 1)]

medium = make_cascadia_medium_properties(0.99f0)

# Check model performance
pos = SA[-25.0, 5.0, -460]
dir_theta = 0.7
dir_phi = 1.3
dir = sph_to_cart(dir_theta, dir_phi)


particles = [
    Particle(pos, dir, 0., 5E4, 0.0, PEMinus),
]
#hits = mc_expectation(particles, targets_single, 1);

n_pmt = get_pmt_count(eltype(targets_single))
c_n = c_at_wl(800.0f0, medium)
t_geos = repeat([calc_tgeo(norm(particles[1].position - t.position) - t.radius, c_n) for t in targets_single], n_pmt)
t0 = particles[1].time

times = -50:1:150
@load models["2"] model hparams opt tf_vec
input = calc_flow_input(particles, targets, tf_vec)


shape_lhs = []
log_expec = 0
for t in times
    _, shape_lh, log_expec = SurrogateModels.evaluate_model(particles, Vector.(eachrow(t .+ t_geos .+ t0)), targets, gpu(model), tf_vec, c_n)
    push!(shape_lhs, collect(shape_lh))
end

shape_lh = reduce(hcat, shape_lhs)

for i in 1:16
    row, col = divrem(i - 1, 4)
    lines!(ga[col+1, row+1], times, exp.(shape_lh[i, :] .+ log_expec[i]), label=mname)

end






compare_mc_model(particles, targets_single, models, medium, hits)