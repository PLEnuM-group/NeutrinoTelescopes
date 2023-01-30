module Plotting
using CairoMakie
using PhotonPropagation
using PhysicsTools
using ...SurrogateModels
using Flux
using ParameterSchedulers
using CUDA
using Random
using StaticArrays
using BSON: @load

export compare_mc_model

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

    #=
    oversample = 500
    @load models["4"] model hparams opt tf_dict
    samples = sample_multi_particle_event(particles, targets, model, tf_dict, c_n, rng, oversample=oversample)
    tgeo = calc_tgeo(norm(particles[1].position - targets[1].position) - targets[1].radius, c_n)
    for i in 1:16
        row, col = divrem(i - 1, 4)
        hist!(ga[col+1, row+1], samples[i] .- tgeo .- t0 , bins=-50:3:150, normalization=:density, fillaplha=0.3, weights=fill(1/oversample, length(samples[i])))
    end
    =#

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

compare_mc_model(particles, targets, models) = compare_mc_model(particles, targets, models, medium, mc_expectation(particles, targets))

end
