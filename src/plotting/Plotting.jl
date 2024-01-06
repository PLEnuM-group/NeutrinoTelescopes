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
using LinearAlgebra
using ...Processing
using DataFrames


export compare_mc_model
export plot_hits_on_module


function compare_mc_model(
    particles::AbstractVector{<:Particle},
    targets::AbstractVector{<:PhotonTarget},
    models::Dict,
    medium::MediumProperties,
    hits; oversampling=1, bin_width=2)

    fig = Figure(resolution=(1500, 1000))
    fig2 = Figure()
    ax2= Axis(fig2[1, 1], xlabel="Time (ns)", ylabel="Hit density (1/ns)", yscale=Makie.pseudolog10, yminorticksvisible=true, yminorticks=IntervalsBetween(5))
    ga = fig[1, 1] = GridLayout(4, 4)


    samples = sample_multi_particle_event(particles, targets, gpu(first(models)[2]), medium; oversample=oversampling, feat_buffer=nothing)

    t_geo = calc_tgeo(particles[1], targets[1], medium)

    for i in 1:16
        row, col = divrem(i - 1, 4)
        mask = hits[:, :pmt_id] .== i
        ax = Axis(ga[col+1, row+1], xlabel="Time Residual(ns)", ylabel="Hit density (1/ns)", title="PMT $i",
        )
        hist!(ax, hits[mask, :tres], bins=-20:3:100, color=:orange, normalization=:density, weights=hits[mask, :total_weight]./oversampling)
        hist!(ax, samples[i] .- t_geo .- particles[1].time, bins=-20:3:100, color=:slateblue, normalization=:density, weights=fill(1/oversampling, length(samples[i])))
    end

    bins = -5:1:20
    hits_per_pmt = combine(groupby(hits, :pmt_id), nrow)
    max_pmt = Int64(sort(hits_per_pmt, :nrow, rev=true)[1, :pmt_id])
    mask = hits[:, :pmt_id] .== max_pmt
    hist!(ax2, hits[mask, :tres], bins=bins, normalization=:density, weights=hits[mask, :total_weight]./oversampling, label="MC Photon Propagation", color=(Makie.wong_colors()[1], 0.5))
    #hist!(ax2, samples[max_pmt] .- t_geo .- particles[1].time, bins=bins, normalization=:density, weights=fill(1/oversampling, length(samples[max_pmt])), color=(colors[2], 0.7))


    n_pmt = get_pmt_count(eltype(targets))

    t_geos = repeat([calc_tgeo(particles[1], t, medium) for t in targets], n_pmt)
    t0 = particles[1].time

    times = -5:0.05:20
 
    for (mname, model) in models
        model = gpu(model)
        shape_lhs = []
        local log_expec
        for t in times
            _, shape_lh, log_expec = SurrogateModels.evaluate_model(particles, Vector.(eachrow(t .+ t_geos .+ t0)), targets, model, medium)
            push!(shape_lhs, collect(shape_lh))
        end

        shape_lh = reduce(hcat, shape_lhs)

        for i in 1:16
            row, col = divrem(i - 1, 4)
            lines!(ga[col+1, row+1], times, exp.(shape_lh[i, :] .+ log_expec[i]), label=mname)

        end
        lines!(ax2, times, exp.(shape_lh[max_pmt, :] .+ log_expec[max_pmt]), label=mname, linewidth=3)
    end

    axislegend(ax2)
    ylims!(ax2, 0, 100)
  

    fig, fig2
end

compare_mc_model(particles, targets, models) = compare_mc_model(particles, targets, models, medium, mc_expectation(particles, targets))



function plot_hits_on_module(data, pos, dir, particles, model, target, medium)

    if eltype(particles)
        !<:AbstractArray
        particles = [particles]
    end

    fig = Figure(resolution=(1500, 1000))
    ga = fig[1, 1] = GridLayout(4, 4)

    t_geo = calc_tgeo_tracks(pos, dir, target.position, medium)
    n_pmt = get_pmt_count(eltype(targets))

    for i in 1:n_pmt
        row, col = divrem(i - 1, 4)
        ax = Axis(ga[col+1, row+1], xlabel="Time Residual(ns)", ylabel="Photons / time", title="PMT $i",
        )
        hist!(ax, data[i] .- t_geo, bins=-20:3:100, color=:orange)
    end

    times = -20:1:100
    for particles in [particles_truth, particles_unfolded]

        shape_lhs = []
        local log_expec
        for t in times
            _, shape_lh, log_expec = SurrogateModels.evaluate_model(particles, Vector.(eachrow(repeat([t + t_geo], n_pmt))), [target], gpu(model), tf_dict, c_n)
            push!(shape_lhs, collect(shape_lh))
        end

        shape_lh = reduce(hcat, shape_lhs)

        for i in 1:n_pmt
            row, col = divrem(i - 1, 4)
            lines!(ga[col+1, row+1], times, exp.(shape_lh[i, :] .+ log_expec[i]))

        end
    end

    return fig
end


end
