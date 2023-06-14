module Processing

using DataFrames
using PhysicalConstants.CODATA2018
using PhotonPropagation
using PhysicsTools
using Unitful
using LinearAlgebra
using StatsBase
using PoissonRandom
using Reexport

include("triggering.jl")
include("io.jl")
@reexport using .Triggering
@reexport using .IO

const c_vac_m_ns = ustrip(u"m/ns", SpeedOfLightInVacuum)

export calc_time_residual!, calc_tgeo, c_at_wl
export calc_time_residual_tracks!, calc_tgeo_tracks


calc_tgeo(distance, c_n::Number) = distance / c_n
calc_tgeo(distance, medium::MediumProperties) = calc_tgeo(distance, c_at_wl(800.0, medium))

function calc_tgeo(distance::Real, target::PhotonTarget{<:Spherical}, c_n_or_medium)
    return  calc_tgeo(distance - target.shape.radius, c_n_or_medium)
end

function calc_tgeo(particle::Particle, target, c_n_or_medium)
    return calc_tgeo(norm(particle.position .- target.shape.position), target, c_n_or_medium)
end


function closest_approach_distance(p0, dir, pos)
    return norm(cross((pos .- p0), dir))
end


function calc_tgeo_tracks(p0, dir, pos, n_ph, n_grp)

    dist = closest_approach_distance(p0, dir, pos)
    dpos = pos .- p0
    t_geo = 1 / c_vac_m_ns * (dot(dir, dpos) + dist * (n_grp * n_ph - 1) / sqrt((n_ph^2 - 1)))
    return t_geo
end

function calc_tgeo_tracks(p0, dir, pos, medium::MediumProperties)

    wl = 800.0
    n_ph = refractive_index(wl, medium)
    n_grp = c_vac_m_ns / group_velocity(wl, medium)

    return calc_tgeo_tracks(p0, dir, pos, n_ph, n_grp)
end

function calc_time_residual_tracks!(df::AbstractDataFrame, setup::PhotonPropSetup)

    targ_id_map = Dict([target.module_id => target for target in setup.targets])

    t0 = setup.sources[1].time
    for (key, subdf) in pairs(groupby(df, :module_id))
        target = targ_id_map[key.module_id]
        tgeo = calc_tgeo_tracks(
            setup.sources[1].position,
            setup.sources[1].direction,
            target.shape.position,
            setup.medium)

        subdf[!, :tres] = (subdf[:, :time] .- tgeo .- t0)
    end
end


function calc_time_residual_cascades!(df::AbstractDataFrame, setup::PhotonPropSetup)

    targ_id_map = Dict([target.module_id => target for target in setup.targets])

    t0 = setup.sources[1].time
    for (key, subdf) in pairs(groupby(df, :module_id))
        target = targ_id_map[key.module_id]
        distance = norm(setup.sources[1].position .- target.shape.position)
        tgeo = calc_tgeo(distance, target, setup.medium)

        subdf[!, :tres] = (subdf[:, :time] .- tgeo .- t0)
    end
end


function calc_time_residual!(df::AbstractDataFrame, setup::PhotonPropSetup)
    if eltype(setup.sources) <: CherenkovTrackEmitter
        return calc_time_residual_tracks!(df, setup)
    else
        return calc_time_residual_cascades!(df, setup)
    end

end

#=
function resample_simulation(hit_times, total_weights, downsample=1.0)
    wsum = sum(total_weights)

    mask = total_weights .> 0
    hit_times = hit_times[mask]
    total_weights = total_weights[mask]

    norm_weights = ProbabilityWeights(copy(total_weights), wsum)
    nhits = min(pois_rand(wsum * downsample), length(hit_times))
    try
        sample(hit_times, norm_weights, nhits; replace=false)
    catch e
        @show length(hit_times)
        error("error")
    end
end


function resample_simulation(df::AbstractDataFrame; downsample=1.0, per_pmt=true, time_col=:time)

    wrapped(hit_times, total_weights) = resample_simulation(hit_times, total_weights, downsample)

    if per_pmt
        groups = groupby(df, [:pmt_id, :module_id])
    else
        groups = groupby(df, :module_id)
    end
    resampled_hits = combine(groups, [time_col, :total_weight] => wrapped => time_col)
    return resampled_hits
end
=#


end
