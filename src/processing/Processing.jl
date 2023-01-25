module Processing

using DataFrames
using PhysicalConstants.CODATA2018
using PhotonPropagation
using PhysicsTools
using Unitful
using LinearAlgebra


const c_vac_m_ns = ustrip(u"m/ns", SpeedOfLightInVacuum)

export calc_time_residual!, calc_tgeo, c_at_wl
export calc_time_residual_tracks!, calc_tgeo_tracks

calc_tgeo(distance, c_n::Number) = distance / c_n
calc_tgeo(distance, medium::MediumProperties) = calc_tgeo(distance, c_at_wl(800.0, medium))

function calc_tgeo(particle::Particle, target::PhotonTarget, c_n_or_medium)
    return calc_tgeo(norm(particle.position .- target.position) - target.radius, c_n_or_medium)
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
            target.position,
            setup.medium)

        subdf[!, :tres] = (subdf[:, :time] .- tgeo .- t0)
    end
end


function calc_time_residual_cascades!(df::AbstractDataFrame, setup::PhotonPropSetup)

    targ_id_map = Dict([target.module_id => target for target in setup.targets])

    t0 = setup.sources[1].time
    for (key, subdf) in pairs(groupby(df, :module_id))
        target = targ_id_map[key.module_id]
        distance = norm(setup.sources[1].position .- target.position)
        tgeo = calc_tgeo((distance - target.radius), setup.medium)

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




end
