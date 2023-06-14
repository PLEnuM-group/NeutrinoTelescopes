module EventReconstruction

using PhotonPropagation
using PhysicsTools
using Rotations
using LinearAlgebra
using ...Processing

include("reco_model.jl")
include("geometric_features.jl")

export mc_expectation, calc_resolution_maxlh

function mc_expectation(particles::AbstractVector{<:Particle}, targets::AbstractVector{<:PhotonTarget}, seed, medium::MediumProperties)

    wl_range = (300.0f0, 800.0f0)
    spectrum = CherenkovSpectrum(wl_range, medium)

    sources = [particle_shape(p) isa Cascade ?
               ExtendedCherenkovEmitter(convert(Particle{Float32}, p), medium, wl_range) :
               CherenkovTrackEmitter(convert(Particle{Float32}, p), medium, wl_range)
               for p in particles]

    targets_c::Vector{POM{Float32}} = convert(Vector{POM{Float32}}, targets)

    photon_setup = PhotonPropSetup(sources, targets_c, medium, spectrum, seed)
    photons = propagate_photons(photon_setup)

    calc_total_weight!(photons, photon_setup)
    calc_time_residual!(photons, photon_setup)

    rot = RotMatrix3(I)
    hits = make_hits_from_photons(photons, photon_setup, rot)
    return hits
end


function calc_resolution_maxlh(targets, sampling_model, eval_model, n; energy=1E4, zenith=0.1, phi=0.1, position=SA[3.0, 10.0, 15.0], time=0.0)

    rng = MersenneTwister(31338)
    hypo = make_cascade_fit_model(seed_x=position[1], seed_y=position[2], seed_z=position[3], seed_time=time)
    set_inactive!(hypo, "pos_x")
    set_inactive!(hypo, "pos_y")
    set_inactive!(hypo, "pos_z")
    set_inactive!(hypo, "time")
    min_vals = []
    for _ in 1:n
        samples = sample_cascade_event(energy, zenith, phi, position, time; targets=targets, model=sampling_model[:model], tf_vec=sampling_model[:tf_dict], rng=rng)
        res = min_lh(hypo, samples, targets, eval_model[:model], eval_model[:tf_dict])
        push!(min_vals, Optim.minimizer(res))
    end
    min_vals = reduce(hcat, min_vals)

    return min_vals
end

calc_resolution_maxlh(targets, model, n) = calc_resolution_maxlh(targets, model, model, n)





end
