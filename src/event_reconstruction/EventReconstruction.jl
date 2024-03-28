module EventReconstruction

using PhotonPropagation
using PhysicsTools
using Rotations
using LinearAlgebra
using ...Processing

include("reco_model.jl")
include("geometric_features.jl")

export calc_resolution_maxlh


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
