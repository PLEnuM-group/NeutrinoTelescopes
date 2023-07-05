using NeutrinoTelescopes
using PhotonPropagation
using PhysicsTools
using Flux
using CUDA
using StaticArrays
using Cthulhu
using Random
using ForwardDiff
using PoissonRandom
using Base.Iterators
using LogExpFunctions

targets_single = [POM(@SVector[-25.0, 0.0, -450.0], 1)]
targets_line = make_detector_line(@SVector[-25.0, 0.0, 0.0], 20, 50, 1)
targets_three_l = [
    make_detector_line(@SVector[-25.0, 0.0, 0.0], 20, 50, 1)
    make_detector_line(@SVector[25.0, 0.0, 0.0], 20, 50, 21)
    make_detector_line(@SVector[0.0, sqrt(50^2 - 25^2), 0.0], 20, 50, 41)]
targets_hex = make_hex_detector(3, 50, 20, 50, truncate=1)

detectors = Dict("Single" => targets_single, "Line" =>targets_line, "Tri" => targets_three_l, "Hex" => targets_hex)
medium = make_cascadia_medium_properties(0.95f0)

targets = targets_single
model_path = joinpath(ENV["WORK"], "time_surrogate")
model = PhotonSurrogate(joinpath(model_path, "extended/amplitude_1_FNL.bson"), joinpath(model_path, "extended/time_1_FNL.bson"))
model = gpu(model)

c_n = c_at_wl(800.0f0, medium)
rng = MersenneTwister(31338)

pos = SA[-25.0, 5.0, -460]
dir_theta = 0.1
dir_phi = 0.3
dir = sph_to_cart(dir_theta, dir_phi)

typeof(pos)

log_energy = 4.

p = Particle(pos, dir, 0.0, 10^log_energy, 0.0, PEMinus)

rng = MersenneTwister(31338)
samples = sample_cascade_event(10^log_energy, dir_theta, dir_phi, pos, 0.; targets=targets, model=model, rng=rng, c_n=c_n)


function make_lh_func(;pos, time, data, targets, model, c_n)

    function evaluate_lh(log_energy::Real, dir_theta::Real, dir_phi::Real)
        return single_cascade_likelihood(log_energy, dir_theta, dir_phi, pos, time, data=data, targets=targets, model=model, c_n=c_n)
    end

    wrapped(pars::Vector{<:Real}) = evaluate_lh(pars...)

    return evaluate_lh, wrapped
end

f, fwrapped = make_lh_func(pos=pos, time=0., data=samples, targets=targets, model=model, c_n=c_n)

@descend fwrapped([log_energy, dir_theta, dir_phi])

