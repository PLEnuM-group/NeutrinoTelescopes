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
targets_line = make_detector_line(@SVector[-25.0, 0.0, 0.0], 20, 50, 1)
targets_three_l = [
    make_detector_line(@SVector[-25.0, 0.0, 0.0], 20, 50, 1)
    make_detector_line(@SVector[25.0, 0.0, 0.0], 20, 50, 21)
    make_detector_line(@SVector[0.0, sqrt(50^2 - 25^2), 0.0], 20, 50, 41)]
targets_hex = make_hex_detector(3, 50, 20, 50, truncate=1)

detectors = Dict("Single" => targets_single, "Line" => targets_line, "Tri" => targets_three_l, "Hex" => targets_hex)
medium = make_cascadia_medium_properties(0.99f0)

# Check model performance
pos = SA[-25.0, 5.0, -460]
dir_theta = 0.7
dir_phi = 1.3
dir = sph_to_cart(dir_theta, dir_phi)


@load models["4"] model hparams tf_vec
particles = [
    Particle(pos, dir, 0, 1E5, 0.0, PEMinus)]

log10_ampl = (get_log_amplitudes(particles, targets_hex, gpu(model), tf_vec; feat_buffer=nothing)[1][:] .+ log(1E4)) ./ log(10)

hist(log10_ampl,  
    axis=(; xlabel="Log10(Number of Photons / PMT)",  title="1E9 GeV EM Cascade "))




particles = [
    Particle(pos, dir, -20.0, 5.0, 0.0, PEMinus),
    Particle(pos, dir, -10.0, 50.0, 0.0, PEMinus),
    Particle(pos, dir, 0.0, 500.0, 0.0, PEMinus),
    Particle(pos, dir, 10.0, 5000.0, 0.0, PEMinus),
    Particle(pos, dir, 20.0, 5E4, 0.0, PEMinus),
    #Particle(pos, dir, 30., 5E5, 0., PEMinus),
]
hits = mc_expectation(particles, targets_single, 1);

compare_mc_model(particles, targets_single, models, medium, hits)


energy = 3E4
rng = MersenneTwister(31338)
particles = [
    Particle(SA[-23.0, -1.0, -450], dir, -10.0, 5.0, 0.0, PEMinus),
    Particle(pos, dir, 0.0, energy, 0.0, PEMinus),
    Particle(pos .+ dir .* 5, dir, 15, energy, 0.0, PEMinus),
    Particle(pos .+ dir .* 10, dir, 25, energy, 0.0, PEMinus)
]

hits = mc_expectation(particles, targets_single, 1);
compare_mc_model(particles, targets_single, models, medium, hits)
