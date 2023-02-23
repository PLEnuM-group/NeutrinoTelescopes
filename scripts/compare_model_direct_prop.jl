using PhotonPropagation
using NeutrinoTelescopes
using PhysicsTools
using StaticArrays
using LinearAlgebra
using Random
using Flux
using BSON: @load
using HDF5
using DataFrames


models_casc = Dict(
    "1" => joinpath(@__DIR__, "../data/full_kfold_1_FNL.bson"),
    "2" => joinpath(@__DIR__, "../data/full_kfold_2_FNL.bson"),
    "3" => joinpath(@__DIR__, "../data/full_kfold_3_FNL.bson"),
    "4" => joinpath(@__DIR__, "../data/full_kfold_4_FNL.bson"),
    "5" => joinpath(@__DIR__, "../data/full_kfold_5_FNL.bson"),
)

# Tracks are simulated only at 100TeV!!!
models_track = Dict(
    "1" => joinpath(@__DIR__, "../data/infinite_bare_muon_1_FNL.bson"),
    "2" => joinpath(@__DIR__, "../data/infinite_bare_muon_2_FNL.bson"),
    "3" => joinpath(@__DIR__, "../data/infinite_bare_muon_3_FNL.bson"),
    "4" => joinpath(@__DIR__, "../data/infinite_bare_muon_4_FNL.bson"),
    "5" => joinpath(@__DIR__, "../data/infinite_bare_muon_5_FNL.bson"),
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

pos = SA[-25.0, 0, -460]
dir_theta = 1.5
dir_phi = 1.3
dir = sph_to_cart(dir_theta, dir_phi)
energy = 3e4

rng = MersenneTwister(31338)
particles = [
    Particle(pos, dir, 0.0, energy, 0.0, PEMinus),
]

hits = mc_expectation(particles, targets_single, 1);
compare_mc_model(particles, targets_single, models_casc, medium, hits)


particles_track = [
    Particle(pos .- 50 .* dir, dir, -50 / 0.3, 1E5, 100, PMuPlus)
]

propagate_muon(particles_track[1])


hits_track = mc_expectation(particles_track, targets_single, 1);
compare_mc_model(particles_track, targets_single, models_track, medium, hits_track)

f = h5open(joinpath(@__DIR__, "../data/photon_table_bare_infinite_0.hd5"))
grp = f["pmt_hits/dataset_101"]
hits = DataFrame(grp[:, :], [:tres, :pmt_id])

attrs(grp)

minimum(hits[:, :tres])

rel_pos = attrs(grp)["distance"] * sph_to_cart(attrs(grp)["pos_theta"], attrs(grp)["pos_phi"])

pos = targets_single[1].position + rel_pos
dir = sph_to_cart(acos(attrs(grp)["dir_costheta"]), attrs(grp)["dir_phi"])

particles_track = [
    Particle(pos .- 50 .* dir, dir, -50 / 0.3, 1E4, 100, PMuPlus)
]
hits_track = mc_expectation(particles_track, targets_single, 1);
compare_mc_model(particles_track, targets_single, models_track, medium, hits_track)
