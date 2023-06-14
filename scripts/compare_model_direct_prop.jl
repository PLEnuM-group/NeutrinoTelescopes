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


model_path = joinpath(ENV["WORK"], "time_surrogate")

models_casc = Dict(
    "1" => joinpath(model_path, "extended/extended_casc_1_FNL.bson"),
    "2" => joinpath(model_path, "extended/extended_casc_2_FNL.bson"),
    "3" => joinpath(model_path, "extended/extended_casc_3_FNL.bson"),
    "4" => joinpath(model_path, "extended/extended_casc_4_FNL.bson"),
    "5" => joinpath(model_path, "extended/extended_casc_5_FNL.bson"),
)

# Tracks are simulated only at 100TeV!!!
models_track = Dict(
    "1" => joinpath(model_path, "infinite_track/infinite_bare_muon_1_FNL.bson"),
    "2" => joinpath(model_path, "infinite_track/infinite_bare_muon_2_FNL.bson"),
    "3" => joinpath(model_path, "infinite_track/infinite_bare_muon_3_FNL.bson"),
    "4" => joinpath(model_path, "infinite_track/infinite_bare_muon_4_FNL.bson"),
    "5" => joinpath(model_path, "infinite_track/infinite_bare_muon_5_FNL.bson"),
)

targets_single = [POM(@SVector[-25.0, 0.0, -450.0], 1)]
targets_line = make_detector_line(@SVector[-25.0, 0.0, 0.0], 20, 50, 1)
targets_three_l = [
    make_detector_line(@SVector[-25.0, 0.0, 0.0], 20, 50, 1)
    make_detector_line(@SVector[25.0, 0.0, 0.0], 20, 50, 21)
    make_detector_line(@SVector[0.0, sqrt(50^2 - 25^2), 0.0], 20, 50, 41)]
targets_hex = make_hex_detector(3, 50, 20, 50, truncate=1)

detectors = Dict("Single" => targets_single, "Line" => targets_line, "Tri" => targets_three_l, "Hex" => targets_hex)
medium = make_cascadia_medium_properties(0.95f0)

pos = SA[-15.0, 10, -460]
dir_theta = 0.6
dir_phi = 0.1
dir = sph_to_cart(dir_theta, dir_phi)
energy = 3e4

rng = MersenneTwister(31338)
particles = [
    Particle(pos, dir, 0.0, energy, 0.0, PEMinus),
]

hits = mc_expectation(particles, targets_single, 1, medium);
compare_mc_model(particles, targets_single, models_casc, medium, hits)


particles_track = [
    Particle(pos .- 50 .* dir, dir, -50 / 0.3, 1E5, 100, PMuPlus)
]

propagated_particle, losses  = propagate_muon(particles_track[1])
length(losses)
hits_track = mc_expectation(losses, targets_single, 1);
compare_mc_model(losses, targets_single, models_casc, medium, hits_track)

c_n = c_at_wl(800., medium)
@load models_casc["1"] model tf_vec

feat_buffer = zeros(9, get_pmt_count(eltype(targets_hex))*length(targets_hex)*length(losses)) 

targets_range_mask = any(norm.([p.position for p in losses] .- permutedims([t.position for t in targets_hex])) .<= 200, dims=1)[1, :]
sample_multi_particle_event(losses, targets_hex[targets_range_mask], model, tf_vec, c_n, rng, feat_buffer=feat_buffer)

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
