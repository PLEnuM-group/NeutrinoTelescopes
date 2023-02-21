using PhotonPropagation
using NeutrinoTelescopes
using PhysicsTools
using StaticArrays
using DataFrames
using Rotations
using LinearAlgebra
using Random
using JLD2
using JSON
using StatsBase
using HDF5
using Arrow


function make_biolumi_sources_from_positions(positions, n_ph, trange)

    positions = SVector{3,Float32}.(positions)
    mask = norm.(positions) .> 0.5
    positions = positions[mask]

    sources = Vector{PointlikeTimeRangeEmitter}(undef, length(positions))

    for (i, pos) in enumerate(positions)
        sources[i] = PointlikeTimeRangeEmitter(
            SVector{3,Float32}(pos),
            (0.0, trange),
            Int64(n_ph)
        )
    end

    return sources
end

function run_sim(target, sources, seed)
    medium = make_cascadia_medium_properties(0.99f0)
    mono_spec = Monochromatic(420.0f0)
    orientation = RotMatrix3(I)

    setup = PhotonPropSetup(sources, [target], medium, mono_spec, seed)
    photons = propagate_photons(setup)
    hits = make_hits_from_photons(photons, setup, orientation)
    calc_total_weight!(hits, setup)
    return hits
end


trange = 1E7
pmt_area = Float32((75e-3 / 2)^2 * π)
target_radius = 0.21f0

target = MultiPMTDetector(
    @SVector[0.0f0, 0.0f0, 0.0f0],
    target_radius,
    pmt_area,
    make_pom_pmt_coordinates(Float32),
    UInt16(1))

# up-looking
target_1pmt = MultiPMTDetector(
    @SVector[0.0f0, 0.0f0, 0.0f0],
    target_radius,
    pmt_area,
    SA_F32[0 0]',
    UInt16(1))

bio_pos_df = Vector{Float64}.(JSON.parsefile(joinpath(@__DIR__, "../assets/relative_emission_positions.json")))

outpath = joinpath(@__DIR__, "../data/biolumi_sims")

if !ispath(outpath)
    mkdir(outpath)
end

for n_sources in [1, 5, 10, 50, 100]
    for i in 1:50
        n_ph = Int(1E9 / n_sources)
        seed = i
        rng = MersenneTwister(seed)
        bio_sources_fd = sample(rng, make_biolumi_sources_from_positions(bio_pos_df, n_ph, trange), n_sources, replace=false)

        hits = run_sim(target, bio_sources_fd, seed)
        hits_1pmt = run_sim(target_1pmt, bio_sources_fd, seed)

        if nrow(hits_1pmt) == 0
            continue
        end
        single_pmt_rate = (sum(hits_1pmt[:, :total_weight]) / trange) * 1E9 / 1E3 #[kHz]

        fname_meta = joinpath(outpath, "meta_$(n_sources)_$i.json")
        fname_table = joinpath(outpath, "hits_$(n_sources)_$i.arrow")

        metadata = Dict(
            "target" => target,
            "target_1pmt" => target_1pmt,
            "sources" => bio_sources_fd,
            "single_pmt_rate" => single_pmt_rate)

        Arrow.write(fname_table, hits, metadata=["metadata_json" => JSON.json(metadata),])
    end
end