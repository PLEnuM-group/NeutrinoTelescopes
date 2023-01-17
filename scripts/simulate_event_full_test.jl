using NeutrinoTelescopes
using StaticArrays
using DataFrames
using LinearAlgebra
using Rotations
using Parquet
using HDF5
using CairoMakie
using CUDA

medium = make_cascadia_medium_properties(0.99f0)

targets_line = make_detector_line(@SVector[-25, 0.0, 0.0], 20, 50, 1)
targets_line = convert.(MultiPMTDetector{Float32}, targets_line)

id_target_map = Dict([targ.module_id => targ for targ in targets_line])


zenith_angle = 90.0f0
azimuth_angle = 0.0f0

p0 = @SVector[0.0f0, 0.0f0, 0.0f0]
dir = sph_to_cart(deg2rad(zenith_angle), deg2rad(azimuth_angle))
pos = p0 .- 100 .* dir

particle = Particle(
    pos,
    dir,
    0.0f0,
    Float32(1E5),
    200.0f0,
    PMuMinus
)


particle2 = Particle(
    pos,
    dir,
    0.0f0,
    Float32(1E4),
    0.0f0,
    PEMinus
)


cher_spectrum = CherenkovSpectrum((300.0f0, 800.0f0), 30, medium)
prop_source_muon = CherenkovTrackEmitter(particle, medium, (300.0f0, 800.0f0))
prop_source_ext = ExtendedCherenkovEmitter(particle2, medium, (300.0f0, 800.0f0))


setup = PhotonPropSetup([prop_source_muon], targets_line, medium, cher_spectrum, 1)
photons = propagate_photons(setup)

orientation = RotMatrix3(I)
hits = make_hits_from_photons(photons, setup, orientation)
calc_total_weight!(hits, setup)
calc_time_residual_tracks!(hits, setup)
hsel = hits[hits[:, :pmt_id].==14, :]
hist(hsel[:, :tres], weights=hsel[:, :total_weight], bins=-10:1:20)

hsel

resampled_hits = resample_simulation(hits)
calc_time_residual!(resampled_hits, setup)
res_grp_pmt = groupby(resampled_hits, [:module_id, :pmt_id]);


geo = DataFrame([(
    module_id=Int64(target.module_id),
    pmt_id=Int64(pmt_id),
    x=target.position[1],
    y=target.position[2],
    z=target.position[3],
    pmt_theta=coord[1],
    pmt_phi=coord[2])
                 for target in targets
                 for (pmt_id, coord) in enumerate(eachcol(target.pmt_coordinates))]
)

geo

outfile = joinpath(@__DIR__, "../assets/event_test_3str.hd5")

resampled_hits

h5open(outfile, "w") do hdl

    g = create_group(hdl, "event_1")

    write(g, "photons", Matrix{Float64}(resampled_hits))
    attributes(g)["energy"] = particle.energy
    attributes(g)["position"] = particle.position
    attributes(g)["direction"] = particle.direction
    attributes(g)["time"] = particle.time

    g = create_group(hdl, "geometry")
    write(g, "geo", geo |> Matrix)

end


write_parquet(joinpath(outdir, "geometry_test_3str.parquet"), geo)
write_parquet(joinpath(outdir, "event_test_3str.parquet"), resampled_hits)
