using NeutrinoTelescopes
using StaticArrays
using Random
using DataFrames
using ProgressLogging
using Formatting
using Distributions
using Rotations
using LinearAlgebra
using HDF5
using TerminalLoggers
using Logging: global_logger
using Sobol
using ArgParse

function save_hdf!(
    fname::AbstractString,
    group::AbstractString,
    dataset::Matrix,
    attributes::Dict)


    if isfile(fname)
        fid = h5open(fname, "r+")
    else
        fid = h5open(fname, "w")
    end

    if !haskey(fid, group)
        g = create_group(fid, group)
        HDF5.attrs(g)["nsims"] = 0
    else
        g = fid[group]
    end


    offset = HDF5.read_attribute(g, "nsims") + 1
    ds_name = format("dataset_{:d}", offset)

    g[ds_name] = dataset# Matrix{Float64}(res.hits[:, [:tres, :pmt_id]])
    f_attrs = HDF5.attrs(g[ds_name])
    for (k, v) in attributes
        f_attrs[k] = v
    end

    HDF5.attrs(g)["nsims"] = offset

    close(fid)
end


function make_setup(
    mode, pos, dir, energy, seed;
    g=0.99f0)

    medium = make_cascadia_medium_properties(g)
    pmt_area = Float32((75e-3 / 2)^2 * π)
    target_radius = 0.21f0
    target = MultiPMTDetector(
        @SVector[0.0f0, 0.0f0, 0.0f0],
        target_radius,
        pmt_area,
        make_pom_pmt_coordinates(Float32),
        UInt16(1)
    )
    wl_range = (300.0f0, 800.0f0)

    spectrum = CherenkovSpectrum(wl_range, 30, medium)

    if mode == :extended
        particle = Particle(
            pos,
            dir,
            0.0f0,
            Float32(energy),
            0.0f0,
            PEMinus
        )
        source = ExtendedCherenkovEmitter(particle, medium, wl_range)
    elseif mode == :bare_infinite_track
        ppos = pos .- 200 .* dir

        particle = Particle(
            ppos,
            dir,
            0.0f0,
            Float32(energy),
            400.0f0,
            PMuMinus
        )
        source = CherenkovTrackEmitter(particle, medium, wl_range)
    else
        particle = Particle(
            pos,
            dir,
            0.0f0,
            Float32(energy),
            0.0f0,
            PEMinus)
        source = PointlikeChernekovEmitter(particle, medium, wl_range)
    end

    setup = PhotonPropSetup(source, target, medium, spectrum, seed)
    return setup

end


function run_sim(
    energy,
    distance,
    dir_costheta,
    dir_phi,
    output_fname,
    seed,
    n_resample=100;
    mode=:extended)

    sim_attrs = Dict(
        "energy" => energy,
        "distance" => distance,
        "dir_costheta" => dir_costheta,
        "dir_phi" => dir_phi)


    direction::SVector{3,Float32} = sph_to_cart(acos(dir_costheta), dir_phi)
        
    if mode == :bare_infinite_track
        r = direction[1] / direction[2]
        ppos = SA_F32[distance / sqrt(1 + r^2), -r*distance / sqrt(1+r^2), 0]
    else
        ppos = SA_F32[0, 0, distance]
    end

    

    base_weight = 1.0
    photons = nothing

    setup = make_setup(mode, ppos, direction, energy, seed)

    while true
        prop_source = setup.sources[1]
        if prop_source.photons > 1E13
            println("More than 1E13 photons, skipping")
            return nothing
        end
        photons = propagate_photons(setup)

        if nrow(photons) > 100
            break
        end

        setup.sources[1] = oversample_source(prop_source, 10)
        println(format("distance {:.2f} photons: {:d}", distance, setup.sources[1].photons))
        base_weight /= 10.0

    end

    nph_sim = nrow(photons)

    if nph_sim > 1E6
        photons = photons[1:1000000, :]
        base_weight *= nph_sim / 1E6
    end

    calc_time_residual!(photons, setup)
    transform!(photons, :position => (p -> reduce(hcat, p)') => [:pos_x, :pos_y, :pos_z])
    calc_total_weight!(photons, setup)
    photons[!, :total_weight] .*= base_weight

    save_hdf!(
        output_fname,
        "photons",
        Matrix{Float64}(photons[:, [:tres, :pos_x, :pos_y, :pos_z, :total_weight]]),
        sim_attrs)

    @progress "Resampling" for j in 1:n_resample
        #=
        PMT positions are defined in a standard upright coordinate system centeres at the module
        Sample a random rotation matrix and rotate the pmts on the module accordingly.
        =#
        orientation = rand(RotMatrix3)
        hits = make_hits_from_photons(photons, setup, orientation)

        if nrow(hits) < 10
            continue
        end


        hits = resample_simulation(hits; per_pmt=true, time_col=:tres)
        #hits[!, :total_weight] ./= oversample
        if nrow(hits) == 0
            continue
        end

        #=
        Rotating the module (active rotation) is equivalent to rotating the coordinate system
        (passive rotation). Hence rotate the position and the direction of the light source with the
        inverse rotation matrix to obtain a description in which the module axis is again aligned with ez
        =#
        direction_rot = orientation' * direction
        position_rot = orientation' * ppos

        position_rot_normed = position_rot ./ norm(position_rot)
        dir_theta, dir_phi = cart_to_sph(direction_rot)
        pos_theta, pos_phi = cart_to_sph(position_rot_normed)

        #= Sanity check:
        if !((dot(ppos / norm(ppos), direction) ≈ dot(position_rot_normed, direction_rot)))
            error("Relative angle not perseved: $(dot(ppos / norm(ppos), direction)) vs. $(dot(position_rot_normed, direction_rot))")
        end
        =#

        sim_attrs["dir_theta"] = dir_theta
        sim_attrs["dir_phi"] = dir_phi
        sim_attrs["pos_theta"] = pos_theta
        sim_attrs["pos_phi"] = pos_phi

        save_hdf!(
            output_fname,
            "pmt_hits",
            Matrix{Float64}(hits[:, [:tres, :pmt_id]]),
            sim_attrs)
    end
end


function run_sims(parsed_args)

    #=
    parsed_args = Dict("n_sims"=>1, "n_skip"=>0)
    =#

    n_sims = parsed_args["n_sims"]
    n_skip = parsed_args["n_skip"]
    mode = Symbol(parsed_args["mode"])
    n_resample = parsed_args["n_resample"]
    e_min = parsed_args["e_min"]
    e_max = parsed_args["e_max"]
    dist_min = parsed_args["dist_min"]
    dist_max = parsed_args["dist_max"]

    if mode == :extended
        sobol = skip(
            SobolSeq(
                [log10(e_min), log10(dist_min), -1, 0],
                [log10(e_max), log10(dist_max), 1, 2 * π]),
            n_sims + n_skip)

        @progress "Photon sims" for i in 1:n_sims

            pars = next!(sobol)
            energy = 10^pars[1]
            distance = Float32(10^pars[2])
            dir_costheta = pars[3]
            dir_phi = pars[4]

            run_sim(energy, distance, dir_costheta, dir_phi, parsed_args["output"], i + n_skip, n_resample, mode=mode)
        end
    elseif mode == :bare_infinite_track
        sobol = skip(
            SobolSeq(
                [log10(dist_min), -1, 0],
                [log10(dist_max), 1, 2 * π]),
            n_sims + n_skip)

        @progress "Photon sims" for i in 1:n_sims

            pars = next!(sobol)
            energy = 1E5
            distance = Float32(10^pars[1])
            dir_costheta = pars[2]
            dir_phi = pars[3]

            run_sim(energy, distance, dir_costheta, dir_phi, parsed_args["output"], i + n_skip, n_resample, mode=mode)
        end
    else
        sobol = skip(
            SobolSeq([log10(dist_min), -1], [log10(dist_max), 1]),
            n_sims + n_skip)

        @progress "Photon sims" for i in 1:n_sims
            pars = next!(sobol)
            energy = 1E5
            distance = Float32(10^pars[1])
            dir_costheta = pars[2]
            dir_phi = 0

            run_sim(energy, distance, dir_costheta, dir_phi, parsed_args["output"], i + n_skip, n_resample, mode=mode)
        end
    end
end

global_logger(TerminalLogger(right_justify=120))

s = ArgParseSettings()

mode_choices = ["extended", "bare_infinite_track", "pointlike"]

@add_arg_table s begin
    "--output"
    help = "Output filename"
    arg_type = String
    required = true
    "--n_sims"
    help = "Number of simulations"
    arg_type = Int
    required = true
    "--n_skip"
    help = "Skip in Sobol sequence"
    arg_type = Int
    required = false
    default = 0
    "--n_resample"
    help = "Number of resamples per photon sim"
    arg_type = Int
    required = false
    default = 100
    "--mode"
    help = "Simulation Mode;  must be one of " * join(mode_choices, ", ", " or ")
    range_tester = (x->x in mode_choices)
    default = "extended"
    "--e_min"
    help = "Minimum energy"
    arg_type = Float64
    required = false
    default = 100.0
    "--e_max"
    help = "Maximum energy"
    arg_type = Float64
    required = false
    default = 1E5
    "--dist_min"
    help = "Minimum distance"
    arg_type = Float64
    required = false
    default = 10.0
    "--dist_max"
    help = "Maximum distance"
    arg_type = Float64
    required = false
    default = 150.0
end
parsed_args = parse_args(ARGS, s)


run_sims(parsed_args)
