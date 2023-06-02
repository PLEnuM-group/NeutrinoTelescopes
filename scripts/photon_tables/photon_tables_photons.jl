using NeutrinoTelescopes
using StaticArrays
using Random
using DataFrames
using Formatting
using Distributions
using Rotations
using LinearAlgebra
using HDF5
using Sobol
using ArgParse
using PhysicsTools
using PhotonPropagation
using JSON3

include("utils.jl")


function run_sim(
    energy,
    distance,
    dir_costheta,
    dir_phi,
    output_fname,
    seed,
    mode=:extended,
    g=0.95)

    direction::SVector{3,Float32} = sph_to_cart(acos(dir_costheta), dir_phi)

    if mode == :bare_infinite_track
        r = direction[1] / direction[2]
        ppos = SA_F32[distance/sqrt(1 + r^2), -r*distance/sqrt(1 + r^2), 0]
    else
        ppos = SA_F32[0, 0, distance]
    end

    sim_attrs = Dict(
        "energy" => energy,
        "mode" => string(mode),
        "distance" => distance,
        "dir_costheta" => dir_costheta,
        "dir_phi" => dir_phi,
        "seed" => seed,
        "source_pos" => JSON3.write(ppos),
        "g" => g,
    )

    base_weight = 1.0
    photons = nothing

    setup = make_setup(mode, ppos, direction, energy, seed, g=g)

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

    # if more than 1E6 photons make it to the module,
    # take the first 1E6 and scale weights

    n_ph_limit = 1000000

    if nph_sim > n_ph_limit
        photons = photons[1:n_ph_limit, :]
        base_weight *= nph_sim / n_ph_limit
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
    g = parsed_args["g"]

    if mode == :extended
        sobol = skip(
            SobolSeq(
                [log10(e_min), log10(dist_min), -1, 0],
                [log10(e_max), log10(dist_max), 1, 2 * π]),
            n_sims + n_skip)

        for i in 1:n_sims

            pars = next!(sobol)
            energy = 10^pars[1]
            distance = Float32(10^pars[2])
            dir_costheta = pars[3]
            dir_phi = pars[4]

            run_sim(energy, distance, dir_costheta, dir_phi, parsed_args["output"], i + n_skip, mode, g)
        end
    elseif mode == :bare_infinite_track
        sobol = skip(
            SobolSeq(
                [log10(dist_min), -1, 0],
                [log10(dist_max), 1, 2 * π]),
            n_sims + n_skip)

        for i in 1:n_sims

            pars = next!(sobol)
            energy = 1E5
            distance = Float32(10^pars[1])
            dir_costheta = pars[2]
            dir_phi = pars[3]

            run_sim(energy, distance, dir_costheta, dir_phi, parsed_args["output"], i + n_skip, mode, g)
        end
    else
        sobol = skip(
            SobolSeq([log10(dist_min), -1], [log10(dist_max), 1]),
            n_sims + n_skip)

        for i in 1:n_sims
            pars = next!(sobol)
            energy = 1E5
            distance = Float32(10^pars[1])
            dir_costheta = pars[2]
            dir_phi = 0

            run_sim(energy, distance, dir_costheta, dir_phi, parsed_args["output"], i + n_skip, mode, g)
        end
    end
end

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
    range_tester = (x -> x in mode_choices)
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
    "--g"
    help = "Mean scattering angle"
    arg_type = Float64
    required = false
    default = 0.95
end
parsed_args = parse_args(ARGS, s)


run_sims(parsed_args)
