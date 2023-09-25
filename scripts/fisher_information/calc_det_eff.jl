using NeutrinoTelescopes
using Sobol
using PhotonPropagation
using PhysicsTools
using LinearAlgebra
using Flux
using DataFrames
using StaticArrays
using StatsBase
using JLD2
using DataStructures
using Distributions
using Logging: global_logger
using ProgressLogging
using TerminalLoggers
using PreallocationTools
using ArgParse
using Glob

global_logger(TerminalLogger(right_justify=120))

function calc_trigger_stats(log_exp_per_pmt)
    pmt_hits = exp.(log_exp_per_pmt) .>= 1

    distinct_pmts = sum(pmt_hits, dims=1)[1, 1, :]
    atleast_two_distinct = distinct_pmts .>= 2
    atleast_three_distinct = distinct_pmts .>= 3
    n_mod_thrsh_two = sum(atleast_two_distinct)
    n_mod_thrsh_three = sum(atleast_three_distinct)
    return n_mod_thrsh_two, n_mod_thrsh_three
end

function estimate_dir_variance(log_exp_per_pmt, modules_range)
    exp_per_mod = sum(exp.(log_exp_per_pmt), dims=1)[:]
    m_positions = [m.shape.position for m in modules_range]

    hit_positions = m_positions[exp_per_mod .> 1]
    max_lhit = 0
    if length(hit_positions) > 0
        hit_positions = reduce(hcat, m_positions[exp_per_mod .> 1])
        dists = norm.(eachslice(hit_positions .- reshape(hit_positions, 3, 1, size(hit_positions, 2)), dims=(2, 3)))
        max_lhit = maximum(dists)
    end
    weighted_mean = mean(m_positions, weights(exp_per_mod))

    rel_pos = m_positions .- Ref(weighted_mean)
    nrelpossq = norm.(rel_pos).^2
    variance_est = 1/sum((nrelpossq .* exp_per_mod))

    return variance_est, max_lhit, nrelpossq
end


function calc_hit_stats(particles, model, detector; feat_buffer)
    
    modules = get_detector_modules(detector)
    modules_range_mask = get_modules_in_range(particles, detector, 200)
    modules_range = (modules[modules_range_mask])
    # npmt, 1, ntargets
    log_exp_per_pmt, _ = get_log_amplitudes(particles, modules_range, gpu(model); feat_buffer=feat_buffer)

    n_mod_thrsh_two, n_mod_thrsh_three = calc_trigger_stats(log_exp_per_pmt)
    n_total = sum(exp.(log_exp_per_pmt))
    stats = (
        n_mod_thrsh_two=n_mod_thrsh_two,
        n_mod_thrsh_three=n_mod_thrsh_three,
        n_total=n_total)
    return stats
end

function calc_geo_stats(particles, cylinder)
    particle = first(particles)
    isec = get_intersection(cylinder, particle)
    if isnothing(isec.first)
        length_in = 0
    else
        length_in = isec.second - isec.first
    end
    proj_area = projected_area(cylinder, particle.direction)
    return (proj_area=proj_area, length_in=length_in)
end

function calculate_efficiencies(args)

    PKGDIR = pkgdir(NeutrinoTelescopes)

    weighter = WeighterPySpline(joinpath(PKGDIR, "assets/transm_inter_splines.pickle"))

    model = PhotonSurrogate(args.model_path_amp, arg.model_path.time)
    medium = make_cascadia_medium_properties(0.95f0)

    pdist = nothing
    ang_dist = nothing
    length_dist = nothing
    model = nothing

    if args["type"] == "track"
        pdist = CategoricalSetDistribution(OrderedSet([PMuPlus, PMuMinus]), [0.5, 0.5])
        ang_dist = LowerHalfSphere()
        length_dist = Dirac(10000.)
    elseif args["type"] == "cascade"
        pdist = CategoricalSetDistribution(OrderedSet([PEMinus, PEPlus]), [0.5, 0.5])
        ang_dist = UniformAngularDistribution()
        length_dist = Dirac(0.)
    else
        error("Unknown choice $(args["type"])")
    end

    #spacings = SobolSeq(30, 200)

    spacings = 30:10:200

    eff_d = []
    logenergies = 3:0.2:6
    time_dist = Dirac(0.0)
    @progress name="spacings" for spacing in spacings

        #spacing = next!(spacings)[1]

        if args["det"] == "cluster"
            targets = make_hex_detector(3, spacing, 20, 50, truncate=1)
        elseif args["det"] == "full"
            targets = make_n_hex_cluster_detector(7, spacing, 20, 50)
        else
            error("Unknown choice $(args["det"])")
        end


        d = Detector(targets, medium)
        hit_buffer = create_input_buffer(d, 1)
        cylinder = get_bounding_cylinder(d, padding_top=100, padding_side=100)
        surface = CylinderSurface(cylinder)

        sim_volume = get_volume(cylinder)

        modules = get_detector_modules(d)
        medium = get_detector_medium(d)

        @progress name="energies" for le in logenergies
            edist = Dirac(10^le)
            if args["type"] == "track"
                inj = SurfaceInjector(surface, edist, pdist, ang_dist, length_dist, time_dist)
            else
                inj = VolumeInjector(cylinder, edist, pdist, ang_dist, length_dist, time_dist)
            end

            @progress name ="events" for i in 1:args["nevents"]
                ev = rand(inj)
                particles = ev[:particles]

                stats_hits = calc_hit_stats(particles, model, detector; feat_buffer=hit_buffer)
                geo_stats = calc_geo_stats(particles, cylinder)

                if args["type"] == "cascade"
                    pdir = cart_to_sph(particles[1].direction)
                    ptype = rand([PNuE, PNuEBar])
                    total_prob = get_total_prob(weighter, ptype, :NU_CC, log10(particles[1].energy), pdir[1], length_in)
                else
                    total_prob = 0.
                end

                other_stats = (dir_theta=theta, dir_phi=phi, log_energy=le, spacing=spacing, sim_volume=sim_volume, total_prob)
                stats = merge(stats_hits, geo_stats, other_stats)
                push(eff_d, stats)


            end
        end
    end
    jldsave(args["outfile"], results=DataFrame(eff_d))
    return nothing
end


function calculate_for_events(args)

    # = WeighterPySpline(joinpath(PKGDIR, "assets/transm_inter_splines.pickle"))

    model = PhotonSurrogate(args["model_path_amp"], args["model_path_time"])
    medium = make_cascadia_medium_properties(0.95f0)


    eff_d = []
    res = jldopen(args["infile"])["results"]
    cyl = res[:injection_volume]
    medium = make_cascadia_medium_properties(0.95)
    if args["det"] == "cluster"
        targets = make_hex_detector(3, res[:spacing], 20, 50, truncate=1, z_start=475)
    else
        targets = make_n_hex_cluster_detector(7, res[:spacing], 20, 50, z_start=475)
    end
    detector = Detector(targets, medium)
    hit_buffer = create_input_buffer(detector, 1)
    sim_volume = get_volume(cyl)

    @progress for (_, e) in zip(res[:fisher_matrices], res[:events])
        
        particles = e[:particles]
        particle = first(particles)

        stats_hits = calc_hit_stats(particles, model, detector; feat_buffer=hit_buffer)
        geo_stats = calc_geo_stats(particles, cyl)

        #=
        if args["type"] == "cascade"
            pdir = cart_to_sph(particle.direction)
            ptype = rand([PNuE, PNuEBar])
            total_prob = get_total_prob(weighter, ptype, :NU_CC, log10(particle.energy), pdir[1], geo_stats[:length_in])
        else
            total_prob = 0.
        end
        =#
        theta, phi = cart_to_sph(particle.direction)

        other_stats = (dir_theta=theta, dir_phi=phi, log_energy=log10(particle.energy), spacing=res[:spacing], sim_volume=sim_volume, weight=e[:weight])
        stats = merge(stats_hits, geo_stats, other_stats)
        push!(eff_d, stats)

    end
    results=DataFrame(eff_d)
    jldsave(args["outfile"], results=results)
    #return results
end


s = ArgParseSettings()

det_choices = ["cluster", "full"]
@add_arg_table s begin
    "--outfile"
    help = "Output filename"
    arg_type = String
    required = true
    "--infile"
    help = "Input filename"
    arg_type = String
    required = true
    "--det"
    help = "Detector type;  must be one of " * join(det_choices, ", ", " or ")
    range_tester = (x -> x in det_choices)
    default = "full"
    "--model_path_amp"
    help = "Amplitude model"
    arg_type = String
    required = true
    "--model_path_time"
    help = "Time model"
    arg_type = String
    required = true
end

args = parse_args(s)
calculate_for_events(args)