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
global_logger(TerminalLogger(right_justify=120))


function calculate_efficiencies(args)

    model_path = joinpath(ENV["WORK"], "time_surrogate")
    models_casc = Dict(
        "Model A" =>  PhotonSurrogate(joinpath(model_path, "extended/amplitude_1_FNL.bson"), joinpath(model_path, "extended/time_1_FNL.bson")),
        "Model B" =>  PhotonSurrogate(joinpath(model_path, "extended/amplitude_2_FNL.bson"), joinpath(model_path, "extended/time_2_FNL.bson")),
        "Model C" =>  PhotonSurrogate(joinpath(model_path, "extended/amplitude_3_FNL.bson"), joinpath(model_path, "extended/time_4_FNL.bson")),

    )

    models_track = Dict(
        "Model A" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_1_FNL.bson"), joinpath(model_path, "lightsabre/time_1_FNL.bson")),
        "Model B" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_2_FNL.bson"), joinpath(model_path, "lightsabre/time_2_FNL.bson")),
        "Model C" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_3_FNL.bson"), joinpath(model_path, "lightsabre/time_4_FNL.bson")),

    )

    medium = make_cascadia_medium_properties(0.95f0)

    pdist = nothing
    ang_dist = nothing
    length_dist = nothing
    model = nothing

    if args["type"] == "track"
        pdist = CategoricalSetDistribution(OrderedSet([PMuPlus, PMuMinus]), [0.5, 0.5])
        ang_dist = LowerHalfSphere()
        length_dist = Dirac(10000.)
        model = models_track["Model A"]
    elseif args["type"] == "cascade"
        pdist = CategoricalSetDistribution(OrderedSet([PEMinus, PEPlus]), [0.5, 0.5])
        ang_dist = UniformAngularDistribution()
        length_dist = Dirac(0.)
        model = models_casc["Model A"]
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
                isec = get_intersection(cylinder, ev[:particles][1])
        
                length_in = isec.second - isec.first

                particles = ev[:particles]

                modules_range_mask = get_modules_in_range(particles, d, 200)
                modules_range = (modules[modules_range_mask])
                # npmt, 1, ntargets
                log_exp_per_pmt, _ = get_log_amplitudes(particles, modules_range, gpu(model); feat_buffer=hit_buffer)
                
                pmt_hits = exp.(log_exp_per_pmt) .>= 1

                distinct_pmts = sum(pmt_hits, dims=1)[1, 1, :]
                atleast_two_distinct = distinct_pmts .>= 2
                atleast_three_distinct = distinct_pmts .>= 3
                n_mod_thrsh_two = sum(atleast_two_distinct)
                n_mod_thrsh_three = sum(atleast_three_distinct)

                # PONE OFFLINE TRIGGER

                n_total = sum(exp.(log_exp_per_pmt))
                theta, phi = cart_to_sph(particles[1].direction)

                
                if args["type"] == "track"
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

                    #n_mod_thrsh = sum(any(exp_per_module .>= 2, dims=1))
                    proj_area = projected_area(cylinder, particles[1].direction)
                   
                    push!(eff_d, (n_mod_thrsh_two=n_mod_thrsh_two, n_mod_thrsh_three=n_mod_thrsh_three, dir_theta=theta, dir_phi=phi, length=length_in, log_energy=le, spacing=spacing, n_total,
                                nrelpossq=sum(nrelpossq), variance_est=variance_est, max_lhit=max_lhit, proj_area=proj_area))
                else
                    push!(eff_d, (n_mod_thrsh_two=n_mod_thrsh_two, n_mod_thrsh_three=n_mod_thrsh_three, dir_theta=theta, dir_phi=phi, length=length_in, log_energy=le, spacing=spacing, n_total, sim_volume=sim_volume))
            
                end
            end
        end
    end
    jldsave(args["outfile"], results=DataFrame(eff_d))
    return nothing
end


s = ArgParseSettings()

type_choices = ["track", "cascade"]
det_choices = ["cluster", "full"]
@add_arg_table s begin
    "--outfile"
    help = "Output filename"
    arg_type = String
    required = true
    "--type"
    help = "Particle Type;  must be one of " * join(type_choices, ", ", " or ")
    range_tester = (x -> x in type_choices)
    default = "cascade"
    "--det"
    help = "Detector type;  must be one of " * join(det_choices, ", ", " or ")
    range_tester = (x -> x in det_choices)
    default = "full"
    "--nevents"
    help ="Number of events"
    required = true
    arg_type = Int64
end

calculate_efficiencies(parse_args(ARGS, s))