using NeutrinoTelescopes
using PhotonPropagation
using PhysicsTools
using Random
using StaticArrays
using BSON: @save, @load
using BSON
using CairoMakie
using LinearAlgebra
using DataFrames
using StatsBase
using Base.Iterators
using Distributions
using Formatting
using ForwardDiff
using DataStructures
using Flux
using Sobol
using JLD2
using BenchmarkTools
using PreallocationTools
using ArgParse

function run(args)

    model = PhotonSurrogate(args["model_path_amp"], args["model_path_time"])
    model = gpu(model)
    medium = make_cascadia_medium_properties(0.95f0)

    pdist = nothing
    ang_dist = nothing
    length_dist = nothing
 
    if args["type"] == "track"
        pdist = CategoricalSetDistribution(OrderedSet([PMuPlus, PMuMinus]), [0.5, 0.5])
        ang_dist = LowerHalfSphere()
        length_dist = Dirac(10000.)
    else
        pdist = CategoricalSetDistribution(OrderedSet([PEMinus, PEPlus]), [0.5, 0.5])
        ang_dist = UniformAngularDistribution()
        length_dist = Dirac(0.)
    end
   
    time_dist = Dirac(0.0)
    logenergies = 2:0.5:6
    spacings = 30:10:200

    n_samples = 30
    n_events = args["nevents"]

    results = []

    for spacing in spacings

        #spacing = next!(spacings)[1]

        targets = nothing
        if args["det"] == "cluster"
            targets = make_hex_detector(3, spacing, 20, 50, truncate=1)
        else
            targets = make_n_hex_cluster_detector(7, spacing, 20, 50)
        end
        d = Detector(targets, medium)
        hit_buffer = create_input_buffer(d, 1)
        cylinder = get_bounding_cylinder(d)
        surface = CylinderSurface(cylinder)

        buffer = (create_input_buffer(d, 1))
        diff_cache = FixedSizeDiffCache(buffer, 6)
           
        hit_generator = SurrogateModelHitGenerator(model, 200.0, hit_buffer)
        
        for le in logenergies
            edist = Dirac(10^le)
            if args["type"] == "track"
                inj = SurfaceInjector(surface, edist, pdist, ang_dist, length_dist, time_dist)
            else
                inj = VolumeInjector(cylinder, edist, pdist, ang_dist, length_dist, time_dist)
            end
            
            matrices, ec = calc_fisher(d, inj, hit_generator, n_events, n_samples, use_grad=true, cache=diff_cache)
            #push!(sds, f)
        
            push!(results, (matrices=matrices, spacing=spacing, log_energy=le, event_collection=ec))

        end
    end

    results = DataFrame(results)
    save(args["outfile"], Dict("results" => results))
end

s = ArgParseSettings()

type_choices = ["track", "cascade"]
det_choices = ["cluster", "full"]
@add_arg_table s begin
    "--outfile"
    help = "Output filename"
    arg_type = String
    required = true
    "--model_path_amp"
    help = "Amplitude model"
    arg_type = String
    required = true
    "--model_path_time"
    help = "Time model"
    arg_type = String
    required = true
    "--type"
    help = "Particle Type;  must be one of " * join(type_choices, ", ", " or ")
    range_tester = (x -> x in type_choices)
    default = "cascade"
    "--det"
    help = "Detector Type;  must be one of " * join(det_choices, ", ", " or ")
    range_tester = (x -> x in det_choices)
    default = "cluster"
    "--nevents"
    help ="Number of events"
    required = true
    arg_type = Int64
end

run(parse_args(ARGS, s))



