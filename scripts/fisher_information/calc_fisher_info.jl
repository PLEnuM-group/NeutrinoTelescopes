using NeutrinoTelescopes
using PhotonPropagation
using PhysicsTools
using Random
using Flux
using DataFrames
using DataStructures
using JLD2
using PreallocationTools
using ArgParse

function make_injector(args, detector)

    cylinder = get_bounding_cylinder(detector)

    if haskey(args, "li-file")
        return LIInjector(args["li-file"], drop_starting=(args["type"] == "lightsabre"), volume=cylinder)
    else
        pdist = nothing
        ang_dist = nothing
        length_dist = nothing
        time_dist = Dirac(0.0)
        edist = Dirac(10^args["log-energy"])

        if args["type"] == "lightsabre"
            pdist = CategoricalSetDistribution(OrderedSet([PMuPlus, PMuMinus]), [0.5, 0.5])
            ang_dist = LowerHalfSphere()
            length_dist = Dirac(10000.)
            surface = CylinderSurface(cylinder)
            return  SurfaceInjector(surface, edist, pdist, ang_dist, length_dist, time_dist)
        elseif args["type"] == "extended"
            pdist = CategoricalSetDistribution(OrderedSet([PEMinus, PEPlus]), [0.5, 0.5])
            ang_dist = UniformAngularDistribution()
            length_dist = Dirac(0.)
            return VolumeInjector(cylinder, edist, pdist, ang_dist, length_dist, time_dist)
        else
            error("Unknown event type")
        end
    end
end


function run(args)

    model = PhotonSurrogate(args["model_path_amp"], args["model_path_time"])
    model = gpu(model)
    medium = make_cascadia_medium_properties(0.95f0)

    n_samples = 150
    n_events = args["nevents"]

    results = []

    targets = nothing

    if args["det"] == "cluster"
        targets = make_hex_detector(3, args["spacing"], 20, 50, truncate=1, z_start=475)
    else
        targets = make_n_hex_cluster_detector(7, args["spacing"], 20, 50, z_start=475)
    end

    d = Detector(targets, medium)
    hit_buffer = create_input_buffer(d, 1)
    diff_cache = FixedSizeDiffCache(hit_buffer, 6)
    
    inj = make_injector(args, d)
    hit_generator = SurrogateModelHitGenerator(model, 200.0, hit_buffer)
        
    matrices, events = calc_fisher(d, inj, hit_generator, n_events, n_samples, use_grad=true, cache=diff_cache)
    
    results = (fisher_matrices = matrices, events=events)

    JLD2.save(args["outfile"], Dict("results" => results))
end

s = ArgParseSettings()

type_choices = ["lightsabre", "extended"]
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
    default = "extended"
    "--det"
    help = "Detector Type;  must be one of " * join(det_choices, ", ", " or ")
    range_tester = (x -> x in det_choices)
    default = "cluster"
    "--nevents"
    help ="Number of events"
    required = true
    arg_type = Int64
    "--spacing"
    help ="Detector spacing"
    required = true
    arg_type = Float64
    "--log-energy"
    help ="Log10(Energy)"
    required = true
    arg_type = Float64
    "--li-file"
    help ="LeptonInjector file"
    required = false
    arg_type = String
end

run(parse_args(ARGS, s))



