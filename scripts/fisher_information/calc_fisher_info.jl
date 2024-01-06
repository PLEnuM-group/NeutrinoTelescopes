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
using Distributions

function make_injector(args, detector)

    cylinder = get_bounding_cylinder(detector)

    if !isnothing(args["li-file"])
        return LIInjector(args["li-file"], drop_starting=(args["type"] == "lightsabre"), volume=cylinder)
    else
        pdist = nothing
        ang_dist = nothing
        length_dist = nothing
        time_dist = Dirac(0.0)
        edist = truncated(Pareto(args["gamma"]-1, args["e_min"]), upper=args["e_max"])

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
    n_gaps = floor(Int64, 1000 / args["vert_spacing"])
    n_modules = n_gaps + 1 
    zstart = 0.5* n_gaps * args["vert_spacing"]

    ## HACK FOR P-ONE-1 lines
    if args["vert_spacing"] == 50
        n_modules = 20
        zstart = 475
    end
    
    if args["det"] == "cluster"
        targets = make_hex_detector(3, args["spacing"], n_modules, args["vert_spacing"], truncate=1, z_start=zstart)
    else
        targets = make_n_hex_cluster_detector(7, args["spacing"], n_modules, args["vert_spacing"], z_start=zstart)
    end

    d = Detector(targets, medium)
    hit_buffer = create_input_buffer(d, 1)
    out_buffer = create_output_buffer(d, 500)
    diff_cache = FixedSizeDiffCache(hit_buffer, 6)
    
    inj = make_injector(args, d)
    hit_generator = SurrogateModelHitGenerator(model, 200.0, hit_buffer, out_buffer)
        
    matrices, events = calc_fisher(d, inj, hit_generator, n_events, n_samples, use_grad=true, cache=diff_cache)
    cylinder = get_bounding_cylinder(d)
    
    results = (fisher_matrices = matrices, events=events, injection_volume=cylinder, spacing=args["spacing"], det=args["det"], n_modules=n_modules, vert_spacing=args["vert_spacing"], z_start=zstart,
               emin=args["e_min"], emax=args["e_max"], gamma=args["gamma"])

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
    default = "full"
    "--nevents"
    help ="Number of events"
    required = true
    arg_type = Int64
    "--spacing"
    help ="Detector spacing"
    required = true
    arg_type = Float64
    "--vert_spacing"
    help ="Vertical Detector spacing"
    required = false
    arg_type = Float64
    default = 50.
    "--gamma"
    help ="Spectral index when sampling in cylinder mode"
    required = false
    arg_type = Float64
    default = 2.
    "--e_max"
    help ="maximum energy when sampling in cylinder mode"
    required = false
    arg_type = Float64
    default = 1E6
    "--e_min"
    help ="minimum energy when sampling in cylinder mode"
    required = false
    arg_type = Float64
    default = 1E2
    "--li-file"
    help ="LeptonInjector file"
    required = false
    arg_type = String
end

run(parse_args(ARGS, s))



