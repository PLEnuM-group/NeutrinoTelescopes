using NeutrinoTelescopes
using PhotonPropagation
using PhysicsTools
using PreallocationTools
using Random
using Sobol
using StaticArrays
using DataStructures
using Distributions
using LinearAlgebra
using Flux
using JLD2
using ProgressBars
using DataFrames
using DataFrames
using ArgParse

function make_cascade_injector(vol)
    pdist = CategoricalSetDistribution(OrderedSet([PEPlus, PEMinus]), [0.5, 0.5])
    edist = Pareto(1, 1E3)
    ang_dist = UniformAngularDistribution()
    length_dist = Dirac(0.0)
    time_dist = Dirac(0.0)
    inj = VolumeInjector(vol, edist, pdist, ang_dist, length_dist, time_dist)
    return inj
end


function make_track_injector(cyl)
    surface = CylinderSurface(cyl)
    pdist = CategoricalSetDistribution(OrderedSet([PMuMinus, PMuPlus]), [0.5, 0.5])
    edist = Pareto(1, 1E3)
    ang_dist = LowerHalfSphere()
    length_dist = Dirac(0.0)
    time_dist = Dirac(0.0)
    inj = SurfaceInjector(surface, edist, pdist, ang_dist, length_dist, time_dist)
    return inj
end

function generate_training_data(args)

    model_tracks = PhotonSurrogate(args["model_path_amp"], args["model_path_time"])
    model = gpu(model_tracks)


    if args["per_string"]
        targets = [make_detector_line(SA_F32[0., 0., 0.], 20, 50, 1)]
        inj = args["type"] == "extended" ? make_cascade_injector(Cylinder(SA[0., 0., -475.], 1200., 100.)) : make_track_injector(Cylinder(SA[0., 0., -475.], 1200., 100.))
        input_buffer = create_input_buffer(16*20, 1)
        output_buffer = create_output_buffer(16*20, 100)
    else
        targets = [[POM(SA_F32[0., 0., 0.], 1)]]
        inj = args["type"] == "extended" ? make_cascade_injector(Sphere(SA[0., 0., 0.], 200.)) : make_track_injector(Cylinder(SA[0., 0., 0.], 200., 400.))
        input_buffer = create_input_buffer(16, 1)
        output_buffer = create_output_buffer(16, 100)
    end

    medium = make_cascadia_medium_properties(0.95f0)

   
    diff_cache = FixedSizeDiffCache(input_buffer, 6)
    hit_generator = SurrogateModelHitGenerator(model, 200.0, input_buffer, output_buffer)

    
    rng = MersenneTwister(args["seed"])

    detector = LineDetector(targets, medium)

    training_data = DataFrame(
        raw_input=Vector{Vector{Float64}}(undef, 0),
        fi=Vector{Matrix{Float64}}(undef, 0),
        chol_upper=Vector{Vector{Float64}}(undef, 0) )

    transformations = [x -> x for _ in 1:8]

    for i in 1:args["nevents"]
        event = rand(rng, inj)
        m, = calc_fisher_matrix(event, detector, hit_generator, use_grad=true, rng=rng, cache=diff_cache)
        #cov = inv(m)
        #cov = 0.5* (cov + cov')
        l = cholesky(m, check = false)
        if !issuccess(l)
            continue
        end

        p = first(event[:particles])
        # If we evaluate an entire string, use position SA[0., 0., -475.] as reference
        if args["per_string"]
            raw_input = FisherSurrogate.calculate_model_input([p], [[0f0, 0f0]], transformations)[:, 1]
        else
            raw_input = NeuralFlowSurrogate._calc_flow_input(p, targets[1], transformations)
        end
        push!(training_data, (raw_input=raw_input, fi=m, chol_upper=l.U[triu!((trues(6,6)))]))
    end

    jldsave(args["outfile"], data=training_data)
end

s = ArgParseSettings()
type_choices = ["lightsabre", "extended"]
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
    "--nevents"
    help ="Number of events"
    required = true
    arg_type = Int64
    "--seed"
    help = "RNG Seed"
    arg_type = Int64
    required = true
    "--per_string"
    help = "Calculate for an entire string"
    action = :store_true
end

parsed_args = parse_args(ARGS, s; as_symbols=false)

generate_training_data(parsed_args)