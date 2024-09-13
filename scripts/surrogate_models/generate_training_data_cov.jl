using NeutrinoTelescopes
using PhotonPropagation
using PhotonSurrogateModel
using NeutrinoSurrogateModelData
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
    length_dist = Dirac(1E4)
    time_dist = Dirac(0.0)
    inj = SurfaceInjector(surface, edist, pdist, ang_dist, length_dist, time_dist)
    return inj
end

function generate_training_data(args)

    model = args["type"] == "extended" ? PhotonSurrogate(em_cascade_time_model(args["time_uncert"])...) : PhotonSurrogate(lightsabre_time_model(args["time_uncert"])...)
    model = gpu(model)


    if args["per_string"]
        targets = [make_detector_line(SA_F32[0., 0., 0.], 20, 50, 1)]
        inj = args["type"] == "extended" ? make_cascade_injector(Cylinder(SA[0., 0., -475.], 1200., 150.)) : make_track_injector(Cylinder(SA[0., 0., -475.], 1200., 150.))
        input_buffer = create_input_buffer(model, 16*20, 1)
        output_buffer = create_output_buffer(16*20, 100)
    else
        targets = [[POM(SA_F32[0., 0., 0.], 1)]]
        inj = args["type"] == "extended" ? make_cascade_injector(Sphere(SA[0., 0., 0.], 200.)) : make_track_injector(Cylinder(SA[0., 0., 0.], 200., 400.))
        input_buffer = create_input_buffer(model, 16, 1)
        output_buffer = create_output_buffer(16, 100)
    end
   

   
    diff_cache = DiffCache(input_buffer, 13)
    rng = MersenneTwister(args["seed"])

    training_data = DataFrame(
        raw_input=Vector{Vector{Float64}}(undef, 0),
        fi=Vector{Matrix{Float64}}(undef, 0),
        chol_upper=Vector{Vector{Float64}}(undef, 0) )

    transformations = [x -> x for _ in 1:8]

    medium = make_cascadia_medium_properties(0.95f0)
    hit_generator = SurrogateModelHitGenerator(model, 200.0, input_buffer, output_buffer)
    detector = LineDetector(targets, medium)

    for i in 1:args["nevents"]

        if args["perturb_medium"]
            abs_scale = 1 + randn(rng)*0.05
            sca_scale = 1 + randn(rng)*0.05
        else
            abs_scale = 1
            sca_scale = 1
        end
        
        event = rand(rng, inj)

        particle = shift_to_closest_approach(first(event[:particles]), [0f0, 0f0, -475f0])

        m, = calc_fisher_matrix(particle, detector, hit_generator, use_grad=true, rng=rng, cache=diff_cache, abs_scale=abs_scale, sca_scale=sca_scale, n_samples=200)
        #cov = inv(m)
        #cov = 0.5* (cov + cov')
        l = cholesky(m, check = false)
        if !issuccess(l)
            continue
        end

        # If we evaluate an entire string, use position SA[0., 0., -475.] as reference
        if args["per_string"]
            raw_input = FisherSurrogate.calculate_model_input([particle], [[0f0, 0f0]], transformations, abs_scale=abs_scale, sca_scale=sca_scale)[:, 1]
        else
            raw_input = NeuralFlowSurrogate.create_flow_input(particle, targets[1], transformations)
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
    "--perturb_medium"
    help = "perturb optical properties"
    action = :store_true
    "--time_uncert"
    help = "Timing uncertainty"
    arg_type = Int64
    required = true
end

parsed_args = parse_args(ARGS, s; as_symbols=false)

generate_training_data(parsed_args)