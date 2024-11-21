using Random
using StaticArrays
using NeutrinoTelescopes
using PhotonPropagation
using PhotonSurrogateModel
using Rotations
using LinearAlgebra
using PhysicsTools
using ForwardDiff
using DataStructures
using Distributions
using Optimisers
using ProgressLogging
using PolyesterForwardDiff: threaded_gradient!
using Logging: global_logger
using TerminalLoggers
using ParameterSchedulers
using Base.Iterators: product
using DataFrames
using JLD2
using ArgParse
using Dates
using ThreadPinning
using TensorBoardLogger
using Logging
using ImageMagick

pinthreads(:affinitymask)

global_logger(TerminalLogger(right_justify=120))

using CairoMakie
import TensorBoardLogger.PNGImage: PngImage
function Base.convert(t::Type{PngImage}, plot::CairoMakie.Figure)
    pb = PipeBuffer()
    show(pb, MIME("image/png"), plot)
    return PngImage(pb)
end

preprocess(name, plot::CairoMakie.Figure, data) = preprocess(name, convert(PngImage, plot), data)
preprocess(name, plots::AbstractArray{<:CairoMakie.Figure}, data) = begin
    for (i, plot)=enumerate(plots)
        preprocess(name*"/$i", plot, data)
    end
    return data
end



function optimize_layout(
    layout,
    metric,
    surrogate,
    injector;
    schedule,
    iterations=50,
    history=nothing,
    state=nothing,
    chunk_size=10,
    batch_size=400,
    start_it=1,
    constraints=nothing,
    fix_angle=true,
    logger=nothing,
    checkpoint=nothing)

    flat, re = destructure(layout)

    if isnothing(state)
        d = OptimiserChain(Optimisers.AdamW(1))
        #d = OptimiserChain(Optimisers.Momentum(1, 0.9))
        state = Optimisers.setup(d, flat)
    end

    if fix_angle
        phi_angle_first_line = atan(flat[2], flat[1])
    end

    diff_res = DiffResults.GradientResult(flat)

    function loss_fn(layout, batch)
        if !isnothing(constraints)
            penalty = sum(c(layout) for c in constraints; init=0.)
        end
        return metric(surrogate, layout, batch) + penalty
    end



    
    @progress for iteration in start_it:(iterations+start_it-1)

        Optimisers.adjust!(state, schedule(iteration))
    
        batch = [rand(injector) for _ in  1:batch_size]
        
        cur_loss = threaded_gradient!(DiffResults.gradient(diff_res), flat, ForwardDiff.Chunk(chunk_size), Val{true}()) do x
            re_layout = re(x)
            return loss_fn(re_layout, batch)
        end


        ∇flat = DiffResults.gradient(diff_res)

        hist_tup = (iteration=iteration, layout= re(copy(flat)), loss=cur_loss, grad_norm=norm(∇flat))

        if typeof(history) <: Channel
            put!(history, hist_tup)
        else
            push!(history, hist_tup)
        end

        if !isnothing(logger)

            module_positions = collect(hist_tup.layout)
            positions_xy = Point2f.(getindex.(module_positions, Ref([1, 2])))

            fig = Figure()
            ax = Axis(fig[1, 1], xlabel="x (m)", ylabel="y (m)")
            scatter!(ax, positions_xy)
            arc!(ax, Point2f(0), injector.volume.radius, -π, π)
            xlims!(ax, (-injector.volume.radius*1.2, injector.volume.radius*1.2))
            ylims!(ax, (-injector.volume.radius*1.2, injector.volume.radius*1.2))


            log_value(logger, "train/loss", cur_loss, step=iteration)
            log_image(logger, "train/layout", convert(PngImage, fig), step=iteration)

        end

        state, flat = Optimisers.update!(state, flat, ∇flat)

        if fix_angle
            radius_first_line = sqrt(flat[1]^2 + flat[2]^2)
            # Fix angle to first line
            flat[1] = radius_first_line * cos(phi_angle_first_line)
            flat[2] = radius_first_line * sin(phi_angle_first_line)
        end

        if !isnothing(checkpoint) && (iteration % 50 == 0)
            jldsave(checkpoint, layout_opt=re(flat), history=DataFrame(history), state=state, inj=injector, surrogate=surrogate, metric=metric)
        end

    end
    

    return re(flat), history, state
end





struct ExpLog10Uniform <: ContinuousUnivariateDistribution
    a::Float64
    b::Float64
end

Base.rand(rng::AbstractRNG, d::ExpLog10Uniform) = 10 .^rand(rng, Uniform(log10(d.a), log10(d.b)))

function make_injector(radius; spectral_index=1.0, e_min=1E2, e_max=1E7)
    cylinder = NeutrinoTelescopes.Cylinder(SA[0., 0., 0.], 1100., radius)
    pdist = CategoricalSetDistribution(OrderedSet([PEMinus, PEPlus]), [0.5, 0.5])
    if spectral_index == 1
        edist = ExpLog10Uniform(e_min, e_max)
    else
        edist = Pareto(spectral_index-1, e_min,)
    end
    ang_dist = UniformAngularDistribution()
    length_dist = Dirac(0.0)
    time_dist = Dirac(0.0)
    inj = VolumeInjector(cylinder, edist, pdist, ang_dist, length_dist, time_dist)

    return inj
end

function make_line_layout(::Type{<:StringLayoutCart} ,n_lines, radius, rng=Random.GLOBAL_RNG)
    
    rand_r = sqrt.(rand(rng, n_lines) .* radius^2)
    rand_phi = rand(rng, n_lines) .* 2 * pi

    rand_x = rand_r .* cos.(rand_phi)
    rand_y = rand_r .* sin.(rand_phi)

    scaling_factor = 100.

    modules = StringLayoutCart(
        [[rx, ry] ./ scaling_factor for (rx, ry) in zip(rand_x, rand_y)],
        range(-475., 475, 20),
        sph_to_cart.(eachcol(make_pom_pmt_coordinates(Float64))),
        [0., 0.],
        scaling_factor)
    return modules
end

#=
function make_line_layout(::Type{<:StringLayoutPolar} ,n_lines, radius, rng=Random.GLOBAL_RNG)
    
    rand_r = sqrt.(rand(rng, n_lines) .* radius^2)
    rand_phi = rand(rng, n_lines) .* 2 * pi


    modules = StringLayoutPolar(
        [[rr, rphi] for (rr, rphi) in zip(rand_r, rand_phi)],
        range(-475., 475, 20),
        sph_to_cart.(eachcol(make_pom_pmt_coordinates(Float64))))
    return modules
end
=#

function CosAnnealDecay(range, offset, period, decay_rate)
    parameters = (Step(range, decay_rate, period), offset, period, true)
    ComposedSchedule(CosAnneal(range, offset, period, true), parameters)
end


function (@main)(ARGS)

    s = ArgParseSettings()

    @add_arg_table s begin
        "--output"
        help = "Output filename"
        arg_type = String
        required = true
        "--radius"
        help = "Radius of the simulation volume"
        arg_type = Float64
        required = true
        "--layout_initial_radius"
        help = "Radius of the initial layout"
        arg_type = Float64
        required = true
        "--seed"
        help = "Seed for the random number generator"
        arg_type = Int
        required = true
        "--lr"
        help = "Initial learning rate"
        arg_type = Float64
        required = true
        "--lr_min"
        help = "Minimum learning rate"
        arg_type = Float64
        required = true
        "--iterations"
        help = "Number of iterations"
        arg_type = Int
        required = true
        "--batch_size"
        help = "Batch size"
        arg_type = Int
        required = true
        "--n_lines"
        help = "Number of lines"
        arg_type = Int
        required = true
        "--gamma"
        help = "Flux spectral index"
        arg_type = Float64
        required = true
        "--chunk_size"
        help = "Chunk size"
        arg_type = Int
        required = true
    end

    args = parse_args(ARGS, s)

    @show args
    radius = args["radius"]
    seed = args["seed"]
    lr = args["lr"]
    lr_min = args["lr_min"]
    iterations = args["iterations"]
    batch_size = args["batch_size"]
    outfile = args["output"]
    n_lines = args["n_lines"]
    gamma = args["gamma"]
    layout_initial_radius = args["layout_initial_radius"]
    
    e_min = 1E3
    e_max = 1E7

    chunk_size = args["chunk_size"]

    inj = make_injector(radius, spectral_index=1, e_min=e_min, e_max=e_max)

    rng = Random.MersenneTwister(seed)
    layout = make_line_layout(StringLayoutCart, n_lines, layout_initial_radius, rng)
    surrogate = SRSurrogateModel()

    # Find normalization factor
    metric = AngResDetEff(batch_size, get_volume(inj.volume)/1E9, 3, 2, -gamma, 1, (e_min, e_max))
    metric_val = metric(surrogate, layout, [rand(inj) for _ in 1:batch_size])
    flux_norm = 1/metric_val 


    metric = AngResDetEff(batch_size, get_volume(inj.volume)/1E9, 3, 2, -gamma, flux_norm, (e_min, e_max))
    @show metric(surrogate, layout, [rand(inj) for _ in 1:batch_size])


    #sched = ParameterSchedulers.Constant(0.07)
    sched = ParameterSchedulers.Constant(lr)

    history = []
    constraints = [SimulationBoundaryConstraint(radius*0.9, 0.001)]


    logger = TBLogger("/home/wecapstor3/capn/capn100h/tensorboard/det_opt/run")
    metric_names = ["train/loss"]
    config = Dict("lr" => lr, "lr_min" => lr_min, "iterations" => iterations, "batch_size" => batch_size, "n_lines" => n_lines, "gamma" => gamma, "radius" => radius, "layout_initial_radius" => layout_initial_radius)
    write_hparams!(logger, config, metric_names)
    
    tstart = now()
    layout_opt, history, state = optimize_layout(
        layout, metric, surrogate, inj, schedule = sched, iterations=iterations, chunk_size=chunk_size, batch_size=batch_size, history=history,
        constraints=constraints,
        fix_angle=false,
        logger=logger,
        checkpoint=outfile*".CHK")
    tend = now()
    @show tend - tstart
    
    history = DataFrame(history)
    JLD2.@save outfile layout_opt history state inj surrogate metric
end

