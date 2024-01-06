using NeutrinoTelescopes
using PhotonPropagation
using PhysicsTools
using PreallocationTools
using Flux
using Random
using Sobol
using AbstractGPs
using SurrogatesAbstractGPs
using StaticArrays
using DataStructures
using Distributions
using Surrogates
using LinearAlgebra
using CairoMakie
using JLD2
using Optim
using Base.Iterators
import Base.GC: gc
using ProgressBars
using ParameterHandling
using StatsBase
using Polyhedra
using GLPK


abstract type OptimizationMetric end

abstract type FisherOptimizationMetric <: OptimizationMetric end

struct SingleEventTypeTotalResolution{FM <: FisherSurrogateModel} <: FisherOptimizationMetric
    events::Vector{Event}
    fisher_model::FM
end

struct SingleEventTypeTotalResolutionNoVertex{FM <: FisherSurrogateModel} <: FisherOptimizationMetric
    events::Vector{Event}
    fisher_model::FM
end


struct SingleEventTypeDOptimal{FM <: FisherSurrogateModel} <: FisherOptimizationMetric
    events::Vector{Event}
    fisher_model::FM
end

struct SingleEventTypeDNoVertex{FM <: FisherSurrogateModel} <: FisherOptimizationMetric
    events::Vector{Event}
    fisher_model::FM
end


struct MultiEventTypeTotalResolution{OM <: OptimizationMetric} <: OptimizationMetric
    metrics::Vector{OM}
end


mask_events_in_metric(fm::T, mask) where {T <: FisherOptimizationMetric} = T(fm.events[mask], fm.fisher_model)


function evaluate_optimization_metric(detector, m::SingleEventTypeTotalResolution)
    sigmasq, _, _ = run_fisher(m.events, detector, m.fisher_model)
    return sqrt(sum(sigmasq))
end

function evaluate_optimization_metric(detector, m::SingleEventTypeTotalResolutionNoVertex)

    _, fishers, _ = run_fisher(m.events, detector, m.fisher_model)
 
    fisher_no_pos = fishers[:, 1:3, 1:3]
    cov = mean(FisherSurrogate.invert_fishers(fisher_no_pos))
    
    met = sqrt(sum(diag(cov)))

    return met
end


function evaluate_optimization_metric(detector, m::SingleEventTypeDOptimal)
    _, _, cov = run_fisher(m.events, detector, m.fisher_model)
    cov = 0.5 * (cov + cov')
    met = det(cov)
    return met
end

function evaluate_optimization_metric(detector, m::SingleEventTypeDNoVertex)
    _, fishers, _ = run_fisher(m.events, detector, m.fisher_model)
 
    fisher_no_pos = fishers[:, 1:3, 1:3]
    cov = mean(FisherSurrogate.invert_fishers(fisher_no_pos))

    met = det(cov)
    return met
end



function evaluate_optimization_metric(xy, detector, m::OptimizationMetric)

    targets = get_detector_modules(detector)
    event_mask = get_events_in_range(m.events, targets, m.fisher_model)

    masked_m = mask_events_in_metric(m, event_mask)

    x, y = xy
    new_targets = new_line(x, y)
    new_det = add_line(detector, new_targets)
    met = evaluate_optimization_metric(new_det, masked_m)
    return met
end

function evaluate_optimization_metric(xy, detector, m::MultiEventTypeTotalResolution)
    return sum(evaluate_optimization_metric(xy, detector, am) for am in m.metrics)
end


function get_polyhedron(detector)
    posv = [convert(Vector{Float64}, t.shape.position) for t in get_detector_modules(detector)]
    v = vrep(posv)
    v = removevredundancy(v, GLPK.Optimizer)
    p = polyhedron(v)
    return p
end


function new_line(x, y)
    return make_detector_line(@SVector[Float32(x), Float32(y), 0f0], 20, 50, 1, DummyTarget)
end


function add_line(detector::LineDetector, targets_new_line)
    targets = detector.modules

    line_mapping = deepcopy(detector.line_mapping)
    new_line_id = length(keys(detector.line_mapping))+1
    
    line_mapping[new_line_id] = (length(targets)+1):(length(targets)+length(targets_new_line))


    new_det = LineDetector([targets; targets_new_line], detector.medium, line_mapping)
    return new_det
end

function make_track_injector(radius=400.)
    #cylinder = get_bounding_cylinder(detector)
    cylinder = Cylinder(SA[0., 0., -475.], 1100., radius)
    surf = CylinderSurface(cylinder)
    pdist = CategoricalSetDistribution(OrderedSet([PMuPlus, PMuMinus]), [0.5, 0.5])
    edist = Pareto(1, 1E4)
    ang_dist = LowerHalfSphere()
    length_dist = Dirac(1E4)
    time_dist = Dirac(0.0)
    inj = SurfaceInjector(surf, edist, pdist, ang_dist, length_dist, time_dist)
    return inj
end

function make_cascade_injector(radius=400.)
    #cylinder = get_bounding_cylinder(detector)
    cylinder = Cylinder(SA[0., 0., -475.], 1100., radius)
    pdist = CategoricalSetDistribution(OrderedSet([PEMinus, PEPlus]), [0.5, 0.5])
    edist = Pareto(1, 1E4)
    ang_dist = UniformAngularDistribution()
    length_dist = Dirac(0.0)
    time_dist = Dirac(0.0)
    inj = VolumeInjector(cylinder, edist, pdist, ang_dist, length_dist, time_dist)
    return inj
end


function profiled_pos(fisher::Matrix)
    IA = fisher[1:3, 1:3]
    IC = fisher[4:6, 1:3]
    ICprime = fisher[1:3, 4:6]
    IB = fisher[4:6, 4:6]
    profiled = IA - ICprime*inv(IB)*IC
    return profiled
end


function get_events_in_range(events, targets, model)
    events_range_mask = falses(length(events))
    for evix in eachindex(events)
        for t in targets
            if FisherSurrogate.is_in_range(first(events[evix][:particles]), t.shape.position[1:2], model)
                events_range_mask[evix] = true
                break
            end
        end
    end
    return events_range_mask
end

function run_fisher(
    events::AbstractVector{<:Event},
    detector,
    fisher_model::FisherSurrogateModel)

    targets = get_detector_modules(detector)
    event_mask = get_events_in_range(events, targets, fisher_model)
    valid_events = events[event_mask]
    covs, fishers = predict_cov(valid_events, detector , fisher_model)

    #mean_fishers = mean(fishers, dims=1)[1, :, :]
    cov = mean(covs)
    #cov = inv(0.5*(mean_fishers + mean_fishers'))
    
    return diag(cov), fishers, cov
end


#=
function run_fisher(
    events::AbstractVector{<:Event},
    detector::LineDetector,
    new_targets::AbstractVector{<:PhotonTarget},
    fisher_model::FisherSurrogateModel)
    
    targets = get_detector_modules(detector)
    event_mask = get_events_in_range(events, targets, fisher_model)
    valid_events = events[event_mask]
    if length(new_targets) > 0
        new_det = add_line(detector, new_targets)
    else
        new_det = detector
    end
    covs, fishers = predict_cov(valid_events, new_det , fisher_model)
    cov = mean(covs)
    return diag(cov), fishers, cov
end



function run_fisher(
    events::AbstractVector{<:Event},
    detector,
    fisher_model::FisherSurrogateModel)    
   return run_fisher(events, detector, PhotonTarget[], fisher_model)
end
=#



function get_surrogate_min(gp_surrogate, seed, lower_bound=-500, upper_bound=500)
    res = optimize(x -> gp_surrogate((x[1], x[2])), [lower_bound, lower_bound] , [upper_bound, upper_bound], seed[:])
    xmin = Optim.minimizer(res)
    fmin = Optim.minimum(res)

    return xmin, fmin
end

function plot_surrogate(gp_surrogate, detector, lower, upper)
    y=x = lower:10.:upper
    #z = [gp_surrogate((xi, yi)) for xi in x for yi in y]
    #=
    p =  first(event[:particles])
    x0 = .-p.direction * 600 .+ p.position
    x1 = p.direction * 600 .+ p.position
    =#
    fig = Figure()
    ax = Axis3(fig[1, 1], viewmode = :fit)
    #lines!(ax, hcat(x0, x1) )

    modules = get_detector_modules(detector)

    mod_pos = reduce(hcat, [m.shape.position for m in modules])
    contour!(ax, x, y, (x1,x2) -> gp_surrogate((x1,x2)), levels=10)
    scatter!(ax, mod_pos)

    xlims!(lower, upper)
    ylims!(lower, upper)
    zlims!(ax, -1000, 0)


    ax2 = Axis3(fig[1, 2],viewmode = :fit, 
        zlabel=L"log_{10}\left (\sqrt{Tr(cov)}\right )",  xticklabelsize=12, yticklabelsize=12, zticklabelsize=12)
    surface!(ax2, x, y, (x, y) -> gp_surrogate((x, y)))
    
    xmin, _ = get_surrogate_min(gp_surrogate, lower, upper)

    scatter!(ax, xmin[1], xmin[2], 0)
    fig
end

function plot_surrogate_anim(data, event, lower, upper)
    y=x = lower:2.:upper
   
    

    fig = Figure()
    ax = Axis3(fig[1, 1], viewmode=:fit)

    if !isnothing(event)
        p =  first(event[:particles])
        x0 = .-p.direction * 600 .+ p.position
        x1 = p.direction * 600 .+ p.position
        lines!(ax, hcat(x0, x1) )
    end
    gp_eval = Observable(rand(length(x), length(y)))
    mod_pos = Observable(zeros(1, 3))

    contour!(ax, x, y, gp_eval, levels=10)
    scatter!(ax, mod_pos)

    xlims!(lower, upper)
    ylims!(lower, upper)
    zlims!(ax, -1000, 0)
    
    ax2 = Axis3(fig[1, 2],viewmode = :fit, 
        zlabel=L"log_{10}\left (\sqrt{Tr(cov)}\right )",  xticklabelsize=12, yticklabelsize=12, zticklabelsize=12)
    surface!(ax2, x, y, gp_eval)

    ax3 = Axis(fig[2, 1], xlabel="Closest String Distance (m)")
    string_distances = Observable(Float64[])
    hist!(ax3, string_distances, bins=0:15:400)

    g_res_vol = fig[2, 2] = GridLayout()

    ax4 = Axis(g_res_vol[1, 1], xlabel="Iteration", ylabel="Resolution", yscale=log10)

    resolutions = Observable[]
    iteration_range = Observable(Float64[])
    labels = ["logE", "theta", "phi", "x", "y", "z"]
    for (i,label) in enumerate(labels)
        obs = Observable(Float64[])
        lines!(ax4, iteration_range, obs, label=label)
        push!(resolutions, obs)
    end

    ax5 = Axis(g_res_vol[2, 1], xlabel="Iteration", ylabel="Volume (km^3)")
    det_vol = Observable(Float64[])
    lines!(ax5, iteration_range, det_vol)
    
    
    fig[2, 3] = Legend(fig, ax4,  framevisible = false, labelsize=14)  

    xmin = Observable([0., 0.])
    #scatter!(ax, xmin[][1], xmin[][2], 0)

    cam3d!(ax.scene)
    update_cam!(ax.scene, cameracontrols(ax.scene), Vec3f(-1200, -1200, 900), Vec3f(0, 0, -500))

    lims = [[lower, upper], [lower, upper], [-1000., 0.]]
    m = maximum(abs(x[2] - x[1]) for x in lims)
    a = [m / abs(x[2] - x[1]) for x in lims]
    Makie.scale!(ax.scene, a...)
 
    xlims!(ax2, lower, upper)
    ylims!(ax2, lower, upper)

    record(fig, "time_animation.mp4", framerate = 15) do io
        for (iteration, d) in enumerate(data)
            modules = get_detector_modules(d[:detector])
            positions = reduce(hcat, [m.shape.position for m in modules])
            mod_pos[] = positions
            gp_surrogate = d[:surrogate]
            gp_eval[] = [gp_surrogate((x1,x2)) for x1 in x, x2 in y]
            xmin[] = d[:xmin]
            zmin, zmax = extrema(gp_eval[])
            zlims!(ax2, zmin, zmax)

            xy_positions = unique(positions[1:2,:], dims=2)

            distances = (norm.(eachcol(xy_positions) .- permutedims(eachcol(xy_positions))))

            distances[diagind(distances)] .= Inf64
            closest = minimum(distances, dims=2)[:]

            #pairwise = distances[triu!(trues(size(distances)), 1)]
            string_distances[] = closest
            reset_limits!(ax3)
            
            iteration_range.val = 1:iteration

            for (obs, res) in zip(resolutions, d[:fisher])
                obs[] = push!(obs[], res)
            end

            bpoly = get_polyhedron(d[:detector])
            det_vol[] = push!(det_vol[],  Polyhedra.volume(bpoly) / 1E9)

            reset_limits!(ax4)
            reset_limits!(ax5)
           
            for i in 1:25
                rotate_cam!(ax.scene, Vec3f(0, deg2rad(360/180), 0))
                recordframe!(io)
            end
        end
    end
end

function plot_surrogate_non_anim(data, event, lower, upper)
    y=x = lower:10.:upper
   
    fig = Figure()
    ax = Axis(fig[1, 1])

    if !isnothing(event)
        p =  first(event[:particles])
        x0 = .-p.direction * 600 .+ p.position
        x1 = p.direction * 600 .+ p.position
        lines!(ax, hcat(x0, x1) )
    end
    gp_eval = Observable(rand(length(x), length(y)))
    mod_pos = Observable(zeros(1, 2))

    contour!(ax, x, y, gp_eval, levels=10)
    scatter!(ax, mod_pos)

    xmin = Observable(zeros(1, 2))
    scatter!(ax, xmin)

    xlims!(lower, upper)
    ylims!(lower, upper)
    
    ax2 = Axis3(fig[1, 2],viewmode = :fit, 
        zlabel="Res. Improv.",  xticklabelsize=12, yticklabelsize=12, zticklabelsize=12)
    surface!(ax2, x, y, gp_eval)

    ax3 = Axis(fig[2, 1], xlabel="Closest String Distance (m)")
    string_distances = Observable(Float64[])
    hist!(ax3, string_distances, bins=0:10:300)

    g_res_vol = fig[2, 2] = GridLayout()

    ax4 = Axis(g_res_vol[1, 1], xlabel="Iteration", ylabel="Resolution", yscale=log10)

    resolutions = Observable[]
    resolutions_alt = Observable[]

    iteration_range = Observable(Float64[])
    labels = ["logE", "theta", "phi", "x", "y", "z"]
    for (i,label) in enumerate(labels)
        obs = Observable(Float64[])
        obs_alt = Observable(Float64[])
        lines!(ax4, iteration_range, obs, label=label)
        lines!(ax4, iteration_range, obs_alt, linestyle=:dash)
        push!(resolutions, obs)
        push!(resolutions_alt, obs_alt)
    end

    ax5 = Axis(g_res_vol[2, 1], xlabel="Iteration", ylabel="Volume (km^3)")
    det_vol = Observable(Float64[])
    lines!(ax5, iteration_range, det_vol)
    
    
    fig[2, 3] = Legend(fig, ax4,  framevisible = false, labelsize=14)  

   
 
    xlims!(ax2, lower, upper)
    ylims!(ax2, lower, upper)

    record(fig, "time_non_animation.mp4", framerate = 1) do io
        for (iteration, d) in enumerate(data)
            modules = get_detector_modules(d[:detector])
            positions = reduce(hcat, [m.shape.position for m in modules])
            gp_surrogate = d[:surrogate]
            gp_eval[] = [gp_surrogate((x1,x2)) for x1 in x, x2 in y]
            zmin, zmax = extrema(gp_eval[])
            zlims!(ax2, zmin, zmax)

            xy_positions = unique(positions[1:2,:], dims=2)
            mod_pos[] = xy_positions
            distances = (norm.(eachcol(xy_positions) .- permutedims(eachcol(xy_positions))))

            distances[diagind(distances)] .= Inf64
            closest = minimum(distances, dims=2)[:]

            #pairwise = distances[triu!(trues(size(distances)), 1)]
            string_distances[] = closest
            reset_limits!(ax3)
            
            iteration_range.val = 1:iteration

            for (obs, obs_alt, res, res_alt) in zip(resolutions, resolutions_alt, d[:fisher], d[:fisher_alt])
                obs[] = push!(obs[], sqrt.(res))
                obs_alt[] = push!(obs_alt[], sqrt.(res_alt))
            end

            bpoly = get_polyhedron(d[:detector])
            det_vol[] = push!(det_vol[],  Polyhedra.volume(bpoly) / 1E9)

            reset_limits!(ax4)
            reset_limits!(ax5)

            recordframe!(io)
            xmin[] = permutedims(d[:xmin])
            recordframe!(io)
            recordframe!(io)

        end
    end
end


function build_gp_prior(hyperparams)
    se_params = hyperparams.se
    se = se_params.σ^2 * with_lengthscale(SqExponentialKernel(), [se_params.ℓ, se_params.ℓ])
    
    #=mat_params = hyperparams.mat
    mat = mat_params.σ^2 * with_lengthscale(Matern52Kernel(), [mat_params.ℓ, mat_params.ℓ])
    kernel = se + mat
    =#
    kernel = se
    return GP(hyperparams.y_mean, kernel)
end


function build_finite_gp(θ, x)
    f = build_gp_prior(θ)
    return f(x, θ.noise_scale^2)
end

function make_loss(x, y)
    function loss(θ)
        fx = build_finite_gp(θ, x)
        lml = logpdf(fx, y)  # this computes the log marginal likelihood
        return -lml
    end
    return loss
end

function optimize_loss_gp(loss, θ_init; maxiter=1_000)
    
    options = Optim.Options(; iterations=maxiter, show_trace=false)

    θ_flat_init, unflatten = ParameterHandling.value_flatten(θ_init)
    loss_packed = loss ∘ unflatten
    result = optimize(loss_packed, θ_flat_init, LBFGS(alphaguess=Optim.LineSearches.InitialStatic(; scaled=true),
    linesearch=Optim.LineSearches.BackTracking(),), options; inplace=false)

    return unflatten(result.minimizer), result
end


function make_initial_guess_gp(x, y, randomize=false)

    if randomize
        signal_sigma = 10^rand(Uniform(-4, log10(2)))
        l =  10^rand(Uniform(0, 3))
        noise_scale = 10^rand(Uniform(-6, 0))
    else
        signal_sigma = std(y)
        x_mean = mean(x, dims=2)
        x_dist = norm.(eachcol(x .- x_mean))
        l = median(x_dist)
        noise_scale = signal_sigma/100
    end

    θ_init = (
        se = (
            σ = bounded(signal_sigma, 1E-4, 10),
            ℓ = bounded(l, 1., 1000.),
        ),
        #=
        mat = (
            σ = positive(1.),
            ℓ = positive(10.),
        ),
        =#
        noise_scale = positive(noise_scale),
        y_mean = mean(y)
    )
    return θ_init
end


function SurrogatesAbstractGPs.add_point!(g::AbstractGPSurrogate{<:ColVecs}, new_x::Tuple, new_y)
    new_x = ColVecs(reshape(collect(new_x), (2,1)))

    x_copy = vcat(g.x, new_x)
    y_copy = copy(g.y)
    push!(y_copy, new_y) 

    updated_posterior = posterior(g.gp(x_copy, g.Σy), y_copy)
    g.x, g.y, g.gp_posterior = x_copy, y_copy, updated_posterior
    nothing
end



function fit_surrogate(metric::OptimizationMetric, detector, lower, upper, previous_metric=nothing; n_samples=300)
    lower_bound = [lower, lower]
    upper_bound = [upper, upper]

    xys = Surrogates.sample(ceil(Int64, 0.8*n_samples), lower_bound, upper_bound, SobolSample())
    losses = evaluate_optimization_metric.(xys, Ref(detector), Ref(metric))

    sane = losses .> 0

    losses = losses[sane]
    xys = xys[sane]

    minfunc = xs -> log10(evaluate_optimization_metric(xs, detector, metric))

    xys_vec = reduce(hcat, collect.(xys))
    losses = log10.(losses ./ previous_metric)

    gp_loss_f = make_loss(ColVecs(xys_vec), losses)
    θ_init = make_initial_guess_gp(xys_vec, losses)
    θ_opt, res = optimize_loss_gp(gp_loss_f, θ_init)


    θ_min = [θ_opt]
    f_min = [Optim.minimum(res)]

    for i in 1:5
        θ_rnd = make_initial_guess_gp(xys_vec, losses, true)
        θ_opt, res = optimize_loss_gp(gp_loss_f, θ_rnd)
        push!(θ_min, θ_opt)
        push!(f_min, Optim.minimum(res))
    end

    minix = argmin(f_min)
    θ_opt = θ_min[minix]
    
    @show θ_opt

    gp_surrogate_opt = AbstractGPSurrogate(
        ColVecs(xys_vec),
        losses,
        gp=build_gp_prior(ParameterHandling.value(θ_opt)),
        Σy=ParameterHandling.value(θ_opt).noise_scale^2)
      
    opt_res = surrogate_optimize(minfunc, LCBS(), lower_bound, upper_bound, gp_surrogate_opt, SobolSample(); maxiters = 50, num_new_samples = n_samples)

    if isnothing(opt_res)
        amin = argmin(gp_surrogate_opt.y)
        opt_min = gp_surrogate_opt.x[amin]
    else
        opt_min = opt_res[1]
    end

    return gp_surrogate_opt, opt_min

end

function setup_fisher_sampler(type)
    model_path = joinpath(ENV["ECAPSTOR"], "snakemake/time_surrogate")
    model = PhotonSurrogate(joinpath(model_path, "$(type)/amplitude_1_FNL.bson"), joinpath(model_path, "$(type)/time_uncert_2.5_1_FNL.bson"))
    model = gpu(model_tracks)
    
    n_lines_max = 50

    input_buffer = create_input_buffer(20*16*n_lines_max, 1)
    output_buffer = create_output_buffer(20*16*n_lines_max, 100)
    diff_cache = FixedSizeDiffCache(input_buffer, 6)
    hit_generator = SurrogateModelHitGenerator(model, 200.0, input_buffer, output_buffer)

    return hit_generator, diff_cache
end

    
function setup_fisher_surrogate(type)
    model_fname = joinpath(ENV["ECAPSTOR"], "snakemake/fisher_surrogates/fisher_surrogate_$type.bson")
    if occursin("per_string", type)
        fisher_surrogate = gpu(FisherSurrogateModelPerLine(model_fname))        
    else
        fisher_surrogate = gpu(FisherSurrogateModelPerModule(model_fname))
    end
    #input_buffer = zeros(8, 20*70*500)
    #diff_cache = FixedSizeDiffCache(input_buffer, 10)
    return fisher_surrogate, nothing
end


function run(type, n_lines_max; extent=1000, n_events=250, same_events=true, seed=42, geo_seed=nothing, inj_radius=400.)
    medium = make_cascadia_medium_properties(0.95f0)
    if isnothing(geo_seed)
        targets_line = make_detector_line(@SVector[0f0, 0f0, 0f0], 20, 50, 1, DummyTarget)
        detector = LineDetector(targets_line, medium, [1 => 1:20])
    else
        detector = geo_seed
    end
    
    rng = Random.MersenneTwister(seed)

    inj_f = occursin("extended", type) ? make_cascade_injector : make_track_injector
    inj_f_alt = occursin("extended", type) ? make_track_injector : make_cascade_injector

    fisher_model, diff_cache = setup_fisher_surrogate(type)
    if occursin("per_string", type)        
        alt_type = occursin("extended", type)  ? "per_string_lightsabre" : "per_string_extended"
    else
        alt_type = occursin("extended", type)  ? "lightsabre" : "extended"
    end
    
    fisher_model_alt, diff_cache_alt = setup_fisher_surrogate(alt_type)
    
    data = []
    current_det = detector
    current_inj = inj_f(inj_radius)
    current_inj_alt = inj_f_alt(inj_radius)

    if same_events
        events = [rand(rng, current_inj) for _ in 1:n_events]
        events_alt = [rand(rng, current_inj_alt) for _ in 1:n_events]
    else
        error("Not implemented")
    end

    metric = SingleEventTypeTotalResolution(events, fisher_model)
    #alt_metric = SingleEventTypeDOptimal(events_alt, fisher_model_alt)

    previous_fisher = nothing

    for i in ProgressBar(1:(n_lines_max-1))
        current_resolution, _, _ = run_fisher(events, current_det, fisher_model)
        current_resolution_alt, _, _ = run_fisher(events_alt, current_det, fisher_model_alt)

        current_metric = evaluate_optimization_metric(current_det, metric)

        sg_res = fit_surrogate(metric, current_det, -extent/2., extent/2., current_metric; n_samples=300)
        
        if length(sg_res) == 3
            return sg_res
        else
            gp_surrogate, opt_min = sg_res
        end
        xmin, fmin = get_surrogate_min(gp_surrogate, opt_min, -extent/2, extent/2)
        push!(data, (detector=current_det, surrogate=gp_surrogate, xmin=xmin, fisher=current_resolution, fisher_alt=current_resolution_alt, target_pred=fmin))
        new_targets = new_line(xmin[1], xmin[2])
        current_det = add_line(current_det, new_targets)
        gc()
        if (i+1) % 5 == 0
            plot_surrogate_non_anim(data, nothing, -extent/2-50, extent/2+50)
            #plot_surrogate_anim(data, nothing, -250, 250)
        end
    end
    return data
end

function main()
    medium = make_cascadia_medium_properties(0.95f0)
    sidelen = 100f0
    height::Float32 = sqrt(3)/2 * sidelen
    targets_triang = [
        make_detector_line(@SVector[0f0, 0f0, 0f0], 20, 50, 1, DummyTarget),
        make_detector_line(@SVector[-sidelen/2, -height, 0f0], 20, 50, 21, DummyTarget),
        make_detector_line(@SVector[sidelen/2, -height, 0f0], 20, 50, 41, DummyTarget),
    ]

    targets_line = make_detector_line(@SVector[0f0, 0f0, 0f0], 20, 50, 1, DummyTarget)

    detector = LineDetector(targets_triang, medium)
    data = run("per_string_lightsabre", 40, n_events=20000, extent=1000, geo_seed=detector, inj_radius=400.)
end

main()
