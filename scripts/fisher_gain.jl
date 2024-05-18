using NeutrinoTelescopes
using PhotonPropagation
using PhysicsTools
using PhotonSurrogateModel
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
using Hexagons
using CairoMakie.GeometryBasics
using DataFrames
using ForwardDiff
using ArgParse


# superficially similar to StatsBase.Histogram API
struct HexHistogram{T <: Real,S <: Real}
    xc::Vector{T}
    yc::Vector{T}
    weights::Vector{S}
    xsize::T
    ysize::T
    isdensity::Bool
end

function fit(::Type{HexHistogram},x::AbstractVector,y::AbstractVector,
             bins::Union{NTuple{1,Int},NTuple{2,Int},Int};
             xyweights=nothing,
             density::Bool=false)
    if length(bins) == 2
        xbins, ybins = bins
    else
        xbins, ybins = (bins...,bins...)
    end
    xmin, xmax = extrema(x)
    ymin, ymax = extrema(y)
    xspan, yspan = xmax - xmin, ymax - ymin
    xsize, ysize = xspan / xbins, yspan / ybins
    fit(HexHistogram,x,y,xsize,ysize,boundingbox=[xmin,xmax,ymin,ymax],
        density=density,xyweights=xyweights)
end

function xy2counts_(x::AbstractArray,y::AbstractArray,
                    xsize::Real,ysize::Real,x0,y0)
    counts = Dict{(Tuple{Int, Int}), Int}()
    @inbounds for i in eachindex(x)
        h = convert(HexagonOffsetOddR,
                    cube_round(x[i] - x0,y[i] - y0,xsize, ysize))
        idx = (h.q, h.r)
        counts[idx] = 1 + get(counts,idx,0)
    end
    counts
end
function xy2counts_(x::AbstractArray,y::AbstractArray,wv::AbstractVector{T},
                    xsize::Real,ysize::Real,x0,y0) where{T}
    counts = Dict{(Tuple{Int, Int}), T}()
    zc = zero(T)
    @inbounds for i in eachindex(x)
        h = convert(HexagonOffsetOddR,
                    cube_round(x[i] - x0,y[i] - y0,xsize, ysize))
        idx = (h.q, h.r)
        counts[idx] = wv[i] + get(counts,idx,zc)
    end
    counts
end
function counts2xy_(counts::Dict{S,T},xsize, ysize, x0, y0) where {S,T}
    nhex = length(counts)
    xh = zeros(nhex)
    yh = zeros(nhex)
    vh = zeros(T,nhex)
    k = 0
    for (idx, cnt) in counts
        k += 1
        xx,yy = Hexagons.center(HexagonOffsetOddR(idx[1], idx[2]),
                                xsize, ysize, x0, y0)
        xh[k] = xx
        yh[k] = yy
        vh[k] = cnt
    end
    xh,yh,vh
end
function fit(::Type{HexHistogram},x::AbstractVector,y::AbstractVector,
             xsize, ysize; boundingbox=[], density::Bool=false,
             xyweights::Union{Nothing,AbstractWeights}=nothing)
    (length(x) == length(y)) || throw(
        ArgumentError("data vectors must be commensurate"))
    (xyweights == nothing ) || (length(xyweights) == length(x)) || throw(
        ArgumentError("data and weight vectors must be commensurate"))

    if isempty(boundingbox)
        xmin, xmax = extrema(x)
        ymin, ymax = extrema(y)
    else
        xmin, xmax, ymin, ymax = (boundingbox...,)
    end
    xspan, yspan = xmax - xmin, ymax - ymin
    x0, y0 = xmin - xspan / 2,ymin - yspan / 2
    if xyweights == nothing
        counts = xy2counts_(x,y,xsize,ysize,x0,y0)
    else
        counts = xy2counts_(x,y,xyweights.values,xsize,ysize,x0,y0)
    end
    xh,yh,vh = counts2xy_(counts,xsize, ysize, x0, y0)
    if density
        binarea = sqrt(27)*xsize*ysize/2
        vh = 1/(sum(vh)*binarea) * vh
    end
    HexHistogram(xh,yh,vh,xsize,ysize,density)
end

function plot_resolution_hex(events, resolution, detector)

    function get_event_position(event::Event)
        return first(event[:particles]).position
    end
    positions = reduce(hcat, get_event_position.(events))

    line_xy = reduce(hcat, [first(l).shape.position[1:2] for l in get_detector_lines(detector)])

    w = Weights(resolution)

    hexhist_cnt = fit(HexHistogram, positions[1, :], positions[2, :], 30)
    hexhist = fit(HexHistogram, positions[1, :], positions[2, :], 30, xyweights = w)
    hexmarker = Polygon(Point2f[(cos(a), sin(a)) for a in range(pi / 6, 13pi / 6; length=7)[1:6]])

    fig = Figure()
    ax = Axis(fig[1, 1])
    s = scatter!(ax, 
        hexhist.xc,
        hexhist.yc,
        color=hexhist.weights ./ hexhist_cnt.weights,
        marker=hexmarker,
        markersize=(hexhist.xsize, hexhist.ysize),
        markerspace=:data,
        colorscale=log10,
        colorrange=(10, 300))

    scatter!(ax, line_xy, color=:black)
    Colorbar(fig[1, 2], s)
    fig
end



struct FisherCalculator{FM <: FisherSurrogateModel}
    events::Vector{Event}
    fisher_model::FM
end

mask_events_in_calc(fc::FisherCalculator, mask) = FisherCalculator(fc.events[mask], fc.fisher_model)

function calc_fisher(
    calc::FisherCalculator,    
    detector::Detector;
    abs_scale,
    sca_scale)

    events = calc.events
    fisher_model = calc.fisher_model

    targets = get_detector_modules(detector)
    event_mask = get_events_in_range(events, targets, fisher_model)
    valid_events = events[event_mask]
    lines = get_detector_lines(detector)

    fishers = predict_fisher(valid_events, lines, fisher_model, abs_scale=abs_scale, sca_scale=sca_scale)
    return fishers, event_mask
end

function calc_fisher(
    calc::FisherCalculator,    
    xy::NTuple{2, <:Real};
    abs_scale,
    sca_scale)

    events = calc.events
    fisher_model = calc.fisher_model

    x, y = xy
    targets = new_line(x, y)
    event_mask = get_events_in_range(events, targets, fisher_model)
    valid_events = events[event_mask]
    fishers = predict_fisher(valid_events, [targets], fisher_model, abs_scale=abs_scale, sca_scale=sca_scale)
    return fishers, event_mask
end




abstract type OptimizationMetric end

abstract type FisherOptimizationMetric <: OptimizationMetric end


struct SingleEventTypeResolution <: FisherOptimizationMetric end
struct SingleEventTypeTotalResolution <: FisherOptimizationMetric end
struct SingleEventTypeTotalRelativeResolution <: FisherOptimizationMetric end
struct SingleEventTypePerEventRelativeResolution <: FisherOptimizationMetric end
struct SingleEventTypeMeanDeviation <: FisherOptimizationMetric end
struct SingleEventTypePerEventMeanDeviation <: FisherOptimizationMetric end
struct SingleEventTypePerEventMeanAngularError <: FisherOptimizationMetric end
struct SingleEventTypeTotalResolutionNoVertex <:  FisherOptimizationMetric end
struct SingleEventTypeDOptimal <: FisherOptimizationMetric end
struct SingleEventTypeDNoVertex <: FisherOptimizationMetric end

struct MultiEventTypeTotalResolution{OM <: OptimizationMetric} <: OptimizationMetric
    metrics::Vector{OM}
end

function evaluate_optimization_metric(::SingleEventTypeTotalResolution, fishers, events)
    covs = invert_fishers(fishers)
    valid = isposdef.(covs)
    mean_cov = @views mean(covs[valid])
    sigmasq = diag(mean_cov)
    return sqrt(sum(sigmasq))
end

function get_event_props(event::Event)
    p = first(event[:particles])
    dir_sph = cart_to_sph(p.direction)
    shift_to_closest_approach(p, [0., 0., 0.])
    return [log10(p.energy), dir_sph..., p.position...]
end


function get_positional_uncertainty(pos, cov)
    d = MvNormal(pos, cov)
    smpl = rand(d, 100)
    dpos = norm.(Ref(pos) .- eachcol(smpl))
    return mean(dpos)
end

function get_directional_uncertainty(dir_sph, cov)
    dir_cart = sph_to_cart(dir_sph)

    dist = MvNormal(dir_sph, cov)
    rdirs = rand(dist, 100)

    dangles = rad2deg.(acos.(dot.(sph_to_cart.(rdirs[1, :], rdirs[2, :]), Ref(dir_cart))))
    return mean(dangles)
end

function evaluate_optimization_metric(::SingleEventTypePerEventMeanDeviation, fishers, events)
    covs = invert_fishers(fishers)
    valid = isposdef.(covs)
    results = zeros(length(events))

    for (i, (c, e)) in enumerate(zip(covs, events))
        if !valid[i]
            results[i] = NaN
            continue
        end
        event_props = get_event_props(e)
        mean_pos_dev = get_positional_uncertainty(event_props[4:6], c[4:6, 4:6])
        mean_ang_dev = get_directional_uncertainty(event_props[2:3], c[2:3, 2:3])
        mean_energy_dev = sqrt(c[1, 1])
        results[i] = mean_pos_dev + mean_ang_dev + mean_energy_dev
    end

    return results
end

function evaluate_optimization_metric(::SingleEventTypePerEventMeanAngularError, fishers, events)
    covs = invert_fishers(fishers)
    valid = isposdef.(covs)
    results = zeros(length(events))

    for (i, (c, e)) in enumerate(zip(covs, events))
        if !valid[i]
            results[i] = NaN
            continue
        end
        event_props = get_event_props(e)
        mean_ang_dev = get_directional_uncertainty(event_props[2:3], c[2:3, 2:3])
        results[i] = mean_ang_dev
    end

    return results
end


function evaluate_optimization_metric(::SingleEventTypeMeanDeviation, fishers, events)
    m = SingleEventTypePerEventMeanDeviation()

    results = evaluate_optimization_metric(m, fishers, events)
    masked_results = filter!(!isnan, results) 

    return mean(masked_results)
end



function evaluate_optimization_metric(::SingleEventTypeTotalRelativeResolution, fishers, events)
    covs = invert_fishers(fishers)
    valid = isposdef.(covs)

    results = zeros(sum(valid))
    for (i, (c, e)) in enumerate(zip(covs[valid], events[valid]))
        event_props = get_event_props(e)
        rel_sigma = diag(c) ./ event_props .^2
        results[i] = sqrt(sum(rel_sigma))
    end

    return mean(results)
end

function evaluate_optimization_metric(::SingleEventTypePerEventRelativeResolution, fishers, events)
    covs = invert_fishers(fishers)
    valid = isposdef.(covs)
    results = zeros(sum(valid))
    for (i, (c, e)) in enumerate(zip(covs[valid], events[valid]))
        event_props = get_event_props(e)
        rel_sigma = diag(c) ./ event_props .^2
        results[i] = sqrt(sum(rel_sigma))
    end

    return results
end


function evaluate_optimization_metric(::SingleEventTypeResolution, fishers, events)
    covs = invert_fishers(fishers)
    valid = isposdef.(covs)
    mean_cov = mean(covs[valid])
    sigmasq = diag(mean_cov)
    return sqrt.(sigmasq)
end



function evaluate_optimization_metric(::SingleEventTypeTotalResolutionNoVertex, fishers, events)
    fisher_no_pos = fishers[:, 1:3, 1:3]
    cov = mean(invert_fishers(fisher_no_pos))
    met = sqrt(sum(diag(cov)))
    return met
end

function evaluate_optimization_metric(::SingleEventTypeDOptimal, fishers, events)
    cov = mean(invert_fishers(fishers))
    cov = 0.5 * (cov + cov')
    met = det(cov)
    return met
end

function evaluate_optimization_metric(::SingleEventTypeDNoVertex, fishers, events)
    fisher_no_pos = fishers[:, 1:3, 1:3]
    cov = mean(invert_fishers(fisher_no_pos))
    cov = 0.5 * (cov + cov')
    met = det(cov)
    return met
end


function evaluate_optimization_metric(xy, detector, m::OptimizationMetric, events; abs_scale, sca_scale)

    targets = get_detector_modules(detector)
    event_mask = get_events_in_range(m.events, targets, m.fisher_model)

    masked_m = mask_events_in_metric(m, event_mask)

    x, y = xy
    new_targets = new_line(x, y)
    new_det = add_line(detector, new_targets)
    met = evaluate_optimization_metric(new_det, masked_m, abs_scale=abs_scale, sca_scale=sca_scale)
    return met
end

function evaluate_optimization_metric(xy, detector, m::MultiEventTypeTotalResolution, events; abs_scale, sca_scale)
    return sum(evaluate_optimization_metric(xy, detector, m, events, abs_scale=abs_scale, sca_scale=sca_scale) for am in m.metrics)
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
    cylinder = NeutrinoTelescopes.Cylinder(SA[0., 0., -475.], 1100., radius)
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
    cylinder = NeutrinoTelescopes.Cylinder(SA[0., 0., -475.], 1100., radius)
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
            @views if FisherSurrogate.is_in_range(first(events[evix][:particles]), t.shape.position[1:2], model)
                events_range_mask[evix] = true
                break
            end
        end
    end
    return events_range_mask
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



function get_surrogate_min(gp_surrogate, lower_bound, upper_bound, seed)
    res = optimize(x -> first(gp_surrogate((x[1], x[2]))), [lower_bound, lower_bound] , [upper_bound, upper_bound], seed[:])
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
    
    #xmin, _ = get_surrogate_min(gp_surrogate, lower, upper, (0, 0))

    #scatter!(ax, xmin[1], xmin[2], 0)
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

function plot_surrogate_non_anim(data, event, lower, upper, fname)
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
    med_spacing = Observable([0.0])
    vlines!(ax3, med_spacing, color=:black)
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
    
    
    #fig[2, 3] = Legend(fig, ax4,  framevisible = false, labelsize=14)  
 
    xlims!(ax2, lower, upper)
    ylims!(ax2, lower, upper)

    record(fig, fname, framerate = 1) do io
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
            med_spacing[] = [median(closest)]
            reset_limits!(ax3)
            
            iteration_range.val = 1:iteration

            for (obs, obs_alt, res, res_alt) in zip(resolutions, resolutions_alt, d[:fisher], d[:fisher_alt])
                obs[] = push!(obs[], sqrt(res))

                if res_alt < 0
                    res_alt = 1E2
                end
                obs_alt[] = push!(obs_alt[], sqrt(res_alt))
            end

            bpoly = get_polyhedron(d[:detector])
            det_vol[] = push!(det_vol[],  Polyhedra.volume(bpoly) / 1E9)

            res_range = extrema(resolutions[1][]) 

            ylims!(ax4, res_range[1]*0.95, res_range[2]*1.05)
            #reset_limits!(ax4)
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

function optimize_loss_gp(loss, θ_init; maxiter=100)
    
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
        y_mean = fixed(0.)# mean(y)
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

    
function setup_fisher_surrogate(type; max_particles=500, max_targets=70*20)
    model_fname = joinpath(ENV["ECAPSTOR"], "snakemake/fisher_surrogates/fisher_surrogate_$type.bson")
    if occursin("per_string", type)
        fisher_surrogate = gpu(FisherSurrogateModelPerLine(model_fname, max_particles, max_targets))        
    else
        fisher_surrogate = gpu(FisherSurrogateModelPerModule(model_fname, max_particles, max_targets))
    end
    #input_buffer = zeros(8, 20*70*500)
    #diff_cache = FixedSizeDiffCache(input_buffer, 10)
    return fisher_surrogate, nothing
end

function precalculate_fisher_proposals(calc, lower_bound, upper_bound, n_samples; abs_scale, sca_scale)
    xys = Surrogates.sample(ceil(Int64, n_samples), lower_bound, upper_bound, SobolSample())

    fisher_line_props = []
    fisher_line_props_masks = []
    for xy in xys
        fisher_line_prop, event_mask = calc_fisher(calc, xy, abs_scale=abs_scale, sca_scale=sca_scale)
        push!(fisher_line_props, fisher_line_prop)
        push!(fisher_line_props_masks, event_mask)
    end

    return fisher_line_props, fisher_line_props_masks, xys
end

function combine_proposals_with_current_geo(calc, proposals, proposal_event_masks, current_fishers, current_fishers_event_masks)
    
    ix_mask_cur = zeros(Int64, length(calc.events))
    ix_mask_cur[current_fishers_event_masks] .= 1:size(current_fishers, 1)

    ix_mask_prop = zeros(Int64, length(calc.events))

    all_fishers = Vector{Vector{Matrix{Float64}}}(undef, 0)
    valid_mask = zeros(Bool, length(proposals))

    for (prop_ix, (fp, fpm)) in enumerate(zip(proposals, proposal_event_masks))
        ix_mask_prop .= 0
        ix_mask_prop[fpm] .= 1:size(fp, 1)
        fishers_combined = Matrix{Float64}[]
        sizehint!(fishers_combined, length(calc.events))
        for evt_ix in eachindex(calc.events)

            if !current_fishers_event_masks[evt_ix]
                continue
            end

            ix_cur = ix_mask_cur[evt_ix]

            if !fpm[evt_ix]
                push!(fishers_combined, @view current_fishers[ix_cur, :, :])
                continue
            end

            ix_prop = ix_mask_prop[evt_ix]
           

            # this event is in range of the proposed line and current geometry
            fm = fp[ix_prop, :, :] .+ current_fishers[ix_cur, :, :]
            #=
            elseif fpm[evt_ix]
                fm = fp[ix_prop, :, :]
            else
                fm = fishers_current[ix_cur, :, :]
            =#
            push!(fishers_combined, fm)
        end
        if length(fishers_combined) > 0
            push!(all_fishers, fishers_combined)
            valid_mask[prop_ix] = true
        end
    end

    return all_fishers, valid_mask

end

function run(type, n_lines_max; vid_fname, extent=1000, n_events=250, same_events=true, seed=42, geo_seed=nothing, inj_radius=400., n_proposals=1000)
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

    metric = SingleEventTypeMeanDeviation()
    #alt_metric = SingleEventTypeDOptimal(events_alt, fisher_model_alt)

    resolution_metric = metric
    calc = FisherCalculator(events, fisher_model)
    calc_alt = FisherCalculator(events_alt, fisher_model_alt)

    abs_scale = 1.
    sca_scale = 1.

    lower = -extent/2
    upper = extent/2
    lower_bound = [lower, lower]
    upper_bound = [upper, upper]

    fisher_line_props, fisher_line_props_masks, xys = precalculate_fisher_proposals(calc, lower_bound, upper_bound, n_proposals, abs_scale=abs_scale, sca_scale=sca_scale)

    for i in ProgressBar(1:(n_lines_max-1))
        fishers_current, mask_current = calc_fisher(calc, current_det, abs_scale=abs_scale, sca_scale=sca_scale)
        fishers_current_alt, mask_current_alt = calc_fisher(calc_alt, current_det, abs_scale=abs_scale, sca_scale=sca_scale)

        current_metric = evaluate_optimization_metric(metric, fishers_current, events[mask_current])

        all_fishers, valid_mask = combine_proposals_with_current_geo(calc, fisher_line_props, fisher_line_props_masks, fishers_current, mask_current)
        metrics_per_pos = evaluate_optimization_metric.(Ref(metric), all_fishers, Ref(events[mask_current]))

        losses = log10.(metrics_per_pos ./ current_metric)
        valid_xys = xys[valid_mask]

        valid_losses = losses .< 0
        losses = losses[valid_losses]
        valid_xys = valid_xys[valid_losses]

        xys_vec = reduce(hcat, collect.(valid_xys))

        function gp_opt_func(xy)
            x, y = xy
            new_targets = new_line(x, y)
            det2 = add_line(current_det, new_targets)
            fishers, event_mask = calc_fisher(calc, det2, abs_scale=abs_scale, sca_scale=sca_scale)
            push!(xys, xy)
            push!(fisher_line_props, fishers)
            push!(fisher_line_props_masks, event_mask)
            return log10(evaluate_optimization_metric(metric, fishers, events[event_mask]) / current_metric)
        end

        gp_loss_f = make_loss(ColVecs(xys_vec), losses)
        θ_init = make_initial_guess_gp(xys_vec, losses)
        θ_opt, res = optimize_loss_gp(gp_loss_f, θ_init)

        θ_min = [θ_opt]
        f_min = [Optim.minimum(res)]

        #=
        for i in 1:5
            θ_rnd = make_initial_guess_gp(xys_vec, losses, true)
            θ_opt, res = optimize_loss_gp(gp_loss_f, θ_rnd)
            push!(θ_min, θ_opt)
            push!(f_min, Optim.minimum(res))
        end

        =#
        minix = argmin(f_min)
        θ_opt = θ_min[minix]

        gp_surrogate_opt = AbstractGPSurrogate(
            ColVecs(xys_vec),
            losses,
            gp=build_gp_prior(ParameterHandling.value(θ_opt)),
            Σy=ParameterHandling.value(θ_opt).noise_scale^2)

        opt_res = surrogate_optimize(
            gp_opt_func,
            LCBS(),
            lower_bound,
            upper_bound,
            gp_surrogate_opt,
            SobolSample();
            maxiters = 50, num_new_samples = ceil(Int64, 1.2*n_proposals))

        if isnothing(opt_res)
            amin = argmin(gp_surrogate_opt.y)
            opt_min = gp_surrogate_opt.x[amin]
            fmin =p_surrogate_opt.y[amin]
        else
            opt_min = opt_res[1]
            fmin = opt_res[2]
        end
 
        opt_min, fmin = get_surrogate_min(gp_surrogate_opt, lower, upper, opt_min)


        current_resolution =  evaluate_optimization_metric(resolution_metric, fishers_current, events[mask_current])
        current_resolution_alt = evaluate_optimization_metric(resolution_metric, fishers_current_alt, events[mask_current_alt])


        push!(data, (detector=current_det, surrogate=gp_surrogate_opt, xmin=opt_min, fisher=current_resolution, fisher_alt=current_resolution_alt, target_pred=fmin))
        new_targets = new_line(opt_min[1], opt_min[2])
        current_det = add_line(current_det, new_targets)
        gc()
        if (i+1) % 5 == 0
            plot_surrogate_non_anim(data, nothing, -extent/2-50, extent/2+50, "$(vid_fname)_$(type).mp4")
            #plot_surrogate_anim(data, nothing, -250, 250)
        end
    end
    return data
end

function main(args)
    medium = make_cascadia_medium_properties(0.95f0)
    sidelen = 80f0
    height::Float32 = sqrt(3)/2 * sidelen
    targets_triang = [
        make_detector_line(@SVector[0f0, 0f0, 0f0], 20, 50, 1, DummyTarget),
        make_detector_line(@SVector[-sidelen/2, -height, 0f0], 20, 50, 21, DummyTarget),
        make_detector_line(@SVector[sidelen/2, -height, 0f0], 20, 50, 41, DummyTarget),
    ]

    targets_line = [make_detector_line(@SVector[0f0, 0f0, 0f0], 20, 50, 1, DummyTarget)]

    detector = LineDetector(targets_line, medium)
    data = run("per_string_$(args[:type])", 6, vid_fname="$(args[:prefix])_$(args[:type])", n_events=args[:n_events], extent=args[:extent], geo_seed=detector, inj_radius=args[:inj_radius])
    return data
end


mode_choices = ["lightsabre", "extended"]
s = ArgParseSettings()
@add_arg_table s begin
    "--type"
    help = "Event type;  must be one of " * join(mode_choices, ", ", " or ")
    range_tester = (x -> x in mode_choices)
    "--inj_radius"
    arg_type=Float64
    default=500.
    "--extent"
    arg_type=Float64
    default=1200.
    "--n_events"
    default=10000
    arg_type=Int64
    "--prefix"
    default="fisher_gain"
end
parsed_args = parse_args(ARGS, s; as_symbols=true)

main(parsed_args)

#data = main()
#jldsave("fisher_gain_data.jld2", data=data)
function __stuff__()
    sidelen = 80f0
    height::Float32 = sqrt(3)/2 * sidelen
    targets_triang = [
        make_detector_line(@SVector[0f0, 0f0, 0f0], 20, 50, 1, DummyTarget),
        make_detector_line(@SVector[-sidelen/2, -height, 0f0], 20, 50, 21, DummyTarget),
        make_detector_line(@SVector[sidelen/2, -height, 0f0], 20, 50, 41, DummyTarget),
    ]

    targets_single = [
        #make_detector_line(@SVector[0f0, 0f0, 0f0], 20, 50, 1, DummyTarget),
        make_detector_line(@SVector[0f0, 0f0, 0f0], 20, 50, 21, DummyTarget),
        #make_detector_line(@SVector[sidelen/2, -height, 0f0], 20, 50, 41, DummyTarget),
    ]

    targets_hex = make_hex_detector(3, 50, 20, 50, truncate=1)

    medium = make_cascadia_medium_properties(0.95f0)
    detector =  LineDetector(targets_hex, medium)
    seed = 42

    type = "per_string_lightsabre"
    rng = Random.MersenneTwister(seed)
    same_events = true

    cylinder = get_bounding_cylinder(detector)
    pdist = CategoricalSetDistribution(OrderedSet([PMuPlus, PMuMinus]), [0.5, 0.5])
    edist = Dirac(1E5)
    ang_dist = LowerHalfSphere()
    length_dist = Dirac(1E4)
    time_dist = Dirac(0.0)
    inj = SurfaceInjector(CylinderSurface(cylinder), edist, pdist, ang_dist, length_dist, time_dist)


    fisher_model, diff_cache = setup_fisher_surrogate(type, max_particles=100000, max_targets=10*20)
    if occursin("per_string", type)        
        alt_type = occursin("extended", type)  ? "per_string_lightsabre" : "per_string_extended"
    else
        alt_type = occursin("extended", type)  ? "lightsabre" : "extended"
    end

    data = []
    current_det = detector
    current_inj = inj

    abs_scale = 1.
    sca_scale = 1.
    n_events = 100000
    evts = [rand(rng, current_inj) for _ in 1:n_events]
    calc = FisherCalculator(evts, fisher_model)
    metric = SingleEventTypePerEventMeanAngularError()

    fishers_current, mask_current = calc_fisher(calc, current_det, abs_scale=abs_scale, sca_scale=sca_scale)
    all_res = evaluate_optimization_metric(metric, fishers_current, evts[mask_current])
    finite_mask = isfinite.(all_res)



    azimuths = Float64[]
    resos = Float64[]
    for (ev, res) in zip(evts[mask_current][finite_mask], all_res[finite_mask])
        p = first(ev[:particles])
        p_rel = p.position .- cylinder.center
        if abs(p_rel[3]) == 1100/2
            continue
        end
        pos_normed = p_rel ./ norm(p_rel)
        push!(azimuths, cart_to_sph(p.direction)[2])
        push!(resos, res)
    end


    h1 = StatsBase.fit(Histogram, azimuths, 0:0.3:2*π)
    h2 = StatsBase.fit(Histogram, azimuths, Weights(resos), 0:0.3:2*π, )

    w = h2.weights ./ h1.weights

    stairs(h1.edges[1], [w; w[end]], step=:post)




    model = PhotonSurrogate(
        joinpath(ENV["ECAPSTOR"], "snakemake/time_surrogate_perturb/extended/amplitude_1_FNL.bson"),
        joinpath(ENV["ECAPSTOR"], "snakemake/time_surrogate_perturb/extended/time_uncert_1_2_FNL.bson"))


    input_buffer = create_input_buffer(model, 16*20, 1)
    output_buffer = create_output_buffer(16*20, 100)
    diff_cache = DiffCache(input_buffer, 13)


    hit_generator = SurrogateModelHitGenerator(gpu(model), 200.0, input_buffer, output_buffer)

    r = 50.
    phi = π/2
    phi = 0.
    p = Particle(SA_F64[r*cos(phi), r*sin(phi), -500], SA_F64[-cos(phi), -sin(phi), 0], 0., 1E4, 0., PEPlus)
    t = POM(SA[0., 0., -500.], 1)

    create_model_input!(model.amp_model, [p], [t], input_buffer, abs_scale=1., sca_scale=1.)

    detector = LineDetector([make_detector_line(@SVector[0f0, 0f0, 0f0], 20, 50, 21, POM)], medium)

    rng = Random.default_rng()
    matrices = first.(calc_fisher_matrix.(evts, Ref(detector), Ref(hit_generator), use_grad=true, rng=rng, cache=diff_cache))


    valid = isposdef.(matrices)
    valid_events = evts[valid]
    valid_matrices = matrices[valid]
    covs = inv.(valid_matrices)
    all_res = sqrt.(reduce(hcat, diag.(covs)))

    plot_resolution_hex(valid_events, sum(all_res, dims=1)[:], detector)


    cov_pred_sum,_ = predict_cov([event], detector_line, fisher_model, abs_scale=1, sca_scale=1)

    hbc, hbg = make_hit_buffers()

    r = 30.
    phis = 0:0.2:2*π
    medium = make_cascadia_medium_properties(0.95f0)
    t = POM(SA[0., 0., -500.], 1)
    detector = UnstructuredDetector([t], medium)
    abs_scale = 1.
    sca_scale = 1.
    rng = Random.MersenneTwister(1)

    nhits = Int64[]
    nhits_mc = Float64[]
    res = []

    for phi in phis
        p = Particle(SA_F64[r*cos(phi), r*sin(phi), -500], SA_F64[sin(phi), cos(phi), 0], 0., 5E4, 0., PEPlus)
        hit_times, mask = generate_hit_times(
                [p],
                detector,
                hit_generator,
                rng,
                device=gpu,
                abs_scale=abs_scale,
                sca_scale=sca_scale,
                )

        df = hit_list_to_dataframe(hit_times, get_detector_modules(detector), mask)

        push!(nhits, nrow(df))

        e = Event()
        e[:particles] = [p]
        f, _ = calc_fisher_matrix(e, detector, hit_generator, use_grad=true, rng=rng, cache=diff_cache, noise_rate=1E4, n_samples=400)
        cov = inv(f)
        all_res = sqrt.(diag(cov))
        push!(res, all_res)


        #=
        hits = propagate_particles([p], [t], 1, medium, hbc, hbg)
        if !isnothing(hits)
            push!(nhits_mc, sum(hits[:, :total_weight]))
        else
            push!(nhits_mc, 0)
        end
        =#

    end

    res = reduce(hcat, res)

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Azimuth", ylabel="Received Amplitude (PE)")
    ax2 = Axis(fig[1,1], ylabel="Total Uncertainty")

    lines!(ax, phis, nhits)

    ax2.yaxisposition = :right
    ax2.yticklabelalign = (:left, :center)
    ax2.xticklabelsvisible = false
    ax2.xticklabelsvisible = false
    ax2.xlabelvisible = false
    ax2.yticklabelcolor = :red
    linkxaxes!(ax,ax2)

    #lines!(ax, phis, nhits_mc)
    lines!(ax2, phis, sqrt.(sum(res.^2, dims=1)[:]) , color=:red)
    fig

    res
    hit_times = generate_hit_times(
                [p],
                det,
                hit_generator,
                rng,
                device=gpu,
                abs_scale=abs_scale,
                sca_scale=sca_scale,
                )
end