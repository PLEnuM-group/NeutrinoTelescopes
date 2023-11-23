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
using Zygote
using StatsBase

function add_line(x, y, detector)
    targets = detector.modules
    targets_new_line = make_detector_line(@SVector[x, y, 0.0], 20, 50, 1)
    new_det = Detector([targets; targets_new_line], detector.medium)
    return new_det
end

function make_muon_injector(detector)
    cylinder = get_bounding_cylinder(detector)
    cylinder = Cylinder(SA[0., 0., -475.], 1000., 200)
    surf = CylinderSurface(cylinder)
    pdist = CategoricalSetDistribution(OrderedSet([PMuPlus, PMuMinus]), [0.5, 0.5])
    edist = Dirac(1E4)
    ang_dist = LowerHalfSphere()
    length_dist = Dirac(1E4)
    time_dist = Dirac(0.0)
    inj = SurfaceInjector(surf, edist, pdist, ang_dist, length_dist, time_dist)
    return inj
end

function make_cascade_injector(detector)
    cylinder = get_bounding_cylinder(detector)
    cylinder = Cylinder(SA[0., 0., -475.], 1100., 200.)
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


function run_fisher(event::Event, detector, hit_generator::SurrogateModelHitGenerator, diff_cache)
    rng = Random.default_rng()
    return sqrt.(diag(inv(calc_fisher_matrix(event, detector, hit_generator, use_grad=true, cache=diff_cache, rng=rng, n_samples=100))))
end


function run_fisher(inj::Injector, detector, fisher_model::FisherSurrogateModel, diff_cache, nevents=200)
    targets = get_detector_modules(detector)

    covs = Matrix{Float64}[]
    for _ in 1:nevents
        event = rand(inj)    
        cov = predict_cov(event[:particles], targets, fisher_model)
        if !isnothing(cov)
            push!(covs, cov)
        end
    end
    cov = mean(covs)
    return sqrt.(diag(cov))
end

function run_fisher(event::Event, detector, fisher_model::FisherSurrogateModel, _)
    targets = get_detector_modules(detector)
    cov = predict_cov(event[:particles], targets, fisher_model)
    return sqrt.(diag(cov))
end

function expected_resolution(xy, event_or_inj, hit_generator_or_surrogate, detector; diff_cache=nothing)
    x, y = xy
    detector2 = add_line(x, y, detector)
    sigma = run_fisher(event_or_inj, detector2, hit_generator_or_surrogate, diff_cache)
    return sum(sigma)
end

function get_surrogate_min(gp_surrogate, lower_bound=-500, upper_bound=500)
    res = optimize(x -> gp_surrogate((x[1], x[2])), [lower_bound, lower_bound] , [upper_bound, upper_bound], [0., 0.], )
    xmin = Optim.minimizer(res)
    fmin = Optim.minimum(res)

    return xmin, fmin
end

function plot_surrogate(gp_surrogate, detector, lower, upper)
    y=x = lower:2.:upper
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
    hist!(ax3, string_distances, bins=0:15:200)

    g_res_vol = fig[2, 2] = GridLayout()

    ax4 = Axis(g_res_vol[1, 1], xlabel="Iteration", ylabel="Resolution", yscale=log10)

    resolutions = Observable[]
    iteration_range = Observable(Float64[])
    labels = ["x", "y", "z", "logE", "theta", "phi"]
    for (i,label) in enumerate(labels)
        obs = Observable(Float64[])
        lines!(ax4, iteration_range, obs, label=label)
        push!(resolutions, obs)
    end

    ax5 = Axis(g_res_vol[2, 1], xlabel="Iteration", ylabel="Detector Volume (Cyl.) km^3")
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

            bcyl = get_bounding_cylinder(d[:detector])
            det_vol[] = push!(det_vol[], get_volume(bcyl) / 1E9)

            reset_limits!(ax4)
            reset_limits!(ax5)
           
            for i in 1:25
                rotate_cam!(ax.scene, Vec3f(0, deg2rad(360/180), 0))
                recordframe!(io)
            end
        end
    end
end


function build_gp_prior(hyperparams)
    se_params = hyperparams.se
    se = se_params.σ^2 * with_lengthscale(Matern32Kernel(), [se_params.ℓ, se_params.ℓ])
    kernel = se
    return GP(kernel)
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


function make_initial_guess_gp()
    θ_init = (
        se = (
            σ = positive(1.),
            ℓ = positive(800.),
        ),
        noise_scale = positive(0.1),
    )
    return θ_init
end


function fit_surrogate(event_or_inj, detector, hit_generator_or_model, diff_cache, lower, upper)
    lower_bound = [lower, lower]
    upper_bound = [upper, upper]

    n_samples = 100

    xys = Surrogates.sample(50, lower_bound, upper_bound, SobolSample())
    zs = expected_resolution.(xys, Ref(event_or_inj), Ref(hit_generator_or_model), Ref(detector), diff_cache=Ref(diff_cache))
    xys_vec = reduce(hcat, collect.(xys))
     
    l = make_loss(ColVecs(xys_vec), log10.(zs))
    θ_init = make_initial_guess_gp()
    θ_opt, _ = optimize_loss_gp(l, θ_init)
   
    gp_surrogate_opt = AbstractGPSurrogate(
        ColVecs(xys_vec),
        log10.(zs),
        gp=build_gp_prior(ParameterHandling.value(θ_init)),
        Σy=ParameterHandling.value(θ_init).noise_scale^2)

    @show θ_opt
    
    minfunc(xs) = log10(expected_resolution(xs, event_or_inj, hit_generator_or_model, detector, diff_cache=diff_cache))
    surrogate_optimize(minfunc, EI(),lower_bound,upper_bound, gp_surrogate_opt, SobolSample(); maxiters = 50, num_new_samples = n_samples)

    return gp_surrogate_opt
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

function setup_fisher_sampler()
    model_path = joinpath(ENV["ECAPSTOR"], "snakemake/time_surrogate")
    model_tracks = PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_1_FNL.bson"), joinpath(model_path, "lightsabre/time_uncert_2.5_1_FNL.bson"))
    model = gpu(model_tracks)
    
    n_lines_max = 50

    input_buffer = create_input_buffer(20*16*n_lines_max, 1)
    output_buffer = create_output_buffer(20*16*n_lines_max, 100)
    diff_cache = FixedSizeDiffCache(input_buffer, 6)
    hit_generator = SurrogateModelHitGenerator(model, 200.0, input_buffer, output_buffer)

    return hit_generator, diff_cache
end

    

function setup_fisher_surrogate()
    model_fname = joinpath(ENV["ECAPSTOR"], "snakemake/fisher_surrogates/fisher_surrogate_extended.bson")
    fisher_surrogate = gpu(FisherSurrogateModel(model_fname))
    return fisher_surrogate, nothing
end


function run()

    targets_line = make_detector_line(@SVector[0., 0.0, 0.0], 20, 50, 1)
    medium = make_cascadia_medium_properties(0.95f0)

    n_lines_max = 70
    fisher_model, diff_cache = setup_fisher_surrogate()

    detector = Detector(targets_line, medium)
    inj = make_cascade_injector(detector)
    rng = MersenneTwister(12)
    
    event = rand(rng, inj)
    data = []
    current_det = detector
    current_inj = inj

    for i in ProgressBar(1:(n_lines_max-1))

        current_resolution = run_fisher(current_inj, current_det, fisher_model, diff_cache)
        current_inj = make_cascade_injector(current_det)
        gp_surrogate = fit_surrogate(current_inj, current_det, fisher_model, diff_cache, -250., 250.)
        xmin, _ = get_surrogate_min(gp_surrogate, -250, 250)
        push!(data, (detector=current_det, surrogate=gp_surrogate, xmin=xmin, fisher=current_resolution))
        current_det = add_line(xmin[1], xmin[2], current_det)
        gc()
        if i % 10 == 0
            plot_surrogate_anim(data, nothing, -250, 250)
        end
    end
end

run()



medium = make_cascadia_medium_properties(0.95f0)
targets_line = make_detector_line(@SVector[0., 0.0, 0.0], 20, 50, 1)

get_bounding_cylinder(Detector(targets_line, medium))


n_lines_max = 70
fisher_model, diff_cache = setup_fisher_surrogate()

detector = Detector(targets_line, medium)
inj = make_cascade_injector(detector)
rng = MersenneTwister(12)

current_det = detector
gp_surrogate = fit_surrogate(inj, current_det, fisher_model, diff_cache, -200.,200.)
xmin, _ = get_surrogate_min(gp_surrogate, -200, 200)
plot_surrogate(gp_surrogate, current_det, -200., 200.)

current_det = add_line(xmin[1], xmin[2], current_det)
gp_surrogate = fit_surrogate(inj, current_det, fisher_model, diff_cache, -150., 150.)
xmin, _ = get_surrogate_min(gp_surrogate, -500, 500)
plot_surrogate(gp_surrogate, current_det, -150., 150.)

current_det = add_line(xmin[1], xmin[2], current_det)
gp_surrogate = fit_surrogate(event, current_det, fisher_model, diff_cache, -500., 500.)
xmin, _ = get_surrogate_min(gp_surrogate, -500, 500)
plot_surrogate(event, gp_surrogate, current_det, -150., 150.)

current_det = add_line(xmin[1], xmin[2], current_det)
gp_surrogate = fit_surrogate(event, current_det, fisher_model, diff_cache, -500., 500.)
xmin, _ = get_surrogate_min(gp_surrogate, -500, 500)
plot_surrogate(event, gp_surrogate, current_det, -150., 150.)