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

function add_line(x, y, detector)
    targets = detector.modules
    targets_new_line = make_detector_line(@SVector[x, y, 0.0], 20, 50, 1)
    new_det = Detector([targets; targets_new_line], detector.medium)
    return new_det
end

function make_injector(detector)
    cylinder = get_bounding_cylinder(detector)
    surf = CylinderSurface(cylinder)
    pdist = CategoricalSetDistribution(OrderedSet([PEMinus, PEPlus]), [0.5, 0.5])
    edist = Dirac(1E4)
    ang_dist = LowerHalfSphere()
    length_dist = Dirac(0.0)
    time_dist = Dirac(0.0)
    inj = SurfaceInjector(surf, edist, pdist, ang_dist, length_dist, time_dist)
    return inj
end


function run_fisher(event, detector, model, input_buffer, output_buffer)
    diff_cache = FixedSizeDiffCache(input_buffer, 6)
    hit_generator = SurrogateModelHitGenerator(model, 200.0, input_buffer, output_buffer)
    m, _ = calc_fisher_matrix(event, detector, hit_generator, use_grad=true, cache=diff_cache, rng=rng)
    return m
end

function det_and_run_and_proc(xy, model, input_buffer, output_buffer, detector)
    x, y = xy
    detector2 = add_line(x, y, detector)
    m = run_fisher(event, detector2, model, input_buffer, output_buffer)
    return  diag(sqrt(inv(m)))
end


function get_surrogate_min(gp_surrogate, lower_bound=-500, upper_bound=500)
    res = optimize(x -> gp_surrogate((x[1], x[2])), [lower_bound, lower_bound] , [upper_bound, upper_bound], [0., 0.], )
    xmin = Optim.minimizer(res)
    fmin = Optim.minimum(res)

    return xmin, fmin
end

function plot_surrogate(event, gp_surrogate, detector, lower, upper)
    y=x = lower:2.:upper
    #z = [gp_surrogate((xi, yi)) for xi in x for yi in y]
    p =  first(event[:particles])
    x0 = p.position
    x1 = p.direction * 600 .+ p.position

    fig = Figure()
    ax = Axis3(fig[1, 1], viewmode = :fit)
    lines!(ax, hcat(x0, x1) )

    modules = get_detector_modules(detector)

    mod_pos = reduce(hcat, [m.shape.position for m in modules])
    contour!(ax, x, y, (x1,x2) -> gp_surrogate((x1,x2)), levels=10)
    scatter!(ax, mod_pos)
    ax2 = Axis3(fig[1, 2],viewmode = :fit)
    surface!(ax2, x, y, (x, y) -> gp_surrogate((x, y)))
    
    xmin, _ = get_surrogate_min(gp_surrogate, lower, upper)

    scatter!(ax, xmin[1], xmin[2], 0)
    fig
end

function plot_surrogate_anim(data, event, lower, upper)
    y=x = lower:2.:upper
    p =  first(event[:particles])
    x0 = .-p.direction * 600 .+ p.position
    x1 = p.direction * 600 .+ p.position

    fig = Figure()
    ax = Axis3(fig[1, 1], viewmode=:fit)
    lines!(ax, hcat(x0, x1) )
    gp_eval = Observable(rand(length(x), length(y)))
    mod_pos = Observable(zeros(1, 3))

    contour!(ax, x, y, gp_eval, levels=10)
    scatter!(ax, mod_pos)

    xlims!(lower, upper)
    ylims!(lower, upper)
    
    ax2 = Axis3(fig[1, 2],viewmode = :fit, 
        zlabel=L"log_{10}\left (\sqrt{Tr(cov)}\right )",  xticklabelsize=12, yticklabelsize=12, zticklabelsize=12)
    surface!(ax2, x, y, gp_eval)

    xmin = Observable([0., 0.])
    #scatter!(ax, xmin[][1], xmin[][2], 0)

    cam3d!(ax.scene)
    update_cam!(ax.scene, cameracontrols(ax.scene), Vec3f(-1800, -1800, 800), Vec3f(0, 0, -500))

    lims = [[lower, upper], [lower, upper], [-1000., 0.]]
    m = maximum(abs(x[2] - x[1]) for x in lims)
    a = [m / abs(x[2] - x[1]) for x in lims]
    Makie.scale!(ax.scene, a...)

   
    zlims!(ax, -1000, 0)
    xlims!(lower, upper)
    ylims!(lower, upper)

    record(fig, "time_animation.mp4", framerate = 15) do io
        for d in data
            modules = get_detector_modules(d[:detector])
            mod_pos[] = reduce(hcat, [m.shape.position for m in modules])
            gp_surrogate = d[:surrogate]
            gp_eval[] = [gp_surrogate((x1,x2)) for x1 in x, x2 in y]
            xmin[] = d[:xmin]
            zmin, zmax = extrema(gp_eval[])
            zlims!(ax2, zmin, zmax)

           
            for i in 1:50
                rotate_cam!(ax.scene, Vec3f(0, deg2rad(360/90), 0))
                recordframe!(io)
            end
        end
    end
end


function fit_surrogate(detector, input_buffer, output_buffer, model, lower, upper)
    n_samples = 100
    lower_bound = [lower, lower]
    upper_bound = [upper, upper]
    k = with_lengthscale(Matern32Kernel(), [upper-lower, upper-lower])

    xys = Surrogates.sample(n_samples, lower_bound, upper_bound, SobolSample())
    zs = reduce(hcat, det_and_run_and_proc.(xys, Ref(model), Ref(input_buffer), Ref(output_buffer), Ref(detector)))
    xys_vec = reduce(hcat, collect.(xys))
    gp_surrogate = AbstractGPSurrogate(ColVecs(xys_vec), log10.(sum(zs, dims=1)[:]), gp= GP(k))

    return gp_surrogate
end

model_path = joinpath(ENV["ECAPSTOR"], "snakemake/time_surrogate")
model_tracks = PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_1_FNL.bson"), joinpath(model_path, "lightsabre/time_uncert_2.5_1_FNL.bson"))
model = gpu(model_tracks)


targets_line = make_detector_line(@SVector[0., 0.0, 0.0], 20, 50, 1)
medium = make_cascadia_medium_properties(0.95f0)

n_lines_max = 10

input_buffer = create_input_buffer(20*16*n_lines_max, 1)
output_buffer = create_output_buffer(20*16*n_lines_max, 100)

detector = Detector(targets_line, medium)
inj = make_injector(detector)
rng = MersenneTwister(49)

data = []
current_det = detector
for i in 1:9
    gp_surrogate = fit_surrogate(current_det, input_buffer, output_buffer, model, -200, 200)
    plot_surrogate(event, gp_surrogate, current_det, -200, 200)
    xmin, _ = get_surrogate_min(gp_surrogate, -200, 200)
    push!(data, (detector=current_det, surrogate=gp_surrogate, xmin=xmin))
    current_det = add_line(xmin[1], xmin[2], current_det)

end

plot_surrogate_anim(data, event, -100, 100)

data




f = Figure()
Axis(f[1, 1])

xs = LinRange(0, 10, 100)
ys = LinRange(0, 15, 100)
zs = [cos(x) * sin(y) for x in xs, y in ys]

contour!(xs, ys, zs,levels=-1:0.1:1)
f


x0

n_gen = 20
s = skip(SobolSeq([-150, -150], [150, 150]), n_gen)
resos = DataFrame(x_det=Float64[], y_det=Float64[], x=Float64[], y=Float64[], z=Float64[], logE=Float64[], theta=Float64[], phi=Float64[])
for i in 1:n_gen
    x, y = next!(s)
    detector2 = make_det_w_second_line(x, y)
    m = run_fisher(event, detector2, model)
    push!(resos, [x; y; diag(sqrt(inv(m)))])
end

fig, ax, s = scatter(resos[:, :x_det], resos[:, :y_det], color=resos[:, :theta])
Colorbar(fig[1, 2], s)
fig
lines(xs, resos[:, :logE])


model = gpu(model_tracks)
hit_generator = SurrogateModelHitGenerator(model, 200.0, input_buffer, output_buffer)

#times, range_mask = generate_hit_times(event, detector, hit_generator, rng, device=cpu)

m, _ = calc_fisher_matrix(event, detector, hit_generator, use_grad=true, cache=diff_cache, rng=rng)

