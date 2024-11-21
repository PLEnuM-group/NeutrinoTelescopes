using Random
using StaticArrays
using PhotonSurrogateModel
using PhotonPropagation
using BSON
using NeutrinoTelescopes
using Rotations
using LinearAlgebra
using PhysicsTools
using ForwardDiff
using CairoMakie
using DataStructures
using Distributions
using BenchmarkTools
using Optimisers
using MLUtils
using ProgressLogging
using LoopVectorization
using Polyhedra
using GLPK
using Symbolics
using Optim
using PolyesterForwardDiff: threaded_gradient!
using Logging: global_logger
using TerminalLoggers
using FHist
using PrettyTables
using FLoops
using ParameterSchedulers
using Base.Iterators: product
using DataFrames
using SmoothingSplines
using Dates
using JLD2

global_logger(TerminalLogger(right_justify=120))

soft_cut(x, slope, cut_val) = 1 / (1 + exp(-slope * (x - cut_val)))

include("optimize_layout.jl")

function calculate_resolution(layout, surrogate, event::Event, diff_res)
    fisher_matrix, total_pred, n_above = DetectorOptimization.calc_total_fisher_matrix(layout, surrogate, event, diff_res)
    
    sym = 0.5 * (fisher_matrix + LinearAlgebra.transpose(fisher_matrix))
    metric = 90
    if !isapprox(det(sym), 0)

        cov = inv(sym)
        cov_sym = 0.5 * (cov + LinearAlgebra.transpose(cov))
        dir_cov = cov_sym[4:5, 4:5]

        dir_sph = cart_to_sph(first(event[:particles]).direction)
        
        if isposdef(dir_cov)
            metric = DetectorOptimization.get_directional_uncertainty(dir_sph, dir_cov)
        end
    end

    return metric
end


function calculate_detection_probability(layout, surrogate, event::Event, per_module_threshold=2, n_module_threshold=3)

    particle = first(event[:particles])
    n_module_det = 0
    @inbounds for mpos in layout
        n_hits_per_mod = 0
        for ppos in layout.pmt_positions
            n_hits_per_mod += surrogate(mpos, ppos, particle.position, particle.direction, particle.energy)
        end

        per_module_prob = poisson_atleast_k(n_hits_per_mod, per_module_threshold)

        n_module_det += per_module_prob
    end

    detection_prob = poisson_atleast_k(n_module_det, n_module_threshold)

    return detection_prob
end

function calculate_detection_probability(layout, surrogate, events::AbstractVector{<:Event}, per_module_threshold=2, n_module_threshold=3)
    probs = similar(events, Float64)
    @floop for i in eachindex(events)
        @inbounds probs[i] = calculate_detection_probability(layout, surrogate, events[i], per_module_threshold, n_module_threshold)    
    end
    return probs
end


function calculate_effective_volume(layout, surrogate, events, gen_volume, nbins=30, det_prob=nothing; per_module_threshold=2, n_module_threshold=3)
    if isnothing(det_prob)
        det_prob = calculate_detection_probability.(Ref(layout), Ref(surrogate), events, per_module_threshold, n_module_threshold)
    end
    energies = [e[:particles][1].energy for e in events]
    ebins = range(2, 7, length=30)

    hw = Hist1D(log10.(energies), weights=det_prob .* gen_volume, binedges=ebins)
    h = Hist1D(log10.(energies), nbins=nbins, binedges=ebins)

    return hw / h
end

function calculate_ang_res_vs_energy(layout, surrogate, events, nbins=30, det_prob=nothing)
    if isnothing(det_prob)
        det_prob = calculate_detection_probability.(Ref(layout), Ref(surrogate), events, per_module_threshold, n_module_threshold)
    end

    diff_res = create_diff_result(surrogate, Float64)
    ang_res = calculate_resolution.(Ref(layout), Ref(surrogate), events, Ref(diff_res))

    ebins = range(2, 7, length=nbins+1)
    bc = 0.5 * (ebins[1:end-1] + ebins[2:end])
    medians = Float64[]
    for i in 1:nbins
        idx = findall(x -> x >= ebins[i] && x < ebins[i+1], log10.([e[:particles][1].energy for e in events]))
        if isempty(idx)
            push!(medians, NaN)
            continue
        end

        vals = ang_res[idx]
        weights = Weights(det_prob[idx])
        push!(medians, median(vals, weights))
    end
    return bc, medians
end



function calc_eff_volume_res(radius)
    inj = make_injector(radius)
    events = [rand(inj) for _ in  1:2000]
    layout = make_line_layout(10, radius)
    det_prob = calculate_detection_probability.(Ref(layout), Ref(surrogate), events)
    energies = [e[:particles][1].energy for e in events]

    ebins = range(2, 7, length=30)

    hw = Hist1D(log10.(energies), weights=det_prob * get_volume(inj.volume), binedges=ebins)
    h = Hist1D(log10.(energies), nbins=20, binedges=ebins)

    ang_res = calculate_resolution.(Ref(layout), Ref(surrogate), events, Ref(diff_res))
    ang_res[.!isfinite.(ang_res)] .= 90
    return hw / h, mean(ang_res)
end




function get_n_above_threshold(event, layout; threshold=3.)
    module_pos = get_module_positions(layout)

    particle = first(event[:particles])
    n_mod_above_thr = 0
    @inbounds for mpos in module_pos
        n_hits_per_mod = 0
        for ppos in layout.pmt_positions
            n_hits_per_mod += predict_sr(mpos, ppos, particle.position, particle.direction, particle.energy)
        end

        if n_hits_per_mod > threshold
            n_mod_above_thr += 1
        end
    end

    return n_mod_above_thr
end

function get_event_positions(events)
    return [e[:particles][1].position for e in events]
end


function get_polyhedron(layout::OptimizationLayout)
    positions = get_module_positions(layout)
    v = vrep(positions)
    v = removevredundancy(v, GLPK.Optimizer)
    p = polyhedron(v)
    return p
end


function plot_optimization_history(layout, history, events)
    framerate = 5
    iterations = eachindex(history)

    fig = Figure(size=(800, 400))
    #ax = Axis3(fig[1,1])
    ax = Axis(fig[1, 1])
    ax2 = Axis(fig[1, 2])
    positions_xy = Observable(Point2f[])
    positions_xz = Observable(Point2f[])
    scatter!(ax, Point2f.(getindex.(get_event_positions(events), Ref([1,2]))), color=(:red, 0.2), markersize=5)
    scatter!(ax, positions_xy)

    scatter!(ax2, Point2f.(getindex.(get_event_positions(events), Ref([1,3]))), color=(:red, 0.2), markersize=5)
    scatter!(ax2, positions_xz)
    fig

    xlims!(ax, -200, 200)
    ylims!(ax, -200, 200)
    xlims!(ax2, -200, 200)
    ylims!(ax2, -600, 600)
    #zlims!(ax, -500, 500)


    record(fig, "time_animation.mp4", iterations;
            framerate = framerate) do it

        positions_xy[] = Point2f.(getindex.(history[it], Ref([1, 2])))
        positions_xz[] = Point2f.(getindex.(history[it], Ref([1, 3])))

    end
end

function plot_optimization_history_lines(history, surrogate, injector, metric; detailed=false, filename="time_animation.mp4", framerate=10, smoothing=10., detail_step=10)
    framerate = framerate
    figsize = detailed ? (800, 800) : (800, 400)

    volume = get_volume(injector.volume)

    fig = Figure(size=figsize)
    #ax = Axis3(fig[1,1])
    ax = Axis(fig[1, 1], xlabel="X (m)", ylabel="Y (m)")
    ax2 = Axis(fig[1, 2],xlabel="Iteration", ylabel="Loss")

    positions_xy = Observable(Point2f[])
    #scatter!(ax, Point2f.(getindex.(get_event_positions(events), Ref([1,2]))), color=(:red, 0.2), markersize=5)
    scatter!(ax, positions_xy)

    arc!(ax, Point2f(0), injector.volume.radius, -π, π)
    losses = Observable(Float64[])
    iterations_o = Observable(Int[])
    lines!(ax2, iterations_o, losses)

    smoothed_loss = Observable(Float64[])
    lines!(ax2, iterations_o, smoothed_loss)

    
    xlims!(ax2, 0, maximum(history.iteration))

    loss_vals = history.loss

    ylims!(ax2, minimum(loss_vals)*0.9, maximum(loss_vals)*1.1)

    #xlims!(ax, -100, 100)
    #ylims!(ax, -100, 100)

    if detailed
        ax3 = Axis(fig[2, 1], yscale=log10, xlabel="Log10(Energy) (GeV)", ylabel="Effective Volume (m^3)")
        ax4 = Axis(fig[2, 2], yscale=log10, xlabel="Log10(Energy) (GeV)", ylabel="Angular Resolution (deg)")
        eff_volume_hist = Observable(Hist1D(binedges=range(2, 7, length=30)))
        stephist!(ax3, eff_volume_hist)

        ang_res_bins = Observable(Float64[])
        ang_res_values = Observable(Float64[])
        lines!(ax4, ang_res_bins, ang_res_values)

        # Reference lines
         model = first(history.layout)
         events = [rand(injector) for _ in 1:5000]
         detection_probs = calculate_detection_probability.(Ref(model), Ref(surrogate), events, metric.per_module_threshold, metric.n_module_threshold)
 
          
         evolfuture = Threads.@spawn calculate_effective_volume(model, surrogate, events, volume, 20, detection_probs)
         aresfuture = Threads.@spawn calculate_ang_res_vs_energy(model, surrogate, events, 20, detection_probs)
 
         eff_vol_hist_ref = Threads.fetch(evolfuture)
         stephist!(ax3, eff_vol_hist_ref, linestyle=:dash)
         ebins_ares_ref, medians_ares_ref = Threads.fetch(aresfuture)
         lines!(ax4, ebins_ares_ref, medians_ares_ref, linestyle=:dash)


        
        xlims!(ax3, 2, 7)
        ylims!(ax3, 1E5, 1E9)

        xlims!(ax4, 2, 7)
        ylims!(ax4, 0.1, 90)
    end    

    

    record(fig, filename, eachrow(history);
            framerate = framerate) do it

        model = it.layout
        iteration = it.iteration
        loss = it.loss


        module_positions = collect(model)
        positions_xy[] = Point2f.(getindex.(module_positions, Ref([1, 2])))
        push!(iterations_o[], iteration)

        losses[] = push!(losses[], loss)
        if length(iterations_o.val) > 3
            spl = fit(SmoothingSpline, Float64.(iterations_o.val), losses.val, smoothing)
            smoothed_loss[] = predict.(Ref(spl), Float64.(iterations_o.val))
        end

        if detailed && (iteration % detail_step == 0)
            events = [rand(injector) for _ in 1:5000]
            detection_probs = calculate_detection_probability.(Ref(model), Ref(surrogate), events, metric.per_module_threshold, metric.n_module_threshold)
          
            evolfuture = Threads.@spawn calculate_effective_volume(model, surrogate, events, volume, 20, detection_probs)
            aresfuture = Threads.@spawn calculate_ang_res_vs_energy(model, surrogate, events, 20, detection_probs)

            eff_volume_hist[] = Threads.fetch(evolfuture)
            ebins, medians = Threads.fetch(aresfuture)
            ang_res_bins.val = ebins  
            ang_res_values[] = medians
        end
    end
    return fig
end



#radius = 380.
radius = 250.
e_min = 1E3
e_max = 1E7
inj = make_injector(radius, spectral_index=1, e_min=e_min, e_max=e_max)

rng = Random.MersenneTwister(1)
layout = make_line_layout(StringLayoutCart, 9, 90., rng)

batch_size = 2000
#metric = AngResDetEff(batch_size, get_volume(inj.volume)/1E9, 3, 2, -1, 1E4)

surrogate = SRSurrogateModel()
diff_res = create_diff_result(surrogate, Float64)

#sched = ParameterSchedulers.Constant(0.07)
sched = ParameterSchedulers.Constant(0.04)
iterations = 50
current_history = []
history = Channel(iterations)
constraints = [SimulationBoundaryConstraint(radius*0.9, 0.001)]


metric_val = metric(surrogate, layout, [rand(inj) for _ in 1:10])

# Find normalization factor
metric = AngResDetEff(batch_size, get_volume(inj.volume)/1E9, 3, 2, -1, 1, (e_min, e_max))
metric_val = metric(surrogate, layout, [rand(inj) for _ in 1:batch_size])
flux_norm = 1/metric_val
metric = AngResDetEff(batch_size, get_volume(inj.volume)/1E9, 3, 2,  -1, flux_norm, (e_min, e_max))
metric_val = metric(surrogate, layout, [rand(inj) for _ in 1:batch_size])



result = Threads.@spawn optimize_layout(
    layout, metric, surrogate, inj, schedule = sched, iterations=iterations, chunk_size=2, batch_size=batch_size, history=history,
    constraints=constraints,
    fix_angle=false)
bind(history, result);


layout_opt, history, state = fetch(result)
iterations = 100
batch_size = 5000
sched = CosAnnealDecay(0.025, 0.005, 50, 0.8)
#sched = ParameterSchedulers.Constant(0.03)
history = Channel(iterations)
result = Threads.@spawn optimize_layout(
    layout_opt, metric, surrogate, inj, schedule = sched, iterations=iterations, chunk_size=2, batch_size=batch_size, history=history,
    constraints=constraints,
    fix_angle=false, start_it=301, state=state)
bind(history, result);


istaskdone(result) && errormonitor(result)

while(isready(history))
    push!(current_history, take!(history))
end

current_history_df = sort(DataFrame(current_history), :iteration)[:, :]

fig = Threads.@spawn plot_optimization_history_lines(
    current_history_df,
    surrogate,
    inj,
    metric,
    detailed=true,
    framerate=20,
    smoothing=100.,
    detail_step=10,
    filename="time_animation_gamma2_detailed.mp4")


fig = Threads.@spawn plot_optimization_history_lines(
    current_history_df,
    surrogate,
    inj,
    metric,
    detailed=false,
    framerate=20,
    smoothing=100.,
    detail_step=20,
    filename="time_animation_gamma1.mp4")
    

DetectorOptimization.calc_total_fisher_matrix(layout, surrogate, rand(inj), diff_res)

function loss_fn(layout, batch)
    if !isnothing(constraints)
        penalty = sum(c(layout) for c in constraints; init=0.)
    end
    return metric(surrogate, layout, batch) + penalty
end

flat, re = destructure(layout)
diff_res = DiffResults.GradientResult(flat)
cur_loss = threaded_gradient!(DiffResults.gradient(diff_res), flat, ForwardDiff.Chunk(2), Val{true}()) do x
    re_layout = re(x)
    return loss_fn(re_layout, [rand(inj) for _ in 1:2000])
end

DiffResults.gradient(diff_res)


data = load("/home/wecapstor3/capn/capn100h/detector_opt/gamma1_test.jld2.CHK")
fig = Threads.@spawn plot_optimization_history_lines(
    data["history"],
    data["surrogate"],
    data["inj"],
    data["metric"],
    detailed=false,
    framerate=20,
    smoothing=100.,
    detail_step=5,
    filename="time_animation_gamma1.mp4")

fig = Threads.@spawn plot_optimization_history_lines(
    data["history"],
    data["surrogate"],
    data["inj"],
    data["metric"],
    detailed=true,
    framerate=20,
    smoothing=100.,
    detail_step=10,
    filename="time_animation_gamma1_detailed.mp4")

history

istaskdone(fig)


istaskdone(fig)

errormonitor(fig)


metric = AngResDetEff(batch_size, get_volume(inj.volume)/1E9, 3, 2, -1, 1E3)
metric(surrogate, layout, [rand(inj) for _ in 1:1000])


istaskdone(result)




spl = fit(SmoothingSpline, Float64.(current_history_df.iteration), current_history_df.grad_norm, 10.)
CairoMakie.lines(current_history_df.iteration, predict.(Ref(spl), Float64.(current_history_df.iteration)))


#sched = ParameterSchedulers.Constant(0.01)
iterations = 50
layout_opt, _, state = fetch(result)
history = Channel(iterations)
result = Threads.@spawn optimize_layout(
    layout_opt, metric, surrogate, inj, schedule = sched, iterations=iterations, chunk_size=2, batch_size=batch_size, history=history,
    constraints=[SimulationBoundaryConstraint(radius*0.9, 0.1)],
    fix_angle=false,
    state=state,
    start_it=6)
bind(history, result);



metric(surrogate, layout, [rand(inj) for _ in 1:10000])
metric(surrogate, current_history_df.layout[250], [rand(inj) for _ in 1:10000])
metric(surrogate, current_history_df.layout[400], [rand(inj) for _ in 1:10000])
metric(surrogate, current_history_df.layout[end], [rand(inj) for _ in 1:10000])




positions = get_event_positions(sim_events)

layout_opt, _, state = fetch(result)
iterations = 100
history = Channel(iterations)
result = Threads.@spawn optimize_layout(
    layout_opt, sim_events, metric, surrogate, inj, schedule = sched, iterations=iterations, chunk_size=2, batch_size=5000, history=history,
    constraints=[SimulationBoundaryConstraint(390., 0.1)],
    fix_angle=true,
    state=state)
bind(history, result);



n_events = [500, 1000, 2000, 5000, 10000]
losses = Channel(length(n_events) * 20)

for nev in n_events
    metric = AngResDetEff(nev, get_volume(inj.volume)/1E9, 3, 2)
    Threads.@threads for it in 1:20
        rng = MersenneTwister(it)
        sim_events = [rand(rng, inj) for _ in  1:nev]
        put!(losses, (nev=nev, it=it, loss=metric(surrogate, layout, sim_events)))
    end
end

losses_fetched = []
while(isready(losses))
    push!(losses_fetched, take!(losses))
end

losses_df = DataFrame(losses_fetched)

summary_df = combine(groupby(losses_df, :nev), :loss => mean => :mean, :loss => std => :std)

summary_df[:, :mean] ./ summary_df[:, :std] 


layout_opt, _, state = fetch(result)

sched_shifted = Shifted(CosAnnealDecay(10., 1., 300, 0.8), -700)
history = Channel(800)
result = Threads.@spawn optimize_layout(
    layout_opt, sim_events, metric, surrogate, inj, schedule = sched_shifted, iterations=800, chunk_size=2, batch_size=1000, history=history,
    constraints=[SimulationBoundaryConstraint(390., 0.1)],
    start_it=801, state=state)
bind(history, result);


istaskdone(result)

current_history_df = sort(DataFrame(current_history), :iteration)



plot_optimization_history_lines(current_history_df[2:end, :], sim_events, surrogate, inj, metric, detailed=false, framerate=20)



spl = fit(SmoothingSpline, Float64.(current_history_df.iteration), current_history_df.loss, 100000.0)

CairoMakie.lines(current_history_df.iteration, predict.(Ref(spl), Float64.(current_history_df.iteration)))

CairoMakie.lines(1:1400, sched_shifted.(1:1400))

isready(history)

istaskdone(result)

errormonitor(result)

sched = CosAnneal(0.1, 3., 100, true)






calculate_detection_probability(layout, surrogate, events)

calculate_detection_probability.(Ref(layout), Ref(surrogate), events)


ang_res = calculate_resolution.(Ref(layout), Ref(surrogate), events, Ref(diff_res))


@time calculate_resolution.(Ref(layout), Ref(surrogate), events, Ref(diff_res))

@time calculate_resolution(layout, surrogate, events, diff_res)

ang_res[.!isfinite.(ang_res)] .= 90

pos_xy = getindex.(positions, Ref([1, 2]))
pos_xz = getindex.(positions, Ref([1, 3]))
ang_res

fig, ax ,s = scatter(Point2f.(pos_xy), color=log10.(ang_res))

scatter!(ax, Point2f.(layout.positions_xy), color=(:red))
fig

m, _, _ = calc_total_fisher_matrix(layout, surrogate, events[1], diff_res)

cov = inv(m)

cov[4:5, 4:5]

sym = 0.5 * (cov + LinearAlgebra.transpose(cov))
isposdef(sym[4:5, 4:5])





diff_res = create_diff_result(surrogate, Float64)


radii = [100., 200., 300., 500., 800.]

eff_vols = []
med_resolutions = []
for radius in radii
    effv, mres = calc_eff_volume_res(radius)

    push!(eff_vols, effv)
    push!(med_resolutions, mres)

end

fig = Figure()
ax = Axis(fig[1, 1], yscale=log10)

for (r, ev) in zip(radii, eff_vols)
    stephist!(ax, ev, label="R = $r m")
end
ylims!(ax, 1E6, 1E9)
fig

med_resolutions





fig, ax, p = plot(hw / h, axis=(; yscale=log10));

fig




n_hits = Float64[]
for m in modules
    n_per_modules = 0
    for pmt_pos in modules.pmt_positions
        n_per_modules += surrogate(events[1][:particles][1], m, pmt_pos)
    end
    push!(n_hits, n_per_modules)
end








