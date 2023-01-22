using NeutrinoTelescopes
using Flux
using CUDA
using Random
using StaticArrays
using BSON: @save, @load
using BSON
using CairoMakie
using Rotations
using LinearAlgebra
using DataFrames
using Zygote

using SpecialFunctions
using StatsBase
using Base.Iterators
using Distributions
using Optim
using QuadGK
using Base.Iterators
using Formatting
using BenchmarkTools
using PyCall

pp = pyimport("proposal")
pp.InterpolationSettings.tables_path = joinpath(@__DIR__,"../assets/proposal_tables")


models = Dict(
    "1" => joinpath(@__DIR__, "../assets/rq_spline_model_l2_0_1_FNL.bson"),
    "2" => joinpath(@__DIR__, "../assets/rq_spline_model_l2_0_2_FNL.bson"),
    "3" => joinpath(@__DIR__, "../assets/rq_spline_model_l2_0_3_FNL.bson"),
    "4" => joinpath(@__DIR__, "../assets/rq_spline_model_l2_0_4_FNL.bson"),
    "5" => joinpath(@__DIR__, "../assets/rq_spline_model_l2_0_5_FNL.bson"),
    #"FULL" => joinpath(@__DIR__, "../assets/rq_spline_model_l2_0_FULL_FNL.bson")
)

targets_single = [make_pone_module(@SVector[-25., 0., -450.], 1)]
targets_line = make_detector_line(@SVector[-25., 0.0, 0.0], 20, 50, 1)
targets_three_l = [
    make_detector_line(@SVector[-25., 0.0, 0.0], 20, 50, 1)
    make_detector_line(@SVector[25., 0.0, 0.0], 20, 50, 21)
    make_detector_line(@SVector[0., sqrt(50^2-25^2), 0.0], 20, 50, 41)]
targets_hex = make_hex_detector(3, 50, 20, 50, truncate=1)

detectors = Dict("Single" => targets_single, "Line" =>targets_line, "Tri" => targets_three_l, "Hex" => targets_hex)
medium = make_cascadia_medium_properties(0.99f0)

function loss_to_particle(loss)
    energy = loss.energy / 1E3
    pos = SA[loss.position.x / 100, loss.position.y / 100, loss.position.z / 100] 
    dir = SA[loss.direction.x, loss.direction.y, loss.direction.z]
    time = loss.time * 1E9

    return Particle(pos, dir, time, energy, 0., PEMinus)
end


function propagate_muon(particle)

    position = particle.position
    direction = particle.direction
    length = particle.length
    time = particle.time

    if particle.type == PMuMinus
        particle = pp.particle.MuMinusDef()
    elseif particle.type == PMuPlus
        particle = pp.particle.MuPlusDef()
    else
        error("Type $(particle.type) not supported")
    end
    propagator = pp.Propagator(particle, joinpath(@__DIR__,"../assets/proposal_config.json"))

    initial_state = pp.particle.ParticleState()
    initial_state.energy = energy*1E3
    initial_state.position = pp.Cartesian3D(position[1]*100, position[2]*100, position[3]*100)
    initial_state.direction = pp.Cartesian3D(direction[1], direction[2], direction[3])
    initial_state.time = time / 1E9
    final_state =  propagator.propagate(initial_state, max_distance=length*100)
    stochastic_losses = final_state.stochastic_losses()
    loss_to_particle.(stochastic_losses)
end



function create_mock_muon(energy, position, direction, time, mean_free_path, length, rng)
    losses = []

    step_dist = Exponential(mean_free_path)
    eloss_logr_dist = Uniform(-6, -1)

    dist_travelled = 0
    while dist_travelled < length && energy > 1
        step_size = rand(rng, step_dist)
        e_loss = energy * 10^rand(rng, eloss_logr_dist)

        energy -= e_loss
        dist_travelled += step_size
        pos = position .+ dist_travelled .* direction
        t = time + dist_travelled / c_vac_m_ns

        push!(losses, Particle(pos, direction, t, e_loss, 0., PEMinus))

    end

    return losses
end

function plot_hits_on_module(data, pos, dir, particles_truth, particles_unfolded, model, target, medium)
    fig = Figure(resolution=(1500, 1000))
    ga = fig[1, 1] = GridLayout(4, 4)

    t_geo = calc_tgeo_tracks(pos, dir, target.position, medium)
    n_pmt = get_pmt_count(eltype(targets))
    
    for i in 1:n_pmt
        row, col = divrem(i - 1, 4)
        ax = Axis(ga[col+1, row+1], xlabel="Time Residual(ns)", ylabel="Photons / time", title="PMT $i",
                  )
        hist!(ax, data[i] .- t_geo, bins=-50:3:150, color=:orange)
    end

    times = -50:1:150
    for particles in [particles_truth, particles_unfolded]
    
      
        shape_lhs = []
        local log_expec
        for t in times
            _, shape_lh, log_expec = SurrogateModels.evaluate_model(particles, Vector.(eachrow(repeat([t + t_geo], n_pmt))), [target], gpu(model), tf_dict, c_n)
            push!(shape_lhs, collect(shape_lh))
        end

        shape_lh = reduce(hcat, shape_lhs)

        for i in 1:n_pmt
            row, col = divrem(i - 1, 4)
            lines!(ga[col+1, row+1], times, exp.(shape_lh[i, :] .+ log_expec[i]))
            
        end
    end

    return fig
end






begin
    c_n = c_at_wl(800f0, medium)
    p0 = SA[8., -5., -450]
    dir_theta = 0.7
    dir_phi = 1.3
    dir = sph_to_cart(dir_theta, dir_phi)
    t0 = 0.
    backtrack = 300
    pos_shifted = p0 .- backtrack .* dir 
    time_shifted = t0 - backtrack / c_vac_m_ns
    energy = 4e5

    theta_deg = rad2deg(dir_theta)
    phi_deg = rad2deg(dir_phi)

    rng = MersenneTwister(3)
   
    muon = Particle(pos_shifted, dir, time_shifted, energy, backtrack*2., PMuMinus)
    losses = propagate_muon(muon)
    losses_filt = [p for p in losses if p.energy > 100]

    @load models["3"] model hparams opt tf_dict
end

begin
    targets = targets_hex
    target_mask = any(norm.([p.position for p in losses_filt] .- permutedims([t.position for t in targets])) .<= 100, dims=1)[1, :]

    targets_range = targets[target_mask]
    data = sample_multi_particle_event(losses_filt, targets_range, model, tf_dict, c_n, rng)

end


data_per_target = reshape(data, 16, Int(length(data)/16))
nhits_per_module = sum(length.(data_per_target), dims=1)[:]
losses_unfolded = unfold_energy_losses(p0, dir, t0; data=data, targets=targets_range, model=model, tf_vec=tf_dict, spacing=1., plength=2*backtrack)


fig = Figure()
ax = Axis(fig[1, 1], yscale=log10, limits=(-1000, 1000, 1E2, 1E5))

hist!(ax, [l.time for l in losses_filt], weights=[l.energy for l in losses_filt], bins=-1000:300:1000, fillto = 1E2)
hist!(ax, [l.time for l in losses_unfolded], weights=[l.energy for l in losses_unfolded], bins=-1000:300:1000, fillto = 1E2)
fig

plot_hits_on_module(data_per_target[:, argmax(nhits_per_module)], p0, dir, losses_filt, losses_unfolded, model, targets_range[argmax(nhits_per_module)], medium)

feat_buffer = zeros(9, get_pmt_count(eltype(targets_range))*length(targets_range)*length(losses_filt))
t_first_likelihood(losses_filt; data=data, targets=targets_range, model=gpu(model), tf_vec=tf_dict, c_n=c_n, feat_buffer=nothing)





begin
    particles = losses_filt
    range = (theta_deg-20, theta_deg+20, phi_deg-20, phi_deg+20, 1)
    axis_ranges = Dict(
        "Tri" => (theta_deg-1, theta_deg+1, phi_deg-1, phi_deg+1, 0.05),
        "Single" => (0, 180, -180, 180, 3),
        "Line" => (theta_deg-10, theta_deg+10, phi_deg-15, phi_deg+15, 0.5),
        "Hex" =>  (theta_deg-0.5, theta_deg+0.5, phi_deg-0.5, phi_deg+0.5, 0.05))


    axis_ranges = Dict(
        "Tri" => range,
        "Single" => range,
        "Line" => range,
        "Hex" =>  range)

    levels = map(sigma -> invlogcdf(Chisq(2), log(1-2*ccdf(Normal(), sigma))), [1, 2, 3])

    fig = Figure()
    ga = fig[1, 1] = GridLayout()

    for (i, (dkey, targets)) in enumerate(detectors)

        ranges = axis_ranges[dkey]
        thetas = deg2rad.(ranges[1]:ranges[end]:ranges[2])
        phis = deg2rad.(ranges[3]:ranges[end]:ranges[4])

        target_mask = any(norm.([p.position for p in losses_filt] .- permutedims([t.position for t in targets])) .<= 100, dims=1)[1, :]

        if !any(target_mask)
            continue
        end

        targets_range = targets[target_mask]
        data = sample_multi_particle_event(particles, targets_range, model, tf_dict, c_n, rng)
        
        #unfolded = unfold_energy_losses(pos, dir; data=data, model=model, tf_vec=tf_dict, targets=targets_range)

        #feat_buffer = zeros(9, get_pmt_count(eltype(targets_range))*length(targets_range)*length(particles))
        #llh2d = track_likelihood_fixed_losses.(log10(energy), thetas, permutedims(phis), Ref(pos); muon_energy=energy, losses=particles, data=data, targets=targets_range, model=gpu(model), tf_vec=tf_dict, c_n=c_n, feat_buffer=feat_buffer)
        
        feat_buffer = zeros(9, get_pmt_count(eltype(targets_range))*length(targets_range)*length(dist_along))
        llh2d = track_likelihood_energy_unfolding.(
            thetas, permutedims(phis), Ref(p0), t0; data=data, targets=targets_range, model=gpu(model),
            tf_vec=tf_dict, c_n=c_n, spacing=0.5, amp_only=true)
        
        
        rellh = maximum(llh2d) .- llh2d
        row, col = divrem(i-1, 2)
        ax = Axis(ga[row+1, col+1], title=dkey, xlabel="Zenith (deg)", ylabel="Azimuth (deg)", limits=ranges[1:4])
        hm = CairoMakie.heatmap!(ax, rad2deg.(thetas), rad2deg.(phis), rellh, colorrange=(0, 200))
        CairoMakie.contour!(ax,  rad2deg.(thetas), rad2deg.(phis), rellh, levels=levels, color=:white)
        CairoMakie.scatter!(ax, theta_deg, phi_deg)
    end
    #Colorbar(fig[1, 2], hm)
    fig
end


rng = MersenneTwister(5)
losses = create_mock_muon(energy, pos, dir, 0., 30, 500, rng)
losses_filt = [p for p in losses if p.energy > 100]

target_mask = any(norm.([p.position for p in losses_filt] .- permutedims([t.position for t in targets_hex])) .<= 100, dims=1)[1, :]
targets_range = targets_hex[target_mask]

data = sample_multi_particle_event(losses_filt, targets_range, model, tf_dict, c_n, rng)

unfolded = unfold_energy_losses(pos, dir; data=data, model=model, tf_vec=tf_dict, targets=targets_range)


fig = Figure()
ax  = Axis(fig[1, 1], yscale=log10)


dist_along = [norm(p.position .- pos) for p in lvec]
mask = escales[:] .> 0
scatter!(ax, dist_along[mask], escales[:][mask] .*energy)
dist_along = [norm(p.position .- pos) for p in losses_filt]
scatter!(ax, dist_along, [p.energy for p in losses_filt])
fig




begin
    pos = SA[-150., -5., -450]
    theta = 1.4
    phi = 0.4
    energy = 5E4
    rng = MersenneTwister(31339)
    dir = sph_to_cart(theta, phi)

    losses = create_mock_muon(energy, pos, dir, 0., 30, 500, rng)
    losses_filt = [p for p in losses if p.energy > 100]


    @load models["4"] model hparams opt tf_dict
    c_n = c_at_wl(800f0, medium)
    target_mask = any(norm.([p.position for p in losses_filt] .- permutedims([t.position for t in targets_hex])) .<= 200, dims=1)[1, :]

    targets_range = targets_hex[target_mask]

    data = sample_multi_particle_event(losses_filt, targets_range, model, tf_dict, c_n, rng)

    gpu_model = gpu(model)
    # @code_warntype SurrogateModels.evaluate_model(losses_filt, data, targets_range, model, tf_dict, c_n)
    #@code_warntype track_likelihood_fixed_losses(log10(energy), theta, phi, pos, 0.; losses=losses_filt, muon_energy=energy, data=data, targets=targets_range, model=model, tf_vec=tf_dict, c_n=c_n)
    llh = track_likelihood_fixed_losses(log10(energy), theta, phi, pos, 0.; losses=losses_filt, muon_energy=energy, data=data, targets=targets_range, model=gpu_model, tf_vec=tf_dict, c_n=c_n)
    b1 = @benchmark track_likelihood_fixed_losses(log10(energy), theta, phi, pos, 0.; losses=losses_filt, muon_energy=energy, data=data, targets=targets_range, model=gpu_model, tf_vec=tf_dict, c_n=c_n)

    feat_buffer = zeros(9, 16*length(targets_range)*length(losses_filt))
    llh2 = track_likelihood_fixed_losses(log10(energy), theta, phi, pos, 0.;
                                            losses=losses_filt, muon_energy=energy, data=data, targets=targets_range, model=gpu_model, tf_vec=tf_dict, c_n=c_n,
                                            feat_buffer=feat_buffer)

    b2 = @benchmark track_likelihood_fixed_losses(log10(energy), theta, phi, pos, 0.; losses=losses_filt, muon_energy=energy, data=data, targets=targets_range, model=gpu_model, tf_vec=tf_dict, c_n=c_n,
                                            feat_buffer=feat_buffer)
    llh ≈ llh2

end

b1
b2

@profview track_likelihood_fixed_losses(log10(energy), theta, phi, pos, 0.; losses=losses_filt, muon_energy=energy, data=data, targets=targets_range, model=gpu(model), tf_vec=tf_dict, c_n=c_n,
feat_buffer=feat_buffer)



begin
    pos = SA[-150., -5., -450]
    theta = 1.4
    phi = 0.4
    energy = 5E4
    rng = MersenneTwister(31339)
    dir = sph_to_cart(theta, phi)

    losses = create_mock_muon(energy, pos, dir, 0., 30, 500, rng)
    losses_filt = [p for p in losses if p.energy > 100]

    particles = losses_filt
    #=particles = [
        Particle(pos, dir, 0., energy, PEMinus),
        Particle(pos .+ dir.*5, dir, 15, energy, PEMinus),
        Particle(pos .+ dir.*10, dir, 25, energy, PEMinus)
] =#

    @load models["4"] model hparams opt tf_dict
    c_n = c_at_wl(800f0, medium)
    #target_mask = any(norm.([p.position for p in particles] .- permutedims([t.position for t in targets_hex])) .<= 200, dims=1)[1, :]
    #targets_range = targets_hex[target_mask]
    targets_range = targets_single


    times = -50:1:100
    fig = Figure(resolution=(1500, 1000))
    ga = fig[1, 1] = GridLayout(4, 4)

    n_pmt = get_pmt_count(eltype(targets_range))

    t_geos = repeat([calc_tgeo(norm(particles[1].position - t.position) - t.radius, c_n) for t in targets_range], n_pmt)
    t0 = particles[1].time

    oversample = 500
    @load models["4"] model hparams opt tf_dict
    samples = sample_multi_particle_event(particles, targets_range, model, tf_dict, c_n, rng, oversample=oversample)
    tgeo = calc_tgeo(norm(particles[1].position - targets_range[1].position) - targets_range[1].radius, c_n)
    for i in 1:16
        row, col = divrem(i - 1, 4)
        hist(ga[col+1, row+1], samples[i] .- tgeo .- t0 , bins=-50:3:100, normalization=:density, fillaplha=0.3, weights=fill(1/oversample, length(samples[i])))
    end

    input = calc_flow_input(particles, targets_range, tf_dict)


    shape_lhs = []
    local log_expec
    for t in times
        _, shape_lh, log_expec = SurrogateModels.evaluate_model(particles, Vector.(eachrow(t .+ t_geos .+ t0)), targets_range, model, tf_dict, c_n)
        push!(shape_lhs, collect(shape_lh))
    end

    shape_lh = reduce(hcat, shape_lhs)

    for i in 1:16
        row, col = divrem(i - 1, 4)
        lines!(ga[col+1, row+1], times, exp.(shape_lh[i, :] .+ log_expec[i]))

    end

    fig
end







pos = SA[8., -5., -450]
dir_theta = 0.7
dir_phi = 1.3
dir = sph_to_cart(dir_theta, dir_phi)
energy = 3e4

rng = MersenneTwister(31338)
particles = [
        Particle(pos, dir, 0., energy, PEMinus),
        Particle(pos .+ dir.*5, dir, 15, energy, PEMinus),
        Particle(pos .+ dir.*10, dir, 25, energy, PEMinus)
]


@load models["1"] model hparams opt tf_dict

c_n = c_at_wl(800f0, medium)







samples = sample_event(1E4, 0.1, 0.1, SA[-10., 10., 10.], targets, model, tf_dict, rng=Random.GLOBAL_RNG)

dir_theta = 0.1
dir_phi = 0.1
pos = SA[10., 10., -10.]
logenergy = 4.
rng = MersenneTwister(31338)
samples = sample_event(10^logenergy, dir_theta, dir_phi, pos, [target], model, tf_dict; rng=rng)

begin
    pos = SA[-10., 10., 10.]
    dir_theta = 0.7
    dir_phi = 0.5
    dir = sph_to_cart(dir_theta, dir_phi)
    energy = 5e4
    particles = [ Particle(pos, dir, 0., energy, PEMinus)]

    hits = mc_expectation(particles, [target], 1)
    compare_mc_model(particles, [target], models, hits)
end



begin
    min_vals = Dict{String, Vector{Any}}()
    for i in 1:50
        hits = mc_expectation(particles, [target], i)
        resampled = resample_simulation(hits, time_col=:tres)
        rs_hits = []
        for i in 1:16
            mask = resampled[:, :pmt_id] .== i
            sel = Vector{Float64}(resampled[mask, :tres])
            push!(rs_hits, sel)
        end

        for (mname, model_path) in models
            if !haskey(min_vals, mname)
                min_vals[mname] = []
            end
            m = BSON.load(model_path)
            res = min_lh(rs_hits, pos, [target], m[:model], m[:tf_dict])
            push!(min_vals[mname], Optim.minimizer(res))
        end
    end
end




fig = Figure()
ax = Axis(fig[1, 1])

bins = 0:5:60
for (k, v) in min_vals

    v = reduce(hcat, v)

    hist!(ax, rad2deg.(acos.(dot.(sph_to_cart.(v[2, :], v[3, :]), Ref(sph_to_cart(dir_theta, dir_phi))))),
    label=k, bins=bins)

end

norm([10, 10, 10])

fig

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Angular Resolution (deg)", ylabel="Counts", title="50TeV extended 17m distance ")
v = reduce(hcat, min_vals["5"])
ang_res = rad2deg.(acos.(dot.(sph_to_cart.(v[2, :], v[3, :]), Ref(sph_to_cart(dir_theta, dir_phi)))))

hist!(ax, ang_res, bins=bins)
vlines!(ax, median(ang_res), color=:black, label=format("Median: {:.2f}°",median(ang_res) ))
leg = Legend(fig[1, 2], ax)
fig





begin
    pos = SA[-10., 10., 0.]
    dir_theta = 0.7
    dir_phi = 0.5
    dir = sph_to_cart(dir_theta, dir_phi)
    energy = 5e4
    particles = [ Particle(pos, dir, 0., energy, PEMinus)]
    hits = mc_expectation(particles, [target])

    dir_theta = 0.71
    dir = sph_to_cart(dir_theta, dir_phi)
    particles = [ Particle(pos, dir, 0., energy, PEMinus)]
    hits2 = mc_expectation(particles, [target])

    fig = Figure(resolution=(1000, 700))
    ga = fig[1, 1] = GridLayout(4, 4)

    for i in 1:16
        row, col = divrem(i - 1, 4)

        ax = Axis(ga[col+1, row+1], xlabel="Time Residual(ns)", ylabel="Photons / time", title="PMT $i")
        mask = hits[:, :pmt_id] .== i
        hist!(ax, hits[mask, :tres], bins=-10:5:100, weights=hits[mask, :total_weight], color=:blue, normalization=:density)
        mask = hits2[:, :pmt_id] .== i
        hist!(ax, hits2[mask, :tres], bins=-10:5:100, weights=hits2[mask, :total_weight], color=:orange, normalization=:density)
    end
    fig
end


begin
    pos = SA[-10., 10., 10.]
    dir_theta = 0.7
    dir_phi = 0.5
    dir = sph_to_cart(dir_theta, dir_phi)
    energy = 5e4

    delta = 50 / 1E6 * energy

    particles = [
         Particle(pos, dir, 0., energy, PEMinus),
         Particle(pos .+ dir .*delta, dir, delta*0.3, energy, PEMinus)
    ]

    hits = mc_expectation(particles, [target])
    compare_mc_model(particles, [target], models, hits)
end



begin
    model_res = Dict()
    for (mname, model_path) in models
        m = BSON.load(model_path)
        Flux.testmode!(m[:model])
        res = calc_resolution_maxlh([target], m, pos, 200)
        model_res[mname] = res
    end

    m1 = BSON.load(models["1"])
    m2 = BSON.load(models["2"])
    Flux.testmode!(m1[:model])
    Flux.testmode!(m2[:model])
    model_res["1-2"] = calc_resolution_maxlh([target], m1, m2, pos, 200)

    fig = Figure()
    ax = Axis(fig[1, 1])

    bins = 0:1:60

    for (k, v) in model_res

        hist!(ax, rad2deg.(acos.(dot.(sph_to_cart.(v[2, :], v[3, :]), Ref(sph_to_cart(0.1, 0.2))))),
        label=k, bins=bins)

    end


end

fig

leg = Legend(fig[1, 2], ax)
fig


@load models["4"] model hparams opt tf_dict
samples = sample_event(1E4, 0.1, 0.2, pos, model, tf_dict)

length(samples)
likelihood(4,  0.1, 0.2, pos, samples, [target], model, tf_dict)


log_energies = 3:0.05:5
lh_vals = [likelihood(e,  0.1, 0.2, pos, samples, [target], model, tf_dict) for e in log_energies]
CairoMakie.scatter(log_energies, lh_vals)

zeniths = 0:0.01:0.5
lh_vals = [likelihood(4,  z, 0.2, pos, samples, [target], model, tf_dict) for z in zeniths]
CairoMakie.scatter(zeniths, lh_vals)



pos = SA[10., 30., 10.]
samples = sample_event(1E4, 0.1, 0.2, pos, tf_dict)

log_energies = 3:0.05:5
lh_vals = [likelihood(e,  0.1, 0.2, pos, samples, [target], tf_dict) for e in log_energies]
scatter(log_energies, lh_vals, axis=(limits=(3, 5, -10000, -500), ))

zeniths = 0:0.01:0.5
lh_vals = [likelihood(4,  z, 0.2, pos, samples, [target], tf_dict) for z in zeniths]
scatter(zeniths, lh_vals)

Zygote.gradient( x -> likelihood(x[1], x[2], x[3], pos, samples, [target], tf_dict), [1E4, 0.1, 0.2])



hist(rad2deg.(acos.(dot.(sph_to_cart.(min_vals[2, :], min_vals[3, :]), Ref(sph_to_cart(0.1, 0.2))))))



function calc_fisher(logenergy, dir_theta, dir_phi, n, targets, model; use_grad=false, rng=nothing)

    matrices = []
    for _ in 1:n

        pos_theta = acos(rand(rng, Uniform(-1, 1)))
        pos_phi = rand(rng, Uniform(0, 2*pi))
        r = sqrt(rand(rng, Uniform(5^2, 50^2)))
        pos = r .* sph_to_cart(pos_theta, pos_phi)


        # select relevant targets

        targets_range = [t for t in targets if norm(t.position .- pos) < 200]

        for __ in 1:100
            samples = sample_event(10^logenergy, dir_theta, dir_phi, pos, targets_range, model, tf_dict; rng=rng)
            if use_grad
                logl_grad = collect(Zygote.gradient(
                    (logenergy, dir_theta, dir_phi) -> likelihood(logenergy, dir_theta, dir_phi, pos, samples, targets_range, model, tf_dict),
                    logenergy, dir_theta, dir_phi))

                push!(matrices, logl_grad .* logl_grad')
            else
                logl_hessian =  Zygote.hessian(
                    x -> likelihood(x[1], x[2], x[3], pos, samples, targets_range, model, tf_dict),
                    [logenergy, dir_theta, dir_phi])
                push!(matrices, .-logl_hessian)
            end
        end
    end

    return mean(matrices)
end


rng = MersenneTwister(31338)
f1 = calc_fisher(4, 0.1, 0.2, 1, targets, model; use_grad=false, rng=rng)
rng = MersenneTwister(31338)
f2 = calc_fisher(4, 0.1, 0.2, 1, targets, model; use_grad=true, rng=rng)

inv(f1)
inv(f2)

logenergies = 2:0.5:5

model_res = Dict()
for (mname, model_path) in models
    @load model_path model hparams opt tf_dict
    Flux.testmode!(model)

    sds= [calc_fisher(e, 0.1, 0.2, 50, model, use_grad=true) for e in logenergies]
    cov = inv.(sds)

    sampled_sds = []
    for c in cov

        cov_za = c[2:3, 2:3]
        dist = MvNormal([0.1, 0.2], 0.5 * (cov_za + cov_za'))
        rdirs = rand(dist, 10000)

        dangles = rad2deg.(acos.(dot.(sph_to_cart.(rdirs[1, :], rdirs[2, :]), Ref(sph_to_cart(0.1, 0.2)))))
        push!(sampled_sds, std(dangles))
    end

    model_res[mname] = sampled_sds
end

fig = Figure()
ax = Axis(fig[1, 1])
for (mname, res) in model_res
    CairoMakie.scatter!(ax, logenergies, Vector{Float64}(res))
end

fig


zazres =  (reduce(hcat, [sqrt.(v) for v in diag.(inv.(sds))])[2:3, :])

zazi_dist = MvNormal()







rad2deg.(acos.(dot.(sph_to_cart.(v[2, :], v[3, :]), Ref(sph_to_cart(0.1, 0.2))))


CairoMakie.scatter(logenergies, reduce(hcat, [sqrt.(v) for v in diag.(inv.(sds))])[1, :], axis=(yscale=log10, ))

CairoMakie.scatter(logenergies, rad2deg.(reduce(hcat, [sqrt.(v) for v in diag.(inv.(sds))])[2, :]))


logl_grad = Zygote.gradient(
    energy -> likelihood(energy, samples, [target], tf_dict),
    1E4)



#@show a

pmt_area = Float32((75e-3 / 2)^2 * π)
target_radius = 0.21f0


pos = SA[0.0f0, 10.0f0, 30.0f0]
dir_theta = deg2rad(50.0f0)
dir_phi = deg2rad(50.0f0)
dir = sph_to_cart(dir_theta, dir_phi)
p = Particle(pos, dir, 0.0f0, Float32(1E5), PEMinus)

target = MultiPMTDetector(
    @SVector[0.0f0, 0.0f0, 0.0f0],
    target_radius,
    pmt_area,
    make_pom_pmt_coordinates(Float32),
    UInt16(1)
)





begin


input = calc_flow_inputs([p], [target], tf_dict)
output = model.embedding(input)

flow_params = output[1:end-1, :]
log_expec = output[end, :]

expec = exp.(log_expec)
pois_expec = pois_rand.(expec)
mask = pois_expec .> 0
