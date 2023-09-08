using NeutrinoTelescopes
using PhotonPropagation
using PhysicsTools
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
using DataStructures

using SpecialFunctions
using StatsBase
using Base.Iterators
using Distributions
using Optim
using QuadGK
using Base.Iterators
using Formatting
using FiniteDifferences
using ForwardDiff
using PreallocationTools
using BenchmarkTools

model_path = joinpath(ENV["WORK"], "time_surrogate")

models_casc = Dict(
    "A1T1" =>  PhotonSurrogate(joinpath(model_path, "extended/amplitude_1_FNL.bson"), joinpath(model_path, "extended/time_1_FNL.bson")),
    "A2T1" =>  PhotonSurrogate(joinpath(model_path, "extended/amplitude_2_FNL.bson"), joinpath(model_path, "extended/time_1_FNL.bson")),
    "A1T2" =>  PhotonSurrogate(joinpath(model_path, "extended/amplitude_1_FNL.bson"), joinpath(model_path, "extended/time_2_FNL.bson")),
    "A2T2" =>  PhotonSurrogate(joinpath(model_path, "extended/amplitude_2_FNL.bson"), joinpath(model_path, "extended/time_2_FNL.bson")),

)

models_tracks = Dict(
    "A1T1" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_1_FNL.bson"), joinpath(model_path, "lightsabre/time_1_FNL.bson")),
    "A1TU1.5_1" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_1_FNL.bson"), joinpath(model_path, "lightsabre/time_uncert_1.5_1_FNL.bson")),
    "A1TU2.5_1" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_1_FNL.bson"), joinpath(model_path, "lightsabre/time_uncert_2.5_1_FNL.bson")),
)


targets_single = [POM(@SVector[-25.0, 0.0, -450.0], 1)]
targets_line = make_detector_line(@SVector[-25.0, 0.0, 0.0], 20, 50, 1)
targets_three_l = [
    make_detector_line(@SVector[-25.0, 0.0, 0.0], 20, 50, 1)
    make_detector_line(@SVector[25.0, 0.0, 0.0], 20, 50, 21)
    make_detector_line(@SVector[0.0, sqrt(50^2 - 25^2), 0.0], 20, 50, 41)]
targets_hex = make_hex_detector(3, 50, 20, 50, truncate=1)

targets_full = make_n_hex_cluster_detector(7, 70, 20, 50)
detectors = Dict("Single" => targets_single, "Line" =>targets_line, "Tri" => targets_three_l, "Hex" => targets_hex, "full" => targets_full)
medium = make_cascadia_medium_properties(0.95f0)


model = models_tracks["A1TU2.5_1"]
model = gpu(model)


targets = targets_full
model_path = joinpath(ENV["WORK"], "time_surrogate")
model = models_tracks["A1TU2.5_1"]
model = gpu(model)

c_n = group_velocity(800.0f0, medium)
rng = MersenneTwister(31338)

pos = SA[-25.0, 5.0, -460]
dir_theta = 0.4
dir_phi = 0.3
dir = sph_to_cart(dir_theta, dir_phi)
log_energy = 4.

p = Particle(pos, dir, 0.0, 10^log_energy, 0.0, PMuMinus)
rng = MersenneTwister(31338)

range_mask = get_modules_in_range([p], targets, 200)
targets_range = targets[range_mask]

samples = sample_multi_particle_event([p], targets_range, model, medium)



f, fwrapped = make_lh_func(time=0., data=samples, targets=targets_range, model=model, medium=medium, diff_cache=nothing, ptype=p.type)
g(dir_theta) = f(log_energy, dir_theta, dir_phi, pos[1], pos[2], pos[3])
dir_thetas = 0:0.01:0.6
log_energies = 3:0.05:5
llhs = g.(dir_thetas)
plot(dir_thetas, llhs)

logl_grad = collect(ForwardDiff.gradient(fwrapped, [log_energy, dir_theta, dir_phi, pos[1], pos[2], pos[3]]))

diffquot(f, ϵ, x0) = (f(x0+ϵ/2) .- f(x0-ϵ/2) ) ./ ϵ
epss = -7:0.1:-2

#logl_grad_b = collect(Zygote.gradient(g, dir_theta))
logl_grad = collect(ForwardDiff.gradient(p -> g(p[1]), [dir_theta]))#

fig, ax = plot(epss, diffquot.(g, 10 .^epss, dir_theta))
#hlines!(ax, logl_grad_b[1])
hlines!(ax, logl_grad[1])
fig


d = Detector(targets_hex, medium)
cylinder = get_bounding_cylinder(d)
surf = CylinderSurface(cylinder)
pdist = CategoricalSetDistribution(OrderedSet([PEMinus, PEPlus]), [0.5, 0.5])
#edist = Pareto(1, 1E4) + 1E4
edist = Dirac(1E4)
ang_dist = LowerHalfSphere()
length_dist = Dirac(0.0)
time_dist = Dirac(0.0)
inj = SurfaceInjector(surf, edist, pdist, ang_dist, length_dist, time_dist)
buffer = (create_input_buffer(d, 1))
diff_cache = FixedSizeDiffCache(buffer, 6)


nev = 1
nsa = 35

model = gpu(models_tracks["A1T1"])
hit_generator = SurrogateModelHitGenerator(model, 200.0, nothing)

m, evts = calc_fisher(d, inj, hit_generator, nev, nsa, use_grad=true, cache=diff_cache)

function profiled_pos(fisher::Matrix)
    fisher
    IA = fisher[1:3, 1:3]
    IC = fisher[4:6, 1:3]
    ICprime = fisher[1:3, 4:6]
    IB = fisher[4:6, 4:6]
    profiled = IA - ICprime*inv(IB)*IC
    return profiled
end

ang_errs = Float64[]
for (mat, e) in zip(m, evts)
    cov_za = inv(mat)[2:3, 2:3]
    cov_za = 0.5 * (cov_za .+ cov_za')
    dir_cart = first(e[:particles]).direction
    dir_sp = cart_to_sph(dir_cart)
    dist = MvNormal(dir_sp, cov_za)
    rdirs = rand(dist, 1000)
    dangles = rad2deg.(acos.(dot.(sph_to_cart.(rdirs[1, :], rdirs[2, :]), Ref(dir_cart))))
    push!(ang_errs, mean(dangles))
end


ang_errs_p = Float64[]
for (mat, e) in zip(m, evts)
    p = profiled_pos(mat)
    cov_za = inv(p)[2:3, 2:3]
    cov_za = 0.5 * (cov_za .+ cov_za')
    dir_cart = first(e[:particles]).direction
    dir_sp = cart_to_sph(dir_cart)
    dist = MvNormal(dir_sp, cov_za)
    rdirs = rand(dist, 1000)
    dangles = rad2deg.(acos.(dot.(sph_to_cart.(rdirs[1, :], rdirs[2, :]), Ref(dir_cart))))
    push!(ang_errs_p, mean(dangles))
end

fig, ax, h = hist(ang_errs, normalization=:pdf)
hist!(ax, ang_errs_p, normalization=:pdf)
fig


d = fit(Rayleigh, ang_errs_p)

plot!(ax, d)
fig

median(ang_errs_p)
percentile(ang_errs_p, 32)
percentile(ang_errs_p, 68)


models_uncert = Dict(0 => models_tracks["A1T1"], 1.5 => models_tracks["A1TU1.5_1"], 2.5 => models_tracks["A1TU2.5_1"])
event = rand(inj)





hit_generator = SurrogateModelHitGenerator(model, 200.0, nothing)
fi, matrices = calc_fisher_matrix(event, d, hit_generator, use_grad=true, n_samples=100, cache=diff_cache)
fi2, matrices2 = calc_fisher_matrix(event, d, hit_generator, use_grad=true, n_samples=1000, cache=diff_cache)
fi3, matrices3 = calc_fisher_matrix(event, d, hit_generator, use_grad=true, n_samples=5000, cache=diff_cache)


mean(tr.(matrices))
mean(tr.(matrices2))
mean(tr.(matrices3))



fig, ax, h = hist(log10.(tr.(matrices)), normalization=:pdf)
hist!(ax, log10.(tr.(matrices2)), normalization=:pdf)
hist!(ax, log10.(tr.(matrices3)), normalization=:pdf)
fig


sstats = []
for ns in [10, 20, 30, 50, 70, 100, 200, 500]
    for i in 1:10
        for (uncert, model) in models_uncert
            model = gpu(model)
            hit_generator = SurrogateModelHitGenerator(model, 200.0, nothing)
            fi, matrices = calc_fisher_matrix(event, d, hit_generator, use_grad=true, n_samples=ns, cache=diff_cache)
            push!(sstats, (cov=inv(fi), det=det(inv(fi)), ns=ns, uncert=uncert, i=i))
        end
    end
end

df = DataFrame(sstats)
df[!, :trace] .= tr.(df[:, :cov])
grouped = groupby(df, :uncert)

fig = Figure()
ax = Axis(fig[1, 1], xscale=log10, yscale=log10)

for (gname, group) in pairs(grouped)
    #scatter!(ax, group[:, :ns], log10.(group[:, :det]), label=string(first(gname)))
    scatter!(ax, group[:, :ns], group[:, :trace], label=string(first(gname)))
end
axislegend("Timing Uncert.")
fig


fig = Figure()
ax = Axis(fig[1, 1])
for (uncert, model) in models_uncert
    model = gpu(model)

    hit_generator = SurrogateModelHitGenerator(model, 200.0, nothing)
    m, _ = calc_fisher_matrix(event, d, hit_generator, use_grad=true, n_samples=150, cache=diff_cache)
    sigmas = sqrt.(diag(inv(m)))
    @show uncert, det(inv(m))
    #=
    hit_generator = SurrogateModelHitGenerator(model, 200.0, nothing)
    m, events = calc_fisher(d, inj, hit_generator, 50, 35, use_grad=true, cache=diff_cache)
    m = mean(m)
    sigmas = sqrt.(diag(inv(m)))
    =#
    scatter!(ax, sigmas, label="$uncert ns")
end
axislegend("Timing Uncert.")
fig

buffer = (create_input_buffer(d, 1))
diff_cache = FixedSizeDiffCache(buffer, 6)

event = rand(inj)
@time  calc_fisher_matrix(event, d, hit_generator, use_grad=true, n_samples=35, cache=diff_cache)
@time  calc_fisher_matrix(event, d, hit_generator, use_grad=true, n_samples=35)


@time calc_fisher(d, inj, hit_generator, 10, 35)

@btime begin
    for i in 1:10
        event = rand(inj)
        calc_fisher_matrix(event, d, hit_generator, use_grad=true, n_samples=35, cache=diff_cache)
    end
end

@btime m, events = calc_fisher(d, inj, hit_generator, 10, 35, use_grad=true, cache=diff_cache)


@btime begin
    for i in 1:10
        calc_fisher_matrix(event, d, hit_generator, use_grad=true, n_samples=35)
    end
end


covariances = Dict()
for (mname, model) in models_casc

    model = gpu(model)
    hit_generator = SurrogateModelHitGenerator(model, 200.0, nothing)
    sds = []
    for le in logenergies
        edist = Dirac(10^le)
        inj = VolumeInjector(cylinder, edist, pdist, ang_dist, length_dist, time_dist)
        
        f = calc_fisher(d, inj, hit_generator, 10, 20, use_grad=true)
        push!(sds, f)
    end

    cov = inv.(sds)

    covariances[mname] = cov
end

model_res = Dict()
for (mname, cov) in covariances
    sampled_sds = []
    for c in cov

        cov_za = c[2:3, 2:3]
        cov_za = 0.5 * (cov_za .+ cov_za')
        @show cov_za
        dist = MvNormal([0.1, 0.2], cov_za)
        rdirs = rand(dist, 10000)

        dangles = rad2deg.(acos.(dot.(sph_to_cart.(rdirs[1, :], rdirs[2, :]), Ref(sph_to_cart(0.1, 0.2)))))
        push!(sampled_sds, std(dangles))
    end

    model_res[mname] = sampled_sds
end

fig = Figure()
ax = Axis(fig[1, 1], yscale=log10)
for (mname, res) in model_res
    CairoMakie.scatter!(ax, logenergies, Vector{Float64}(res), label=mname)
end
axislegend(ax)
fig

keys(models_casc)


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
