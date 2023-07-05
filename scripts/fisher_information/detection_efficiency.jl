using NeutrinoTelescopes
using Sobol
using Distributions
using PhotonPropagation
using PhysicsTools
using DataStructures
using PreallocationTools
using LinearAlgebra
using Flux
using PoissonRandom
using CairoMakie
using DataFrames
using StaticArrays
using Optim
using Formatting
using Polynomials
args = Dict("type" => "track")

model_path = joinpath(ENV["WORK"], "time_surrogate")
models_casc = Dict(
    "A1S1" =>  PhotonSurrogate(joinpath(model_path, "extended/amplitude_1_FNL.bson"), joinpath(model_path, "extended/time_1_FNL.bson")),
    "A2S1" =>  PhotonSurrogate(joinpath(model_path, "extended/amplitude_2_FNL.bson"), joinpath(model_path, "extended/time_1_FNL.bson")),
    "A1S2" =>  PhotonSurrogate(joinpath(model_path, "extended/amplitude_1_FNL.bson"), joinpath(model_path, "extended/time_2_FNL.bson")),
    "A2S2" =>  PhotonSurrogate(joinpath(model_path, "extended/amplitude_2_FNL.bson"), joinpath(model_path, "extended/time_2_FNL.bson")),

)


models_tracks = Dict(
    "A1S1" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_1_FNL.bson"), joinpath(model_path, "lightsabre/time_1_FNL.bson")),
    "A2S1" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_2_FNL.bson"), joinpath(model_path, "lightsabre/time_1_FNL.bson")),
    "A1S2" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_1_FNL.bson"), joinpath(model_path, "lightsabre/time_2_FNL.bson")),
    "A2S2" =>  PhotonSurrogate(joinpath(model_path, "lightsabre/amplitude_2_FNL.bson"), joinpath(model_path, "lightsabre/time_2_FNL.bson")),

)

medium = make_cascadia_medium_properties(0.95f0)

pdist = nothing
ang_dist = nothing
length_dist = nothing

if args["type"] == "track"
    pdist = CategoricalSetDistribution(OrderedSet([PMuPlus, PMuMinus]), [0.5, 0.5])
    ang_dist = LowerHalfSphere()
    length_dist = Dirac(10000.)
else
    pdist = CategoricalSetDistribution(OrderedSet([PEMinus, PEPlus]), [0.5, 0.5])
    ang_dist = UniformAngularDistribution()
    length_dist = Dirac(0.)
end

model = models_tracks["A1S1"]

time_dist = Dirac(0.0)
logenergies = 2:0.5:5
spacings = SobolSeq(30, 200)
#targets_hex = make_hex_detector(3, spacing, 20, 50, truncate=1)

length(logenergies)

eff_d = []
for i in 1:10

    spacing = next!(spacings)[1]

    targets_cluster = make_n_hex_cluster_detector(7, spacing, 20, 50)

    d = Detector(targets_cluster, medium)
    hit_buffer = create_input_buffer(d, 1)
    cylinder = get_bounding_cylinder(d)
    surface = CylinderSurface(cylinder)

    buffer = (create_input_buffer(d, 1))
    diff_cache = FixedSizeDiffCache(buffer, 6)

    modules = get_detector_modules(d)
    medium = get_detector_medium(d)

    for le in logenergies
        edist = Dirac(10^le)
        if args["type"] == "track"
            inj = SurfaceInjector(surface, edist, pdist, ang_dist, length_dist, time_dist)
        else
            inj = VolumeInjector(cylinder, edist, pdist, ang_dist, length_dist, time_dist)
        end

        max_distance = 200
        events_passed = []

        
        n_total = 100
        for i in 1:n_total
            ev = rand(inj)
            isec = get_intersection(cylinder, ev[:particles][1])

            length_in = isec.second - isec.first

            particles = ev[:particles]
            modules_range_mask = get_modules_in_range(particles, d, 200)
            modules_range = (modules[modules_range_mask])
            # npmt, 1, ntargets
            log_exp_per_pmt, _ = get_log_amplitudes(particles, modules_range, gpu(model); feat_buffer=hit_buffer)
            
            exp_per_module = sum(exp.(log_exp_per_pmt), dims=1)

            n_mod_thrsh = sum(any(exp_per_module .>= 2, dims=1))
            n_total = sum(exp.(log_exp_per_pmt))
            theta, phi = cart_to_sph(particles[1].direction)
            push!(eff_d, (n_mod_thrsh=n_mod_thrsh, dir_theta=theta, dir_phi=phi, length=length_in, log_energy=le, spacing=spacing, n_total))
        end
    end
end
    

df = DataFrame(eff_d)

dfc = combine(groupby(df, :spacing), :length => mean)

scatter(dfc[:, :spacing], dfc[:, :length_mean])



df_comb = combine(groupby(df, [:spacing, :log_energy]), :n_total => mean, :n_mod_thrsh => median, :n_mod_thrsh => (v -> quantile(v, 0.9)) => :n_mod_thrsh_90)


log_energies = [3, 4, 5]

fig = Figure()
ax = Axis(fig[1, 1], yscale=log10, xlabel="Spacing (m)", ylabel="Mean Total Hits")
for (e, col) in zip(log_energies, Makie.wong_colors())
    dfs = sort(df_comb[df_comb[:, :log_energy] .== e, :], :spacing)
    scatter!(ax, dfs[:, :spacing], dfs[:, :n_total_mean], label=format("{:.0d} TeV", 10^e/1000), color=col)
    poly = Polynomials.fit(log.(dfs[:, :spacing]), log.(dfs[:, :n_total_mean]), 1)

    @show poly
    low, hi = extrema(dfs[:, :spacing])
    xs = low:1:hi
    lines!(ax, xs, exp.(poly.(log.(xs))))


end
axislegend("Energy")
fig





fig = Figure()
ax = Axis(fig[1, 1], xlabel="Spacing (m)", ylabel="Median Modules with >= 2 Hits", yscale=Makie.pseudolog10,
yminorticksvisible=true, yminorticks=IntervalsBetween(10))
for (e, col) in zip(log_energies, Makie.wong_colors())
    dfs = sort(df_comb[df_comb[:, :log_energy] .== e, :], :spacing)
    scatter!(ax, dfs[:, :spacing], dfs[:, :n_mod_thrsh_median],  label=format("{:.0d} TeV", 10^e/1000), color=col)
    poly = Polynomials.fit(log.(dfs[:, :spacing]), log.(dfs[:, :n_mod_thrsh_median]), 1)

    @show poly
    low, hi = extrema(dfs[:, :spacing])
    xs = low:1:hi
    lines!(ax, xs, exp.(poly.(log.(xs))))
    #scatter!(ax, dfs[:, :spacing], dfs[:, :n_mod_thrsh_90],  label=format("{:.0d} TeV", 10^e/1000), marker='🐱', color=col)
end
fig

unique( df[:, :spacing])


y1 = 50
x1 = 43

y2 = 11
x2 = 175

x0 = (sqrt(y1/y2) * x1 - x2) / (sqrt(y1/y2) -1 )
b = y1*(x1-x0)^2

f(x, b, x0) = b / (x - x0)^2


xs = 40:1.:200
lines!(ax, xs, f.(xs, b, x0))
axislegend("Energy", merge=true)
fig



Sn_thrsh = 3:8
effs = [sum(df[:, :n_mod_thrsh] .>= n) / n_total for n in n_thrsh]

scatter(n_thrsh, effs)



hist(cos.(df[df[:, :n_mod_thrsh] .> 6, :dir_theta]), normalization=:probability)

sum(any((exp.(log_exp)) .>= 2, dims=1))


for le in logenergies
    edist = Dirac(10^le)
    inj = VolumeInjector(cylinder, edist, pdist, ang_dist, length_dist, time_dist)
end


p = Particle(SA[0., 0., 0.], SA[0., 0., 1.], 0., 1E3, 2000., PMuMinus)

energies = []
for i in 1:100
    p_prop, losses = propagate_muon(p)

    push!(energies, p_prop.energy)
end

mean(energies)