using PhotonPropagation
using NeutrinoTelescopes
using PhysicsTools
using StaticArrays
using LinearAlgebra
using Random
using Flux
using BSON: @load
using HDF5
using DataFrames
using CairoMakie
using StatsBase
using Distributions
using ArraysOfArrays


function poisson_confidence_interval(k, alpha=0.32)
    if k == 0
       low = 0
    else
        low = 0.5*quantile(Chisq(2*k), alpha/2)
    end
    high = 0.5*quantile(Chisq(2*k+2), 1-alpha/2)
    return [low, high]
end


workdir = ENV["ECAPSTOR"]

model = PhotonSurrogate(
    joinpath(workdir, "snakemake/time_surrogate_perturb/extended/amplitude_1_FNL.bson"),
    joinpath(workdir, "snakemake/time_surrogate_perturb/extended/time_uncert_0_1_FNL.bson")
)
   

targets_single = [POM(@SVector[-25.0, 0.0, -450.0], 1)]
targets_line = make_detector_line(@SVector[-25.0, 0.0, 0.0], 20, 50, 1)
targets_three_l = [
    make_detector_line(@SVector[-25.0, 0.0, 0.0], 20, 50, 1)
    make_detector_line(@SVector[25.0, 0.0, 0.0], 20, 50, 21)
    make_detector_line(@SVector[0.0, sqrt(50^2 - 25^2), 0.0], 20, 50, 41)]
targets_hex = make_hex_detector(3, 50, 20, 50, truncate=1)

pos = SA[-10.0, 1., -460]
dir_theta = deg2rad(110)
dir_phi = deg2rad(50)
dir = sph_to_cart(dir_theta, dir_phi)
energy = 1e5

particles = [
    Particle(pos, dir, 0.0, energy, 0., PEPlus),
]

hbc, hbg = make_hit_buffers()

medium = make_cascadia_medium_properties(0.95f0, 1f0, 1f0)
hits = propagate_particles(particles, targets_single, 1, medium, hbc, hbg)
figs = compare_mc_model(particles, targets_single, Dict("casc" => model, ), medium, hits, noise_rate=1E6)

figs[1]


abs_scales = 0.95:0.02:1.1
sca_scales = 0.95:0.02:1.1


all_hits = []
for sca_scale in sca_scales
    for i in 1:5
        medium = make_cascadia_medium_properties(0.95f0, 1f0, Float32(sca_scale))
        hits = propagate_particles(particles, targets_single, 1, medium, hbc, hbg)
        hits[!, :sca_scale] .= sca_scale
        hits[!, :prop_id] .= i
        push!(all_hits, hits)
    end
end
all_hits = reduce(vcat, all_hits)


bins = -10:10:150
for (grpkey, group) in pairs(groupby(all_hits, :sca_scale))
    hist!(ax, group[:, :time], weights=group[:, :total_weight], bins=bins)
end
fig

combine(groupby(all_hits, [:sca_scale, :prop_id]),:total_weight => sum)

summed_hits = combine(groupby(combine(groupby(all_hits, [:sca_scale, :prop_id]), :total_weight => sum), :sca_scale), :total_weight_sum => median, :total_weight_sum => iqr)

model = gpu(model)
feat_buffer = create_input_buffer(model, 16, 1)

expec_model = DataFrame()
for sca_scale in sca_scales
    log_expec_per_pmt, _ = get_log_amplitudes(particles, targets_single, model; feat_buffer=feat_buffer, device=gpu, abs_scale=1., sca_scale=sca_scale)
    total_expec = sum(exp.(log_expec_per_pmt))
    push!(expec_model, (sca_scale=sca_scale, total_expec=total_expec))
end
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Scattering Length Scale", ylabel="Total Number of Hits")
colors = Makie.wong_colors()
errorbars!(
    ax,
    summed_hits[:, :sca_scale],
    summed_hits[:, :total_weight_sum_median],
    summed_hits[:, :total_weight_sum_iqr],
    color=colors[1],
    whiskerwidth = 10)
scatter!(ax, summed_hits[:, :sca_scale], summed_hits[:, :total_weight_sum_median], color=colors[1], label="MC Photon Prop.")
lines!(ax, expec_model[:, :sca_scale], expec_model[:, :total_expec], color=colors[2], label="Model Prediction")
axislegend(position=:rb)
fig

