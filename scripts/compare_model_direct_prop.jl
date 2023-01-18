using NeutrinoTelescopes
using CairoMakie
using Distributions
using Random
using BSON
using Flux
using StaticArrays
using DSP
using DataFrames
using Rotations

using LinearAlgebra


medium = make_cascadia_medium_properties(0.99f0)
pmt_area = Float32((75e-3 / 2)^2 * π)
target_radius = 0.21f0
target = MultiPMTDetector(
    @SVector[0.0f0, 0.0f0, 0.0f0],
    target_radius,
    pmt_area,
    make_pom_pmt_coordinates(Float32),
    UInt16(1)
)
wl_range = (300.0f0, 800.0f0)
spectrum = CherenkovSpectrum(wl_range, 30, medium)


dir = SA[0f0, 1f0, 0f0]
pos = SA[0f0, 0f0, 10f0]
ppos = pos .- 200 .* dir

energy = 1E5

particle = Particle(
    ppos,
    dir,
    0.0f0,
    Float32(energy),
    400.0f0,
    PMuMinus
)
source = CherenkovTrackEmitter(particle, medium, wl_range)

setup = PhotonPropSetup(source, target, medium, spectrum, 1)

photons = propagate_photons(setup)
calc_time_residual!(photons, setup)
calc_total_weight!(photons, setup)
orientation = RotMatrix3(I)
hits = make_hits_from_photons(photons, setup, orientation)

fig = Figure(resolution=(1500, 1000))
ga = fig[1, 1] = GridLayout(4, 4)

for i in 1:16
    row, col = divrem(i - 1, 4)
    mask = hits[:, :pmt_id] .== i
    ax = Axis(ga[col+1, row+1], xlabel="Time Residual(ns)", ylabel="Photons / time", title="PMT $i",
              )
    hist!(ax, hits[mask, :tres], bins=-50:3:150, weights=hits[mask, :total_weight], color=:orange, normalization=:density,)
end
fig



prop_source_ext = ExtendedCherenkovEmitter(particle, medium, (300f0, 800f0))
prop_source_che = PointlikeCherenkovEmitter(particle, medium, (300f0, 800f0))
prop_source_iso = PointlikeIsotropicEmitter(particle.position, particle.time, Int64(1E11), medium, (300f0, 800f0))

results_che, nph_sim_che = propagate_source(prop_source_che, distance, medium)
results_ext, nph_sim_ext = propagate_source(prop_source_ext, distance, medium)
results_iso, nph_sim_iso = propagate_source(prop_source_iso, distance, medium)


dir_weight_new = get_dir_reweight.(results_iso[:, :initial_directions], Ref(particle.direction), results_iso[:, :ref_ix])


histogram(results_che[:, :tres],  weights=results_che[:, :total_weight], xlim=(-10, 100), yscale=:log10, ylim=(1E-5, 1E1))


directions = [PhotonPropagation.PhotonPropagationCuda.sample_cherenkov_direction(prop_source_che, medium, 350f0) for _ in 1:10000]

scatter([dir[1] for dir in directions], [dir[2] for dir in directions], [dir[3] for dir in directions], ms=0.3)

plot!([0, 2*particle.direction[1]], [0, 2*particle.direction[2]], [0, 2*particle.direction[3]], lw=5)

data = BSON.load(joinpath(@__DIR__, "../assets/photon_model.bson"), @__MODULE__)
model = data[:model] |> gpu
output_trafos = [:log, :log, :neg_log_scale]
model_params, sources, mask, distances = evaluate_model(targets, particle, medium, 0.5f0, model, output_trafos)

poissons = poisson_dist_per_module(model_params, sources, mask)
shapes = shape_mixture_per_module(model_params, sources, mask, distances, medium)

event = sample_event(poissons, shapes, sources)

histogram(event)

prop_source_ext = ExtendedCherenkovEmitter(particle, medium, (300f0, 800f0))
prop_source_che = PointlikeCherenkovEmitter(particle, medium, (300f0, 800f0))

single_source = CherenkovSegment(prop_source_che.position, prop_source_che.direction, prop_source_che.time, Float32(prop_source_che.photons))

model_params_single = source_to_input(single_source, target)
model_pred = cpu(model(gpu(reshape(collect(model_params_single), (2, 1)))))
transform_model_output!(model_pred, output_trafos)
model_pred = reshape(model_pred, (3, 1, 1))


shape_single = shape_mixture_per_module(model_pred, [single_source] , reshape([true], (1, 1)), reshape([distance], (1, 1)), medium)
poisson_single = poisson_dist_per_module(model_pred, [single_source],  reshape([true], (1, 1)))


shape_single

event_single = sample_event(poisson_single, shape_single, [single_source])


results_che, nph_sim_che = ppcu.propagate_source(prop_source_che, distance, medium)
results_ext, nph_sim_ext = ppcu.propagate_source(prop_source_ext, distance, medium)

results_che[!, :pmt_acc_weight] = p_one_pmt_acc.(results_che[:, :wavelength])
results_che[!, :total_weight] = results_che[:, :abs_weight] .* results_che[!, :pmt_acc_weight]


c_vac = ustrip(u"m/ns", SpeedOfLightInVacuum)
tgeo = (distance - target_radius) ./ (c_vac / get_refractive_index(800.0f0, medium))
tres = (event_single[1] .- tgeo .- prop_source_ext.time)

histogram(tres, bins=-50:1:50, alpha=0.7)
histogram!(results_che[:, :tres], weights=results_che[:, :total_weight],  bins=-50:1:50, alpha=0.7)

histogram(results_che[:, :tres], weights=results_che[:, :total_weight],  bins=0:1:100, yscale=:log10)


scatter(results_che[1:1000, :tres], results_che[1:1000, :pmt_acc_weight], xlim=(0, 100))


typeof(results_ext)

results_ext


@show cos(results_iso[1, :directions][3] - dot(src_targ, prop_source.direction))

results_iso[1, :directions]

dir_weight_new = get_dir_reweight_new.(results_iso[:, :directions], Ref(src_targ), results_iso[:, :ref_ix])

results_iso[!, :norm_weights] = 
results_iso[!, :norm_weights] ./= sum(results_iso[!, :norm_weights])

results_che[!, :norm_weights] = results_che[:, :total_weight] / sum(results_che[:, :total_weight])


all(isfinite.(dir_weight_new))

histogram(cos.(results_che[:, :thetas]), weights=results_che[:, :total_weight],
 bins = -1:0.1:1,   normalize=true)
histogram!(cos.(results_iso[:, :thetas]), weights=results_iso[:, :total_weight].*results_iso[:, :dir_weight],
 bins = -1:0.1:1, legend=:topleft,  normalize=true)
histogram!(cos.(results_iso[:, :thetas]), weights=results_iso[:, :total_weight].*dir_weight_new,
 bins = -1:0.1:1, legend=:topleft, normalize=true)


histogram(results_iso[:, :times], weights=results_iso[:, :total_weight].*results_iso[:, :dir_weight],
bins = 280:1:360)
histogram!(results_che[:, :times], weights=results_che[:, :total_weight],
 bins = 280:1:360)
histogram!(results_iso[:, :times], weights=results_iso[:, :total_weight].*dir_weight_new,
bins = 280:1:360)

times


test = [0, 0, 0, 10, 10, 10]
weights = [10, 10, 10, 1, 1, 1]

histogram(test, weights=weights, bins=0:11)



sources

minimum(tres)

plot([source.time for source in sources], [source.photons for source in sources])


long_param = prop_source.long_param

scale = (1 / long_param.b)
shape = (long_param.a)

medium64 = make_cascadia_medium_properties(Float64)
long_param = LongitudinalParameterisation(1E5, medium64, get_longitudinal_params(PEMinus))


states = [ ppcu.initialize_photon_state(prop_source, medium) for _ in 1:100000]

histogram([source.time for source in states], normalize=true)
plot!([source.time for source in sources], [source.photons for source in sources] ./ 4E9)


histogram([source.position[3] for source in states], normalize=true)
plot!([source.position[3] for source in sources], [source.photons for source in sources] ./ 1E9)

track_dir = sample_cherenkov_track_direction(Float32)
dir_rot = ppcu.rotate_to_axis(prop_source.direction, track_dir)

prop_source.direction

sources[1].position
sources[1].time

sources[1].position[3] / sources[1].time 
states[1].position[3] / states[1].time 
