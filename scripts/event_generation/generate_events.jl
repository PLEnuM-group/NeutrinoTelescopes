using NeutrinoTelescopes
using PhotonPropagation
using PhysicsTools
using PMTSimulation
using CairoMakie
using Distributions
using Random
using BSON
using Flux
using StaticArrays
using DataStructures
using JSON3
using DataFrames
using Arrow
using JLD2
using CUDA


workdir = ENV["ECAPSTOR"]

model = PhotonSurrogate(
    joinpath(workdir, "snakemake/time_surrogate_perturb/lightsabre/amplitude_1_FNL.bson"),
    joinpath(workdir, "snakemake/time_surrogate_perturb/lightsabre/time_uncert_0_1_FNL.bson")
)
model = gpu(model)

targets_hex = make_n_hex_cluster_detector(7, 50, 20, 50)

abs_scale = 1f0
sca_scale = 1f0
medium = make_cascadia_medium_properties(0.95f0, abs_scale, sca_scale)
d = LineDetector(targets_hex, medium)
cylinder = get_bounding_cylinder(d)
hit_generator = SurrogateModelHitGenerator(model, 200.0, d)

li_file = "/home/wecapstor3/capn/capn100h/snakemake/leptoninjector-lightsabre-0.hd5"

injector = LIInjector(li_file, drop_starting=false)
event = rand(injector)

muon = event[:particles][1]
muon_propagated, losses = propagate_muon(muon)

muon

muon_propagated
muon
stochastic_losses = [loss for loss in losses if loss.energy > 50 ]
continuous_losses = [loss for loss in losses if loss.energy <= 50 ]
continuous_energy = sum(loss.energy for loss in continuous_losses)

lightsabre = Particle(muon.position, muon.direction, muon.time, continuous_energy, 1E4, muon.type)
any(get_modules_in_range([lightsabre; stochastic_losses], d, hit_generator.max_valid_distance))


hits, mask = generate_hit_times([lightsabre; stochastic_losses], d, hit_generator; abs_scale=abs_scale, sca_scale=sca_scale)
hit_list_to_dataframe(hits, get_detector_modules(d), mask)










#pdist = CategoricalSetDistribution(OrderedSet([PEMinus, PEPlus]), [0.5, 0.5])
pdist = CategoricalSetDistribution(OrderedSet([PMuPlus, PMuMinus]), [0.5, 0.5])
edist = Pareto(1, 1E4) + 1E4
#ang_dist = UniformAngularDistribution()
ang_dist = LowerHalfSphere()
length_dist = Dirac(0.0)
time_dist = Dirac(0.0)
#inj = VolumeInjector(cylinder, edist, pdist, ang_dist, length_dist, time_dist)
inj = SurfaceInjector(CylinderSurface(cylinder), edist, pdist, ang_dist, length_dist, time_dist)
model = gpu(model)
hit_generator = SurrogateModelHitGenerator(model, 200.0, d)



events = Event[]
event_hits = []
for _ in 1:10
    event = rand(inj)
    hits = generate_hit_times!(event, d, hit_generator)
    push!(events, event)

    ev_d = Dict(:hits => event[:photon_hits], :mc_truth=>JSON3.write(event[:particles][1]))

    push!(event_hits, ev_d)
end

events


save(joinpath(workdir, "test_muons.jld2"), Dict("event_hits" => event_hits, "geo" => get_detector_pmts(d)))

data = load(joinpath(workdir, "test_cascades.jld2"))

data["event_hits"]

event
groups = groupby(hits, [:pmt_id, :module_id])

function unfold_per_module(hits)
    min_time, max_time = extrema(hits[:, :time])

    min_time = div(min_time-20, 1/STD_PMT_CONFIG.sampling_freq) * 1/STD_PMT_CONFIG.sampling_freq
    max_time = div(max_time+20, 1/STD_PMT_CONFIG.sampling_freq) * 1/STD_PMT_CONFIG.sampling_freq

    per_pmt_pulses = make_reco_pulses(hits, STD_PMT_CONFIG, (min_time, max_time))
    return per_pmt_pulses
end

pulsesmap = combine(groups, unfold_per_module)[:, :x1]

bins = 10 .^(-1:0.2:3)

fig, ax = hist(combine(groups, nrow)[:, :nrow], bins=bins, 
    axis=(;xscale=log10, yscale=log10, limits=(0.1, 1000, 0.1, 500)), fillto=0.1)
hist!(ax, (get_total_charge.(pulsesmap)), bins=bins, fillto=0.1)

fig

event
push!(ec, event)

Base.iterate(e::EventCollection) = iterate(e.events)

open("test.arrow", "w") do io

    for e in ec
        record = (event_id=[e.id], times=[e[:photon_hits][:, :time]])
        Arrow.write(io, record)

    end
end
JSON3.write(ec)
