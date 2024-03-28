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


workdir = ENV["ECAPSTOR"]

model = PhotonSurrogate(
    joinpath(workdir, "snakemake/time_surrogate_perturb/lightsabre/amplitude_1_FNL.bson"),
    joinpath(workdir, "snakemake/time_surrogate_perturb/lightsabre/time_uncert_0_1_FNL.bson")
)
    


targets_single = [POM(@SVector[-25.0, 0.0, -450.0], 1)]
targets_line = make_detector_line(@SVector[-25.0, 0.0, 0.0], 20, 50, 1)
targets_three_l = [
    make_detector_line(@SVector[-25.0, 0.0, 0.0], 20, 50, 1)
    make_detector_line(@SVector[25.0, 0.0, 0.0], 20, 50, 21)
    make_detector_line(@SVector[0.0, sqrt(50^2 - 25^2), 0.0], 20, 50, 41)]
targets_hex = make_hex_detector(3, 50, 20, 50, truncate=1)

abs_scale = 0.95f0
sca_scale = 1.1f0

medium = make_cascadia_medium_properties(0.95, abs_scale, sca_scale)
d = LineDetector(targets_hex, medium)


cylinder = get_bounding_cylinder(d)
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
