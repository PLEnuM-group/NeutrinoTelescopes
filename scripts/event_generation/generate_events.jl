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
using Unitful

using PhysicalConstants.CODATA2018: c_0

c_vac = ustrip(u"m/ns", c_0)




workdir = ENV["ECAPSTOR"]

model_lightsabre = PhotonSurrogate(
    joinpath(workdir, "snakemake/time_surrogate_perturb/lightsabre/amplitude_2_FNL.bson"),
    joinpath(workdir, "snakemake/time_surrogate_perturb/lightsabre/time_uncert_0_1_FNL.bson")
)
model_lightsabre = gpu(model_lightsabre)

model_extended = PhotonSurrogate(
    joinpath(workdir, "snakemake/time_surrogate_perturb/extended/amplitude_2_FNL.bson"),
    joinpath(workdir, "snakemake/time_surrogate_perturb/extended/time_uncert_0_1_FNL.bson")
)
model_extended = gpu(model_extended)

targets_hex = make_n_hex_cluster_detector(7, 50, 20, 50)

abs_scale = 1f0
sca_scale = 1f0
medium = make_cascadia_medium_properties(0.95f0, abs_scale, sca_scale)
d = LineDetector(targets_hex, medium)
cylinder = get_bounding_cylinder(d)
hit_generator_lightsabre = SurrogateModelHitGenerator(model_lightsabre, 200.0, d)
hit_generator_extended = SurrogateModelHitGenerator(model_extended, 200.0, d)
li_file = "/home/wecapstor3/capn/capn100h/snakemake/leptoninjector-lightsabre-0.hd5"

injector = LIInjector(li_file, drop_starting=false)
prop_mu_minus = ProposalInterface.make_propagator(PMuMinus)
prop_mu_plus = ProposalInterface.make_propagator(PMuPlus)
targets = get_detector_modules(d)

event_collection = EventCollection(injector)
for _ in 1:100
    event = rand(injector)
    muon = event[:particles][1]
    isec = get_intersection(cylinder, muon)

    if isnothing(isec.first) || isec.first < 0
        event[:hits] = DataFrame()
        event[:total_hits] = 0
        push!(event_collection, event)
        continue
    end

    this_prop = muon.type == PMuMinus ? prop_mu_minus : prop_mu_plus
    muon_propagated, stoch, cont = propagate_muon(muon, propagator=this_prop, length=isec.second)

    stochastic_losses::Vector{Particle} = [loss for loss in stoch if loss.energy > 50 ]
    continuous_losses = vcat([loss for loss in stoch if loss.energy <= 50 ], cont)

    if !isempty(stochastic_losses)
        hits, mask = generate_hit_times(stochastic_losses, d, hit_generator_extended; abs_scale=abs_scale, sca_scale=sca_scale)
        hits_stoch = hit_list_to_dataframe(hits, targets, mask)
    else
        hits_stoch = DataFrame()
    end

    if !isempty(continuous_losses)
        ls_muon = Particle(
            muon.position .+  muon.direction .* isec.first,
            muon.direction,
            muon.time .+ isec.first / c_vac,
            sum(loss.energy for loss in continuous_losses),
            1E4,
            muon.type)
        hits, mask = generate_hit_times([ls_muon], d, hit_generator_lightsabre; abs_scale=abs_scale, sca_scale=sca_scale)
        hits_ls = hit_list_to_dataframe(hits, targets, mask)
    else
        hits_ls = DataFrame()
    end
    
    hits = vcat(hits_stoch, hits_ls)

    event[:hits] = hits
    event[:total_hits] = nrow(hits)

    push!(event_collection, event)
end


hits = event_collection[argmax([e[:total_hits] for e in event_collection])][:hits]

grpd = groupby(hits, [:module_id, :pmt_id])
hits_per_mod = combine(grpd, nrow)
mrow = argmax(hits_per_mod.nrow)

hits_max_mod = grpd[(module_id=hits_per_mod[mrow, :module_id], pmt_id=hits_per_mod[mrow, :pmt_id])]

hist(hits_max_mod[:, :time])

hits_max_mod = groupby(hits, [:module_id])[(;module_id=hits_per_mod[mrow, :module_id])]


all_lc_triggers = []

for (groupn, group) in pairs(groupby(hits, [:module_id]))

    lc_this_mod = lc_trigger(sort(group, :time))
    if isempty(lc_this_mod)
        continue
    end
    push!(all_lc_triggers, lc_this_mod)
end


module_trigger(reduce(vcat, all_lc_triggers))




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
