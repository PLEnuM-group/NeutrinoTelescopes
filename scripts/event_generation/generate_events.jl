using NeutrinoTelescopes
using PhotonPropagation
using PhysicsTools
using PhotonSurrogateModel
using NeutrinoSurrogateModelData
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
using HDF5
using Format
using PhysicalConstants.CODATA2018: c_0
using ProgressLogging
using Base.Iterators
using StatsBase
using Base.Iterators
using LinearAlgebra
using PreallocationTools
using ProposalInterface


function get_events_in_range(events, targets, model)
    events_range_mask = falses(length(events))
    for evix in eachindex(events)
        for t in targets
            @views if FisherSurrogate.is_in_range(first(events[evix][:particles]), t.shape.position[1:2], model)
                events_range_mask[evix] = true
                break
            end
        end
    end
    return events_range_mask
end


function get_event_props(event::Event)
    p = first(event[:particles])
    dir_sph = cart_to_sph(p.direction)
    shift_to_closest_approach(p, [0., 0., 0.])
    return [log10(p.energy), dir_sph..., p.position...]
end

function get_directional_uncertainty(dir_sph, cov)
    dir_cart = sph_to_cart(dir_sph)

    dist = MvNormal(dir_sph, cov)
    rdirs = rand(dist, 100)

    dangles = rad2deg.(acos.(dot.(sph_to_cart.(rdirs[1, :], rdirs[2, :]), Ref(dir_cart))))
    return mean(dangles)
end

function calc_dir_uncert(fishers, events)
    covs = invert_fishers(fishers)
    valid = isposdef.(covs)
    results = zeros(length(events))

    for (i, (c, e)) in enumerate(zip(covs, events))
        if !valid[i]
            results[i] = NaN
            continue
        end
        event_props = get_event_props(e)
        mean_ang_dev = get_directional_uncertainty(event_props[2:3], c[2:3, 2:3])
        results[i] = mean_ang_dev
    end

    return results
end

function calc_muon_energy_at_entry(muon, t_entry, stoch, cont)
    energy_loss = 0
    for p in stoch
        t_stoch = mean((p.position - muon.position) ./ muon.direction)
        if t_stoch < t_entry
            energy_loss += p.energy
        else
            break
        end
    end

    for p in cont
        t_cont = mean((p.position - muon.position) ./ muon.direction)
        if t_cont < t_entry
            energy_loss += p.energy
        else
            break
        end
    end

    return muon.energy - energy_loss
end


c_vac = ustrip(u"m/ns", c_0)

workdir = ENV["ECAPSTOR"]

model_lightsabre = gpu(PhotonSurrogate(lightsabre_time_model(2)...))
model_extended = gpu(PhotonSurrogate(em_cascade_time_model(2)...))

targets_hex = make_n_hex_cluster_detector(7, 80, 20, 50)

abs_scale = 1f0
sca_scale = 1f0
medium = make_cascadia_medium_properties(0.95f0, abs_scale, sca_scale)
d = LineDetector(targets_hex, medium)
cylinder = get_bounding_cylinder(d, padding_side=75, padding_top=75)
hit_generator_lightsabre = SurrogateModelHitGenerator(model_lightsabre, 200.0, d)
hit_generator_extended = SurrogateModelHitGenerator(model_extended, 200.0, d)
li_file = "/home/wecapstor3/capn/capn100h/leptoninjector/muons.hd5"


injector = LIInjector(li_file, drop_starting=false)
prop_mu_minus = ProposalInterface.make_propagator(PMuMinus)
prop_mu_plus = ProposalInterface.make_propagator(PMuPlus)
targets = get_detector_modules(d)

event_collection = EventCollection()

for ev in take(injector, 10000)
    muon = ev[:particles][1]
    isec = get_intersection(cylinder, muon)

    # THROW OUT STARTING MUONS
    if isnothing(isec.first) || isec.first < 0 || muon.energy < 1
        push!(event_collection, ev)
        continue
    end

    this_prop = muon.type == PMuMinus ? prop_mu_minus : prop_mu_plus
    muon_propagated, stoch, cont = propagate_muon(muon, propagator=this_prop, length=isec.first)

    muon_energy_at_entry = calc_muon_energy_at_entry(muon, isec.first, stoch, cont)

    stochastic_losses::Vector{Particle} = [loss for loss in stoch if loss.energy > 50 ]
    continuous_losses = vcat([loss for loss in stoch if loss.energy <= 50 ], cont)

    ev[:cascade_emitters] = stochastic_losses
    
    ls_emitters::Vector{Particle} = []

    if !isempty(continuous_losses) && isec.first > 0
        cont_loss = sum(loss.energy for loss in continuous_losses)
        ls_muon = Particle(
            muon.position .+  muon.direction .* isec.first,
            muon.direction,
            muon.time .+ isec.first / c_vac,
            cont_loss,
            1E4,
            muon.type)
        push!(ls_emitters, ls_muon)
    end
      
    ev[:lightsabre_emitters] = ls_emitters
    ev[:e_entry] = muon_energy_at_entry

    muon_entry = Particle(muon.position + muon.direction * isec.first, muon.direction, muon.time + isec.first / c_vac, muon_energy_at_entry, isec.second - isec.first, muon.type)

    ev[:muon_at_entry] = muon_entry
    push!(event_collection, ev)
end


@progress for event in event_collection
    if haskey(event, :cascade_emitters)

        stochastic_losses = event[:cascade_emitters]
        lightsabre_losses = event[:lightsabre_emitters]
        muon_entry = event[:muon_at_entry]


        println("Event $(event.id) with $(length(stochastic_losses)) stochastic and $(length(lightsabre_losses)) lightsabre emitters")

        if !isempty(stochastic_losses)
            hits, mask = generate_hit_times(stochastic_losses, d, hit_generator_extended; abs_scale=abs_scale, sca_scale=sca_scale, device=cpu)
            hits_stoch = hit_list_to_dataframe(hits, targets, mask)
        else
            hits_stoch = DataFrame(time = Float64[], module_id = Int64[], pmt_id = Int64[])
        end

        if !isempty(lightsabre_losses)
            hits, mask = generate_hit_times(lightsabre_losses, d, hit_generator_lightsabre; abs_scale=abs_scale, sca_scale=sca_scale, device=cpu)
            hits_ls = hit_list_to_dataframe(hits, targets, mask)
        else
            hits_ls = DataFrame(time = Float64[], module_id = Int64[], pmt_id = Int64[])
        end

        hits_lightsabre, mask = generate_hit_times([muon_entry], d, hit_generator_lightsabre; abs_scale=abs_scale, sca_scale=sca_scale, device=cpu)
        hits_lightsabre = hit_list_to_dataframe(hits_lightsabre, targets, mask)

        hits = vcat(hits_stoch, hits_ls)
        event[:cont_hits] = hits_ls
        event[:stochastic_hits] = hits_stoch
        event[:lightsabre_hits] = hits_lightsabre
    else
        hits = DataFrame(time = Float64[], module_id = Int64[], pmt_id = Int64[])
    end

    event[:hits] = hits
    event[:total_hits] = nrow(hits)
end

for event in event_collection

    module_triggers = ModuleCoincTrigger[]

    if haskey(event, :hits) && nrow( event[:hits]) > 0
        hits = event[:hits]
        all_lc_triggers = []
    
        for (groupn, group) in pairs(groupby(hits, [:module_id]))

            lc_this_mod = lc_trigger(sort(group, :time))
            if isempty(lc_this_mod)
                continue
            end
            push!(all_lc_triggers, lc_this_mod)
        end
        
        if !isempty(all_lc_triggers)
            module_triggers = module_trigger(reduce(vcat, all_lc_triggers))
        end            
    end

    event[:module_triggers] = module_triggers
end

events = [event for event in event_collection if haskey(event, :muon_at_entry) && event[:muon_at_entry].energy > 100]

event = events[1]

input_buffer = create_input_buffer(model_lightsabre, d, 1)
#output_buffer = create_output_buffer(d, 100)
diff_cache = DiffCache(input_buffer, 13)

fishers_calc = [calc_fisher_matrix(event[:muon_at_entry], d, hit_generator_lightsabre, cache=diff_cache, n_samples=100)[1] for event in events]
dir_uncert = calc_dir_uncert(fishers_calc, events)



type = "per_string_lightsabre"
model_fname = joinpath(ENV["ECAPSTOR"], "snakemake/fisher_surrogates/fisher_surrogate_$type.bson")
max_particles = 1000
max_targets = 70*20
if occursin("per_string", type)
    fisher_surrogate = gpu(FisherSurrogateModelPerLine(model_fname, max_particles, max_targets))        
else
    fisher_surrogate = gpu(FisherSurrogateModelPerModule(model_fname, max_particles, max_targets))
end

det_lines = get_detector_lines(d)

event_mask = get_events_in_range(events, targets, fisher_surrogate)
valid_events = events[event_mask]
fishers_pred = predict_fisher(valid_events, det_lines, fisher_surrogate, abs_scale=1., sca_scale=1.)
dir_uncert_pred = calc_dir_uncert(fishers_pred, valid_events)


energies = [e[:muon_at_entry].energy for e in valid_events]
zeniths = [e[:muon_at_entry].direction[3] for e in valid_events]
lengths = [e[:muon_at_entry].length for e in valid_events]

fig = Figure()
ax = Axis(fig[1,1])
scatter!(ax, log10.(energies), dir_uncert)
scatter!(ax, log10.(energies), dir_uncert_pred)
ylims!(0, 2)
fig


energy_bins = 2:0.2:7
e_center = (energy_bins[2:end] .+ energy_bins[1:end-1]) ./ 2
median_res = Float64[]

for i in eachindex(energy_bins)
    if i == length(energy_bins)
        break
    end

    low = energy_bins[i]
    high = energy_bins[i+1]

    mask = (10 .^low .<= energies) .&& (energies .< 10 .^high) .&& (lengths .> 100)

    if any(mask)
        push!(median_res, median(dir_uncert[mask]))
    else
        push!(median_res, 0)
    end
end


fig, ax, _ = CairoMakie.lines(e_center, median_res)
ylims!(ax, 1E-3, 1)
fig

fig = Figure()
ax = Axis(fig[1,1], yscale=log10)
scatter!(ax, lengths, dir_uncert, )

fig


geo = DataFrame([(
    module_id=Int64(target.module_id),
    pmt_id=Int64(pmt_id),
    x=target.shape.position[1],
    y=target.shape.position[2],
    z=target.shape.position[3],
    pmt_x=coord[1],
    pmt_y=coord[2],
    pmt_z=coord[3])
    for target in targets
    for (pmt_id, coord) in enumerate(sph_to_cart.(eachcol(target.pmt_coordinates)))]
)






jldopen("/home/wecapstor3/capn/capn100h/leptoninjector/muons.jld2", "w") do hdl
    event_group = JLD2.Group(hdl, "events")
    for ev in event_collection
        event_group["event_$(ev.id)"] = ev
    end
    hdl["geo"] = geo
end


