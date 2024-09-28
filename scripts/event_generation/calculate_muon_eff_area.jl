using NeutrinoTelescopes
using PhotonPropagation
using PhysicsTools
using PhotonSurrogateModel
using NeutrinoSurrogateModelData
using Distributions
using Random
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
using LinearAlgebra
using PreallocationTools
using ProposalInterface
using CairoMakie
using ArgParse

const c_vac = ustrip(u"m/ns", c_0)

function generate_muon_losses(ec::EventCollection; stoch_cut = 100)
    prop_mu_minus = ProposalInterface.make_propagator(PMuMinus)
    prop_mu_plus = ProposalInterface.make_propagator(PMuPlus)

    cylinder = ec.gen_info.injector.surface

    for ev in ec
        muon = ev[:particles][1]

        if particle_shape(muon) != Track()
            continue
        end

        isec = get_intersection(cylinder, muon)
        
        this_prop = muon.type == PMuMinus ? prop_mu_minus : prop_mu_plus
        muon_propagated, stoch, cont = propagate_muon(muon, propagator=this_prop, length=isec.second)

        muon_energy_at_entry = calc_muon_energy_at_entry(muon, isec.first, stoch, cont)

        stochastic_losses::Vector{Particle} = [loss for loss in stoch if loss.energy > stoch_cut ]
        continuous_losses = vcat([loss for loss in stoch if loss.energy <= stoch_cut ], cont)

        ev[:cascade_emitters] = stochastic_losses
        
        ls_emitters::Vector{Particle} = []

        if !isempty(continuous_losses) && isec.first > 0
            cont_loss = sum(loss.energy for loss in continuous_losses)
            ls_muon_cont = Particle(
                muon.position,
                muon.direction,
                muon.time,
                cont_loss,
                1E4,
                muon.type)
            push!(ls_emitters, ls_muon_cont)
        end
        
        ev[:lightsabre_emitters] = ls_emitters
        ev[:e_entry] = muon_energy_at_entry

        muon_entry = Particle(muon.position + muon.direction * isec.first, muon.direction, muon.time + isec.first / c_vac, muon_energy_at_entry, isec.second - isec.first, muon.type)

        ev[:muon_at_entry] = muon_entry
    end
end


function create_hits(event_collection, detector; abs_scale=1., sca_scale=1.)
    targets = get_detector_modules(detector)
    
    model_lightsabre = gpu(PhotonSurrogate(lightsabre_time_model(2)...))
    model_extended = gpu(PhotonSurrogate(em_cascade_time_model(2)...))
    hit_generator_lightsabre = SurrogateModelHitGenerator(model_lightsabre, 200.0, detector)
    hit_generator_extended = SurrogateModelHitGenerator(model_extended, 200.0, detector)

    @progress for event in event_collection
        if haskey(event, :cascade_emitters)
    
            stochastic_losses = event[:cascade_emitters]
            lightsabre_losses = event[:lightsabre_emitters]
            muon_entry = event[:muon_at_entry]
    
            println("Event $(event.id) with $(length(stochastic_losses)) stochastic and $(length(lightsabre_losses)) lightsabre emitters")
    
            if !isempty(stochastic_losses)
                hits, mask = generate_hit_times(stochastic_losses, detector, hit_generator_extended; abs_scale=abs_scale, sca_scale=sca_scale, device=gpu)
                hits_stoch = hit_list_to_dataframe(hits, targets, mask)
            else
                hits_stoch = DataFrame(time = Float64[], module_id = Int64[], pmt_id = Int64[])
            end
    
            if !isempty(lightsabre_losses)
                hits, mask = generate_hit_times(lightsabre_losses, detector, hit_generator_lightsabre; abs_scale=abs_scale, sca_scale=sca_scale, device=gpu)
                hits_ls = hit_list_to_dataframe(hits, targets, mask)
            else
                hits_ls = DataFrame(time = Float64[], module_id = Int64[], pmt_id = Int64[])
            end
    
            hits_lightsabre, mask = generate_hit_times([muon_entry], detector, hit_generator_lightsabre; abs_scale=abs_scale, sca_scale=sca_scale, device=gpu)
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
end    

function run_triggers(event_collection, hits_name::Symbol; lc_tw=20, postfix="")
    @progress for event in event_collection

        module_triggers = ModuleCoincTrigger[]

        if haskey(event, hits_name) && nrow(event[hits_name]) > 0
            hits = event[hits_name]
            all_lc_triggers = []
        
            for (groupn, group) in pairs(groupby(hits, [:module_id]))

                lc_this_mod = lc_trigger(sort(group, :time), time_window=lc_tw)
                if isempty(lc_this_mod)
                    continue
                end
                push!(all_lc_triggers, lc_this_mod)
            end
            
            if !isempty(all_lc_triggers)
                module_triggers = module_trigger(reduce(vcat, all_lc_triggers))
            end            
        end

        key_name = Symbol("module_triggers_" * string(hits_name) * postfix)

        event[key_name] = module_triggers
    end
end


function get_geo_proj_area(p, direction)
    basis = hcat(orthonormal_basis(direction)...)
    p2 = project(p, basis)
    vol = 0.
    try
        vol = Polyhedra.volume(polyhedron(hrep(p2)))
    catch y
        @show p2, hrep(p2), direction
        vol = missing
    end
    return vol
end

function get_average_geo_proj_area(p; n_samples=1000, min_cos=-1, max_cos=1)

    thetas = acos.(rand(Uniform(min_cos, max_cos), n_samples))
    phis = rand(Uniform(0, 2*π), n_samples)

    directions = sph_to_cart.(thetas, phis)

    return mean(skipmissing(get_geo_proj_area.(Ref(p), directions)))
end

function make_track_injector(cylinder)
    surf = CylinderSurface(cylinder)
    pdist = CategoricalSetDistribution(OrderedSet([PMuPlus, PMuMinus]), [0.5, 0.5])
    edist = PowerLogUniform(1E2, 1E8)
    ang_dist = LowerHalfSphere()
    length_dist = Dirac(1E4)
    time_dist = Dirac(0.0)
    inj = SurfaceInjector(surf, edist, pdist, ang_dist, length_dist, time_dist)
    return inj
end

function calculate_weights(event_collection)
    gen_info = event_collection.gen_info
    inj = gen_info.injector
    n_events = gen_info.n_events

    cylinder = Cylinder(inj.surface)
    for ev in event_collection
        ev[:generation_area] = projected_area(cylinder, ev[:muon_at_entry].direction)
        ev[:spectrum_weight] = pdf(inj.e_dist, ev[:e_entry])
        ev[:area_weight] = n_events * ev[:spectrum_weight] / acceptance(cylinder, -1, 1) 
        ev[:area_weight_alt] = n_events * ev[:spectrum_weight] / ev[:generation_area]
    end
end

function get_directional_uncertainty(dir_sph, cov)
    dir_cart = sph_to_cart(dir_sph)

    dist = MvNormal(dir_sph, cov)
    rdirs = rand(dist, 100)

    dangles = rad2deg.(acos.(dot.(sph_to_cart.(rdirs[1, :], rdirs[2, :]), Ref(dir_cart))))
    return mean(dangles)
end


function calculate_fisher_uncert(event_collection, detector; abs_scale=1., sca_scale=1., skip_n=10)
    model_lightsabre = gpu(PhotonSurrogate(lightsabre_time_model(2)...))
    hit_generator_lightsabre = SurrogateModelHitGenerator(model_lightsabre, 200.0, detector)

    input_buffer = create_input_buffer(model_lightsabre, detector, 1)
    diff_cache = DiffCache(input_buffer, 13)

    triggered_events = [haskey(ev, :module_triggers_hits20) && length(ev[:module_triggers_hits20]) > 1 for ev in event_collection]

    selected_ix = (1:length(event_collection))[triggered_events]

    for event_ix in selected_ix[begin:skip_n:end]
        println("Calculating Fisher for event $event_ix")
        event = event_collection[event_ix]
        fisher_calc = calc_fisher_matrix(event[:muon_at_entry], detector, hit_generator_lightsabre, cache=diff_cache, n_samples=100, abs_scale=abs_scale, sca_scale=sca_scale)[1] 

        if isapprox(det(fisher_calc), 0)
            dir_uncert = NaN
        else
            cov = inv(fisher_calc)

            try
                dir_sph = cart_to_sph(event[:muon_at_entry].direction)
                dir_uncert = get_directional_uncertainty(dir_sph, cov[2:3, 2:3])
            catch e
                dir_uncert = NaN
            end
        end
        event[:dir_uncert] = dir_uncert
    end
end



function calc_fisher_surrogate(event_collection, detector, abs_scale=1., sca_scale=1.,)
    type = "per_string_lightsabre"
    model_fname = joinpath(ENV["ECAPSTOR"], "snakemake/fisher_surrogates/fisher_surrogate_$type.bson")
    max_particles = 1000
    max_targets = 70*20
    if occursin("per_string", type)
        fisher_surrogate = gpu(FisherSurrogateModelPerLine(model_fname, max_particles, max_targets))        
    else
        fisher_surrogate = gpu(FisherSurrogateModelPerModule(model_fname, max_particles, max_targets))
    end

    det_lines = get_detector_lines(detector)
    targets = get_detector_modules(detector)

    events = event_collection.events
    event_mask = get_events_in_range(events, targets, fisher_surrogate)
    valid_events = events[event_mask]

    fishers_pred = predict_fisher([p[:muon_at_entry] for p in valid_events], det_lines, fisher_surrogate, abs_scale=abs_scale, sca_scale=sca_scale)
    dir_uncert_pred = calc_dir_uncert(fishers_pred, valid_events)

    for (ev, uncert) in zip(valid_events, dir_uncert_pred)
        ev[:dir_uncert_surrogate] = uncert
    end
end


function main(args)
    targets_hex = make_n_hex_cluster_detector(7, args[:hor_spacing], 20, args[:vert_spacing])
    abs_scale = 1f0
    sca_scale = 1f0
    medium = make_cascadia_medium_properties(0.95f0, abs_scale, sca_scale)
    d = LineDetector(targets_hex, medium)
    cylinder = get_bounding_cylinder(d, padding_side=75, padding_top=75)
    inj = make_track_injector(cylinder)

    n_events = args[:n_events]

    rng = MersenneTwister(args[:seed])

    events = [rand(rng, inj) for _ in 1:n_events]

    gen_info = GenerationInfo(inj, n_events)
    event_collection = EventCollection(events, gen_info)

    generate_muon_losses(event_collection)
    create_hits(event_collection, d)

    run_triggers(event_collection, :hits, lc_tw=10., postfix="10")
    run_triggers(event_collection, :hits, lc_tw=20., postfix="20")
    run_triggers(event_collection, :lightsabre_hits, lc_tw=10., postfix="10")
    run_triggers(event_collection, :lightsabre_hits, lc_tw=20., postfix="20")

    calculate_weights(event_collection)

    calc_fisher_surrogate(event_collection, d, abs_scale=abs_scale, sca_scale=sca_scale)
    calculate_fisher_uncert(event_collection, d, abs_scale=abs_scale, sca_scale=sca_scale, skip_n=5)

    targets = get_detector_modules(d)
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


    jldopen(args[:outfile], "w") do hdl
        event_group = JLD2.Group(hdl, "events")
        for ev in event_collection
            event_group["event_$(ev.id)"] = ev
        end
        hdl["geo"] = geo
        hdl["gen_info"] = event_collection.gen_info
    end
end


s = ArgParseSettings()
@add_arg_table s begin
    "--outfile"
    help = "Output file"
    required = true
    "--vert_spacing"
    help = "Vertical Module Spacing"
    required = true
    arg_type = Float64
    "--hor_spacing"
    help = "Horizontal Line Spacing"
    required = true
    arg_type = Float64
    "--n_events"
    help = "Number of events"
    required = true
    arg_type = Int64
    "--seed"
    help = "Seed"
    required = true
    arg_type = Int64
end
parsed_args = parse_args(ARGS, s; as_symbols=true)
main(parsed_args)