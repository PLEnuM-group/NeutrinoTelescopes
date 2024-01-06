module FisherInformation
using ForwardDiff
using PhysicsTools
using Random
using PhotonPropagation
using StaticArrays
using StatsBase
using Flux
using DataFrames
using PreallocationTools
using Base.Iterators
using ...SurrogateModels.NeuralFlowSurrogate
using ...SurrogateModels.SurrogateModelHits
using ...EventGeneration

using LinearAlgebra

import Base.GC: gc

export SimpleDiffCache
export calc_fisher, calc_fisher_matrix, make_lh_func


struct SimpleDiffCache{T <: AbstractArray, S <: AbstractArray}
    du::T
    dual_du::S
end

function SimpleDiffCache(du::AbstractArray, chunk_size)
    dual_du = zeros(ForwardDiff.Dual{nothing, eltype(du), chunk_size}, size(du))
    return SimpleDiffCache(du, dual_du)
end

get_cache(cache::SimpleDiffCache, ::Type{<:ForwardDiff.Dual}) = cache.dual_du
get_cache(cache::SimpleDiffCache, ::Type{<:Real}) = cache.du


function make_lh_func(;time, data, targets, model, medium, diff_cache, ptype, device=gpu)

    function evaluate_lh(log_energy::Real, dir_theta::Real, dir_phi::Real, pos_x::Real, pos_y::Real, pos_z::Real; cache=nothing)
        T = promote_type(typeof(pos_x), typeof(pos_y), typeof(pos_z), typeof(log_energy), typeof(dir_theta), typeof(dir_phi))

        pos = SVector{3, T}(pos_x, pos_y, pos_z)

        length = particle_shape(ptype) == Track() ? T(10000.) : T(0.)
        
        direction = sph_to_cart(dir_theta, dir_phi)
        p = Particle(T.(pos), T.(direction), T(time), T(10^log_energy), T(length), ptype)

        #=
        if particle_shape(p) == Track()
            p = shift_to_closest_approach(p, SVector{3, T}(0, 0, 0))
        end
        =#

        return multi_particle_likelihood([p], data=data, targets=targets, model=model, medium=medium, feat_buffer=cache, amp_only=false, device=device)

    end

    function wrapped(pars::Vector{<:Real})
        if !isnothing(diff_cache)
            dc = get_tmp(diff_cache, pars)
        else
            dc = nothing
        end
        return evaluate_lh(pars...; cache=dc)

    end

    return evaluate_lh, wrapped
end


function filter_medad_eigen(matrices, threshold=10)
    x = eigmax.(matrices)
    med = median(x)
    medad = median(abs.(x .- med ))
    dev = abs.(x .- med) ./ medad
    return matrices[dev .< threshold]
end


function _calc_single_fisher_matrix(event::Event, detector::Detector, generator, rng, device, cache, use_grad)
    times, range_mask = generate_hit_times(event, detector, generator, rng, device=device)
        
    if sum(length.(times)) == 0
        return nothing
    end

    # Test sampling each dimension independently
    targets_range = get_detector_modules(detector)[range_mask]  

   
   
    p::Particle = event[:particles][1]

    if particle_shape(p) == Track()
        n_pmt = get_pmt_count(eltype(targets_range))
        total_hits = sum(length.(times))

        lin_ixs = LinearIndices((1:n_pmt, 1:length(targets_range)))

        weighted_pos = []
        for (i, t) in enumerate(targets_range)
            this_n_hits = sum(length.(times[lin_ixs[:, i]]))
            push!(weighted_pos, t.shape.position .* this_n_hits)
        end
        cad_com = closest_approach_param(p, sum(weighted_pos) / sum(total_hits))
        p = shift_particle(p, cad_com)
    end

    dir = p.direction
    dir_theta, dir_phi = cart_to_sph(dir)
    logenergy = log10(p.energy)
    pos = p.position

    medium = get_detector_medium(detector)

    f, fwrapped = make_lh_func(time=p.time, data=times, targets=targets_range, model=generator.model, medium=medium, diff_cache=cache, ptype=p.type, device=device)
   
    if use_grad
        logl_grad = collect(ForwardDiff.gradient(fwrapped, [logenergy, dir_theta, dir_phi, pos...]))
        fi = logl_grad .* logl_grad' 
    else
        logl_hessian =  ForwardDiff.hessian(fwrapped, [logenergy, dir_theta, dir_phi, pos...])
        fi = .-logl_hessian
    end

    return fi
end

"""
    calc_fisher_matrix(
        event::Event,
        detector::Detector{T, MP},
        generator::SurrogateModelHitGenerator;
        use_grad=false,
        rng=Random.GLOBAL_RNG,
        n_samples=100,
        cache=nothing,
        device=gpu,
        filter_outliers=true) where {T <: PhotonTarget, MP <: MediumProperties}

Calculate fisher information matrix for `event`.
"""
function calc_fisher_matrix(
    event::Event,
    detector::Detector{T, MP},
    generator::SurrogateModelHitGenerator;
    use_grad=true,
    rng=Random.GLOBAL_RNG,
    n_samples=100,
    cache=nothing,
    device=gpu,
    filter_outliers=true) where {T <: PhotonTarget, MP <: MediumProperties}

    matrices = Matrix[]
    for __ in 1:n_samples
        fi = _calc_single_fisher_matrix(event, detector, generator, rng, device, cache, use_grad)
        if isnothing(fi)
            continue
        end
        push!(matrices, fi)
    end

    if (length(matrices) > 0) && filter_outliers
        matrices = filter_medad_eigen(matrices)
    end
    if length(matrices) > 0
        fisher_matrix = mean(matrices)
        fisher_matrix = 0.5 * (fisher_matrix + fisher_matrix')
    else
        fisher_matrix = zeros(6, 6)
    end

    return fisher_matrix, matrices

end


function calc_fisher(
    d::Detector,
    inj::Injector,
    g::SurrogateModelHitGenerator,
    n_events::Integer,
    n_samples::Integer; use_grad=false, rng=rng=Random.GLOBAL_RNG, cache=nothing, device=gpu)
    matrices = Matrix[]
    #ec = EventCollection(inj)
    events = Event[]
    for _ in 1:n_events
        event = rand(inj)
        m, _ = calc_fisher_matrix(event, d, g, use_grad=use_grad, rng=rng, n_samples=n_samples, cache=cache, device=device)

        push!(matrices, m)
        push!(events, event)
        gc()
    end

    
    return matrices, events
end


end