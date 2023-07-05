module FisherInformation
using ForwardDiff
using PhysicsTools
using Random
using PhotonPropagation
using StaticArrays
using StatsBase
using Flux
using PreallocationTools
using ...SurrogateModels.ExtendedCascadeModel
using ...EventGeneration

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
        p = Particle(pos, direction, T(time), 10^log_energy, length, ptype)

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



function calc_fisher_matrix(
    event::Event,
    detector::Detector{T, MP},
    generator::SurrogateModelHitGenerator;
    use_grad=false,
    rng=Random.GLOBAL_RNG,
    n_samples=100,
    cache=nothing,
    device=gpu) where {T <: PhotonTarget, MP <: MediumProperties}

    medium = get_detector_medium(detector)

    modules::Vector{T} = get_detector_modules(detector)

    matrices = []
    for __ in 1:n_samples
        times, range_mask = generate_hit_times(event, detector, generator, rng, device=device)
        
        if sum(length.(times)) == 0
            continue
        end

        p::Particle = event[:particles][1]
        dir = p.direction
        dir_theta, dir_phi = cart_to_sph(dir)
        logenergy = log10(p.energy)
        pos = p.position

        targets_range = get_detector_modules(detector)[range_mask]

        f, fwrapped = make_lh_func(time=0., data=times, targets=targets_range, model=generator.model, medium=medium, diff_cache=cache, ptype=p.type, device=device)
       
        if use_grad
            logl_grad = collect(ForwardDiff.gradient(fwrapped, [logenergy, dir_theta, dir_phi, pos...]))
            push!(matrices, logl_grad .* logl_grad')
        else
            logl_hessian =  ForwardDiff.hessian(fwrapped, [logenergy, dir_theta, dir_phi, pos...])
            push!(matrices, .-logl_hessian)
        end
    end

    if length(matrices) > 0
        fisher_matrix = mean(matrices)
        fisher_matrix = 0.5 * (fisher_matrix + fisher_matrix')
    else
        fisher_matrix = zeros(6, 6)
    end

    return fisher_matrix

end


function calc_fisher(
    d::Detector,
    inj::Injector,
    g::SurrogateModelHitGenerator,
    n_events::Integer,
    n_samples::Integer; use_grad=false, rng=rng=Random.GLOBAL_RNG, cache=nothing, device=gpu)
    matrices = []
    ec = EventCollection(inj)
    for _ in 1:n_events
        event = rand(inj)
        m = calc_fisher_matrix(event, d, g, use_grad=use_grad, rng=rng, n_samples=n_samples, cache=cache, device=device)

        push!(matrices, m)
        push!(ec, event)
    end
    
    return matrices, ec
end


end