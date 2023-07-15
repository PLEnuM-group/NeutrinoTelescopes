module PMTHits

using ...SurrogateModels
using ...Processing
using ..Detectors
using PhotonPropagation
using Random
using LinearAlgebra
using PhysicsTools
using Base.Iterators

import ..Event
import ..get_lightemitting_particles

export SurrogateModelHitGenerator, generate_hit_times, generate_hit_times!
export create_input_buffer
export get_modules_in_range

abstract type HitGenerator end

mutable struct SurrogateModelHitGenerator{M <: PhotonSurrogate}
    model::M
    max_valid_distance::Float64
    buffer::Union{Nothing, Matrix{Float64}}
end

function SurrogateModelHitGenerator(model, max_valid_distance, detector::Detector; max_particles=500) 
    buffer = create_input_buffer(detector, max_particles)
    return SurrogateModelHitGenerator(model, max_valid_distance, buffer)
end

function create_input_buffer(detector::Detector, max_particles=500)
    modules = get_detector_modules(detector)
    return zeros(24, get_pmt_count(eltype(modules))*length(modules)*max_particles)
end


"""
    get_modules_in_range(particles, modules, generator)

Return a BitMask of modules that are in range of at least one particle
"""
function get_modules_in_range(particles, modules::AbstractVector{<:PhotonTarget}, max_valid_distance)
    closest_d = reshape(mapreduce(((p, t),) -> closest_approach_distance(p, t), vcat, product(particles, modules)), length(particles), length(modules))

    modules_range_mask::Vector{Bool} = any(closest_d .<= max_valid_distance, dims=1)[:]

   return modules_range_mask
end

function get_modules_in_range(particles, detector::Detector{T, <:MediumProperties}, max_valid_distance) where {T}
    modules::Vector{T} = get_detector_modules(detector)
    return get_modules_in_range(particles, modules, max_valid_distance)
end




function generate_hit_times(particles::Vector{<:Particle}, detector::Detector, generator::SurrogateModelHitGenerator, rng=Random.GLOBAL_RNG; device=gpu)
    modules = get_detector_modules(detector)
    medium = get_detector_medium(detector)
    
    # Test whether any modules is within max_valid_distance
    modules_range_mask = get_modules_in_range(particles, detector, generator.max_valid_distance)
    if !isnothing(generator.buffer)
        n_p_buffer = size(generator.buffer, 2) / get_pmt_count(eltype(modules))*length(modules)
        if length(particles) > n_p_buffer
            @warn "Resizing buffer"
            generator.buffer = create_input_buffer(detector, length(particles))
        end
    end
    hit_list = sample_multi_particle_event(particles, modules[modules_range_mask], generator.model, medium, rng, feat_buffer=generator.buffer, device=device)

    return hit_list, modules_range_mask

end

function generate_hit_times(event::Event, detector::Detector, generator::SurrogateModelHitGenerator, rng=nothing; device=gpu)
    particles = get_lightemitting_particles(event)
    return generate_hit_times(particles, detector, generator, rng, device=device)
end

function generate_hit_times!(event::Event, detector::Detector, generator::SurrogateModelHitGenerator, rng=nothing; device=gpu)
    hits, modules_range_mask = generate_hit_times(event, detector, generator, rng, device=device)
    modules = get_detector_modules(detector)
    hits_df = hit_list_to_dataframe(hits, modules, modules_range_mask)
    event[:photon_hits] = hits_df
end
end