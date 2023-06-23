module PMTHits

using ...SurrogateModels
using ..Detectors
using PhotonPropagation
using Random
using LinearAlgebra
using PhysicsTools

import ..Event
import ..get_lightemitting_particles

export SurrogateModelHitGenerator, generate_hit_times, generate_hit_times!

abstract type HitGenerator end

mutable struct SurrogateModelHitGenerator{M <: PhotonSurrogate}
    model::M
    max_valid_distance::Float64
    buffer::Union{Nothing, Matrix{Float64}}
end

function SurrogateModelHitGenerator(model, max_valid_distance, detector::Detector) 
    buffer = create_input_buffer(detector)
    return SurrogateModelHitGenerator(model, max_valid_distance, buffer)
end

function create_input_buffer(detector::Detector, max_particles=500)
    modules = get_detector_modules(detector)
    return zeros(24, get_pmt_count(eltype(modules))*length(modules)*max_particles)
end

function generate_hit_times(particles::Vector{<:Particle}, detector::Detector, generator::SurrogateModelHitGenerator, rng=Random.GLOBAL_RNG)
    modules = get_detector_modules(detector)
    medium = get_detector_medium(detector)
    c_n = c_at_wl(800., medium)
    
    # Test whether any modules is within max_valid_distance
    modules_range_mask = any(norm.([p.position for p in particles] .- permutedims([t.shape.position for t in modules])) .<= generator.max_valid_distance, dims=1)[1, :]
    
    if !isnothing(generator.buffer)
        n_p_buffer = size(generator.buffer, 2) / get_pmt_count(eltype(modules))*length(modules)
        if length(particles) > n_p_buffer
            @warn "Resizing buffer"
            generator.buffer = create_input_buffer(detector, length(particles))
        end
    end
    hit_list = sample_multi_particle_event(particles, modules[modules_range_mask], generator.model, c_n, rng, feat_buffer=generator.buffer)

    return hit_list, modules_range_mask

end

function generate_hit_times(event::Event, detector::Detector, generator::SurrogateModelHitGenerator, rng=nothing)
    particles = get_lightemitting_particles(event)
    return generate_hit_times(particles, detector, generator, rng)
end

function generate_hit_times!(event::Event, detector::Detector, generator::SurrogateModelHitGenerator, rng=nothing)
    hits, modules_range_mask = generate_hit_times(event, detector, generator, rng)
    modules = get_detector_modules(detector)
    hits_df = hit_list_to_dataframe(hits, modules, modules_range_mask)
    event[:photon_hits] = hits_df
end
end