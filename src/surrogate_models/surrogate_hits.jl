
module SurrogateModelHits

using ..PhotonSurrogates
using ...Processing
using ...EventGeneration.Detectors
using ...EventGeneration
using PhotonPropagation
using Random
using LinearAlgebra
using PhysicsTools
using Base.Iterators
using Flux
using ArraysOfArrays

export SurrogateModelHitGenerator, generate_hit_times, generate_hit_times!
export create_input_buffer, create_output_buffer
export get_modules_in_range

mutable struct SurrogateModelHitGenerator{M <: PhotonSurrogate}
    model::M
    max_valid_distance::Float64
    input_buffer::Union{Nothing, Matrix{Float64}}
    output_buffer::Union{Nothing, VectorOfVectors{Float64}}
end

function SurrogateModelHitGenerator(model, max_valid_distance, detector::Detector; max_particles=500, expected_hits_per_module=100) 
    input_buffer = create_input_buffer(model, detector, max_particles)
    output_buffer = create_output_buffer(detector, expected_hits_per_module)
    return SurrogateModelHitGenerator(model, max_valid_distance, input_buffer, output_buffer)
end


function create_input_buffer(input_size::Integer, n_det::Integer, max_particles=500)
    return zeros(input_size, n_det*max_particles)
end

function create_input_buffer(model::PhotonSurrogate, n_det::Integer, max_particles=500)
    input_size = size(model.time_model.embedding.layers[1].weight, 2)
    return create_input_buffer(input_size, n_det, max_particles)
end

function create_input_buffer(model_or_det, detector::Detector, max_particles=500)
    modules = get_detector_modules(detector)
    return create_input_buffer(model_or_det, get_pmt_count(eltype(modules))*length(modules), max_particles)
end


function create_output_buffer(n_det::Integer, expected_hits_per=100)
    buffer = VectorOfArrays{Float64, 1}()
    sizehint!(buffer, n_det, (expected_hits_per, ))
    return buffer
end

function create_output_buffer(detector::Detector, expected_hits_per=100)
    modules = get_detector_modules(detector)
    n_pmts = get_pmt_count(eltype(modules))*length(modules)
    return create_output_buffer(n_pmts, expected_hits_per)
end


"""
    get_modules_in_range(particles, modules, generator)

Return a BitMask of modules that are in range of at least one particle
"""
function get_modules_in_range(particles, modules::AbstractVector{<:PhotonTarget}, max_valid_distance)

    modules_range_mask = zeros(Bool, length(modules))

    for modix in eachindex(modules)
        t = modules[modix]
        for p in particles
            if closest_approach_distance(p, t) <= max_valid_distance
                modules_range_mask[modix] = true
                break
            end
        end
    end
   return modules_range_mask
end

function get_modules_in_range(particles, detector::Detector{T, <:MediumProperties}, max_valid_distance) where {T}
    modules::Vector{T} = get_detector_modules(detector)
    return get_modules_in_range(particles, modules, max_valid_distance)
end


function generate_hit_times(
    particles::Vector{<:Particle},
    detector::Detector,
    generator::SurrogateModelHitGenerator,
    rng=Random.default_rng();
    device=gpu,
    noise_rate=1E4,
    time_window=1E4,
    abs_scale,
    sca_scale)
    
    modules = get_detector_modules(detector)
    medium = get_detector_medium(detector)
    
    # Test whether any modules is within max_valid_distance
    modules_range_mask = get_modules_in_range(particles, detector, generator.max_valid_distance)
    if !isnothing(generator.input_buffer)
        n_p_buffer = size(generator.input_buffer, 2) / get_pmt_count(eltype(modules))*length(modules)
        if length(particles) > n_p_buffer
            @warn "Resizing buffer"
            generator.buffer = create_input_buffer(detector, length(particles))
        end
    end
    hit_list = sample_multi_particle_event(
        particles,
        modules[modules_range_mask],
        generator.model,
        medium,
        rng,
        feat_buffer=generator.input_buffer,
        output_buffer=generator.output_buffer,
        device=device,
        abs_scale=abs_scale,
        sca_scale=sca_scale,
        noise_rate=noise_rate,
        time_window=time_window)

    return hit_list, modules_range_mask

end

function generate_hit_times(
    event::Event,
    detector::Detector,
    generator::SurrogateModelHitGenerator,
    rng=Random.default_rng();
    device=gpu,
    abs_scale=1.,
    sca_scale=1.,
    noise_rate=1E4,
    time_window=1E4)
    particles = get_lightemitting_particles(event)
    return generate_hit_times(
        particles,
        detector,
        generator,
        rng,
        device=device,
        abs_scale=abs_scale,
        sca_scale=sca_scale,
        noise_rate=noise_rate,
        time_window=time_window)
end

function generate_hit_times(
    events::AbstractVector{<:Event},
    detector::Detector,
    generator::SurrogateModelHitGenerator,
    rng=Random.default_rng();
    device=gpu,
    abs_scale=1.,
    sca_scale=1.,
    noise_rate=1E4,
    time_window=1E4)
    particles_per_event = get_lightemitting_particles.(events)
    particles = reduce(vcat, particles_per_event)
    return generate_hit_times(
        particles,
        detector,
        generator,
        rng,
        device=device,
        abs_scale=abs_scale,
        sca_scale=sca_scale,
        noise_rate=noise_rate,
        time_window=time_window)
end


function generate_hit_times!(
    event::Event,
    detector::Detector,
    generator::SurrogateModelHitGenerator,
    rng=Random.default_rng();
    device=gpu,
    abs_scale=1.,
    sca_scale=1.,
    noise_rate=1E4,
    time_window=1E4)
    hits, modules_range_mask = generate_hit_times(
        event,
        detector,
        generator,
        rng,
        device=device,
        abs_scale=abs_scale,
        sca_scale=sca_scale,
        noise_rate=noise_rate,
        time_window=time_window
        )
    modules = get_detector_modules(detector)
    hits_df = hit_list_to_dataframe(hits, modules, modules_range_mask)
    event[:photon_hits] = hits_df
end
end