module FisherSurrogate

using Flux
using BSON
using ..PhotonSurrogates: create_model_input!, apply_normalizer, Normalizer, HyperParams, create_mlp_embedding
using ..SurrogateModelHits
using ...EventGeneration
using LinearAlgebra
using MLUtils
using PhysicsTools
using StaticArrays
using PhotonPropagation
using Base.Iterators
using ParameterSchedulers
using ParameterSchedulers: Stateful

export FisherSurrogateModel
export FisherSurrogateModelPerLine, FisherSurrogateModelPerModule
export FisherSurrogateModelParams
export predict_cov
export predict_cov_improvement
export predict_fisher
export invert_fishers


Base.@kwdef struct FisherSurrogateModelParams <: HyperParams
    mlp_layers::Int64 = 2
    mlp_layer_size::Int64 = 512
    lr::Float64 = 0.001
    dropout::Float64 = 0.1
    non_linearity::Function = relu
    l2_norm_alpha = 0.0
    lr_min = 1E-9
    epochs = 100
end

function Base.Dict(hp::FisherSurrogateModelParams) 
    d = Dict(
        "mlp_layers" => hp.mlp_layers,
        "mlp_layer_size" => hp.mlp_layer_size,
        "lr" => hp.lr,
        "dropout" => hp.dropout,
        "l2_norm_alpha" => hp.l2_norm_alpha,
        "non_linearity" => string(hp.non_linearity),
        "lr_min" => hp.lr_min,
        "epochs" => hp.epochs
    )
    return d
end


abstract type FisherSurrogateModel end

struct FisherSurrogateModelPerModule <: FisherSurrogateModel
    model::Chain
    tf_in::Vector{Normalizer}
    tf_out::Vector{Normalizer}
    max_valid_distance::Float64
    input_buffer::Matrix{Float32}    
end

function FisherSurrogateModelPerModule(fname::String, max_particles=500, max_targets=70*20)
    m = BSON.load(fname)
    Flux.testmode!(m[:model])
    buffer = zeros(Float32, 8, max_targets*max_particles)
    return FisherSurrogateModelPerModule(m[:model], m[:tf_in], m[:tf_out], 200., buffer)
end

struct FisherSurrogateModelPerLine <: FisherSurrogateModel
    model::Chain
    tf_in::Vector{Normalizer}
    tf_out::Vector{Normalizer}
    range_cylinder::Cylinder{Float64}
    input_buffer::Matrix{Float32}    
end

function FisherSurrogateModelPerLine(fname::String, max_particles=500, max_targets=70*20)
    m = BSON.load(fname)
    Flux.testmode!(m[:model])
    buffer = zeros(Float32, 9, max_targets*max_particles)
    #range_cylinder = Cylinder(SA[0., 0., -475.], 1200., 200.)
    return FisherSurrogateModelPerLine(m[:model], m[:tf_in], m[:tf_out], m[:range_cylinder], buffer)
end


Flux.gpu(s::FisherSurrogateModelPerModule) = typeof(s)(gpu(s.model), s.tf_in, s.tf_out, s.max_valid_distance, s.input_buffer)
Flux.cpu(s::FisherSurrogateModelPerModule) = typeof(s)(cpu(s.model), s.tf_in, s.tf_out, s.max_valid_distance, s.input_buffer)

Flux.gpu(s::FisherSurrogateModelPerLine) = typeof(s)(gpu(s.model), s.tf_in, s.tf_out, s.range_cylinder, s.input_buffer)
Flux.cpu(s::FisherSurrogateModelPerLine) = typeof(s)(cpu(s.model), s.tf_in, s.tf_out, s.range_cylinder, s.input_buffer)

function is_in_range(particle, line_xy, model::FisherSurrogateModelPerLine)
    # Cylinder centered around module

    shifted_pos = SA[
        line_xy[1],
        line_xy[2],
        model.range_cylinder.center[3]]


    range_cyl = Cylinder(shifted_pos, model.range_cylinder.height, model.range_cylinder.radius)
    if particle_shape(particle) == Track()
        isec = get_intersection(range_cyl, particle)
        in_range = !isnothing(isec.first)
    else
        in_range = point_in_volume(range_cyl, particle.position)
    end
    return in_range
end

#=
function SurrogateModelHits.get_modules_in_range(particles, modules, model::FisherSurrogateModelPerModule)
    return get_modules_in_range(particles, modules, model.max_valid_distance)
end

function SurrogateModelHits.get_modules_in_range(particles, modules, model::FisherSurrogateModelPerLine)
    return get_modules_in_range(particles, modules, model.max_valid_distance)
end
=#


function calculate_model_input!(
    particle_pos,
    particle_dir,
    particle_energy,
    line_xy,
    tf_vec, output;
    abs_scale, sca_scale)

    r = sqrt((particle_pos[1]-line_xy[1])^2 + (particle_pos[2]-line_xy[2])^2)
    h = particle_pos[3]
    phi = atan(particle_pos[2]-line_xy[2], particle_pos[1]-line_xy[1])

    @inbounds begin
        output[1] = tf_vec[1](log(r))
        output[2] = tf_vec[2](log(particle_energy))
        output[3] = tf_vec[3](particle_dir[1])
        output[4] = tf_vec[4](particle_dir[2])
        output[5] = tf_vec[5](particle_dir[3])
        output[6] = tf_vec[6](cbrt(h))
        output[7] = tf_vec[7](phi)
        output[8] = tf_vec[8](abs_scale)
        output[9] = tf_vec[9](sca_scale)
    end

    return output
end


function calculate_model_input!(
    particle::Particle,
    line_xy,
    tf_vec, output; abs_scale, sca_scale)

    if particle_shape(particle) == Track()
        particle = shift_to_closest_approach(particle, SA_F32[line_xy[1], line_xy[2], -475f0])
    end

    return calculate_model_input!(particle.position, particle.direction, particle.energy, line_xy, tf_vec, output, abs_scale=abs_scale, sca_scale=sca_scale)
    
end


function calculate_model_input!(particles::AbstractVector{<:Particle}, line_xys::AbstractVector{<:AbstractVector}, tf_vec, output; abs_scale, sca_scale)

    out_ix = LinearIndices((eachindex(particles), eachindex(line_xys)))
    
    for (p_ix, t_ix) in product(eachindex(particles), eachindex(line_xys))
        particle = particles[p_ix]
        line_xy = line_xys[t_ix]

        ix = out_ix[p_ix, t_ix]
        @views calculate_model_input!(particle, line_xy, tf_vec, output[:, ix], abs_scale=abs_scale, sca_scale=sca_scale)
    end
    return output
end

function calculate_model_input(particles::AbstractVector{<:Particle}, line_xys::AbstractVector{<:AbstractVector}, tf_vec; abs_scale, sca_scale)
    output = zeros(Float32, 9, 1)
    calculate_model_input!(particles, line_xys, tf_vec, output, abs_scale=abs_scale, sca_scale=sca_scale)
    return output
end


function setup_training(hparams::FisherSurrogateModelParams, n_batches)

    # Instantiate the optimizer
    opt = hparams.l2_norm_alpha > 0 ? Flux.Optimiser(WeightDecay(hparams.l2_norm_alpha), ADAM(hparams.lr)) : ADAM(hparams.lr)

    schedule = Stateful(CosAnneal(λ0=hparams.lr_min, λ1=hparams.lr, period=hparams.epochs))

    cov_model = create_mlp_embedding(
        hidden_structure=[hparams.mlp_layer_size for _ in 1:hparams.mlp_layers],
        n_in=9,
        n_out=21,
        dropout=hparams.dropout,
        non_linearity=hparams.non_linearity,
    ) |> gpu

    optim = Flux.setup(opt, cov_model)
    
    return cov_model, optim, schedule
end

function make_dataloaders(data, batch_size=4096)

    train_data, test_data = splitobs(data, at=0.8, shuffle=true)
    train_loader = Flux.DataLoader(train_data, batchsize=batch_size, shuffle=true)
    test_loader = Flux.DataLoader(test_data, batchsize=size(test_data[1], 2), shuffle=false)

    return train_loader, test_loader
end




function _predict_fisher(all_particles::AbstractVector{<:Particle}, targets, model::FisherSurrogateModel;  abs_scale, sca_scale)
    n_events = length(all_particles)
    normalizers::Vector{Normalizer{Float32}} = model.tf_out
    inv_y_tf = inv.(normalizers)
    # particles is the inner loop

    inp = @view calc_model_input!(all_particles, targets, model.tf_in, model.input_buffer, abs_scale=abs_scale, sca_scale=sca_scale)[:, 1:(length(all_particles) * length(targets))]

    
    triu_pred = cpu(apply_normalizer(model.model(gpu(inp)), inv_y_tf).^3)
    triu_pred = reshape(triu_pred, (size(triu_pred, 1), n_events, length(targets)))

    triu = zeros(6,6)
    triu_ixs = triu!((trues(6,6)))

    all_fishers = zeros(n_events, 6, 6)

    # Loop over events
    for (i, tp) in enumerate(eachslice(triu_pred, dims=2))

        modules_range_mask = get_modules_in_range([all_particles[i]], targets, model.max_valid_distance)

        if !any(modules_range_mask)
            return continue
        end

        # Loop over targets
        @inbounds for (j, trl) in enumerate(eachcol(tp))
            if !modules_range_mask[j]
                continue
            end
            triu[triu_ixs] .= trl
            all_fishers[i, :, :] .+= (triu'*triu)
        end
    end

    return all_fishers
end


function predict_fisher(events::Vector{<:Event}, targets, model::FisherSurrogateModelPerModule; abs_scale, sca_scale)
    all_particles::Vector{Particle{Float32}} = [(first(event[:particles])) for event in events]
    return _predict_fisher(all_particles, targets, model, abs_scale=abs_scale, sca_scale=sca_scale)
    
end

function predict_fisher(events::Vector{<:Event}, lines, model::FisherSurrogateModelPerLine; abs_scale, sca_scale)
    all_particles::Vector{Particle{Float32}} = [first(event[:particles]) for event in events]
    line_xys = [[first(l).shape.position[1], first(l).shape.position[2]] for l in lines]

    n_events = length(events)

    normalizers::Vector{Normalizer{Float32}} = model.tf_out

    inv_y_tf = inv.(normalizers)
    # particles is the inner loop
   
    inp = @view calculate_model_input!(all_particles, line_xys, model.tf_in, model.input_buffer, abs_scale=abs_scale, sca_scale=sca_scale)[:, 1:(length(all_particles) * length(line_xys))]
    

    triu_pred = cpu(apply_normalizer(model.model(gpu(inp)), inv_y_tf).^3)
    triu_pred = reshape(triu_pred, (size(triu_pred, 1), n_events, length(line_xys)))

    triu = zeros(6,6)
    triu_ixs = triu!((trues(6,6)))

    all_fishers = zeros(n_events, 6, 6)

    # Loop over events
    for (i, tp) in enumerate(eachslice(triu_pred, dims=2))
        # Loop over targets
        @inbounds for (j, trl) in enumerate(eachslice(tp, dims=2))

            p = all_particles[i]
            xy = line_xys[j]

            if !is_in_range(p, xy, model)
                continue
            end

            triu[triu_ixs] .= trl
            all_fishers[i, :, :] .+= (triu'*triu)
        end
    end

    return all_fishers
end

function invert_fishers(all_fishers)
    inverted = Matrix[]
    if typeof(all_fishers) <: AbstractVector
        it_func = all_fishers
    else
        it_func = eachslice(all_fishers, dims=1)
    end
    for m in it_func
        if isapprox(det(m), 0)
            push!(inverted, fill(NaN64, (6,6)))
            continue
        end

        invm::Matrix{Float64} = inv(m)
        invm .= 0.5 * (invm + invm')

        push!(inverted, invm)
    end
    return inverted
end




function predict_cov(events::Vector{<:Event}, detector, model::FisherSurrogateModel)

    modules = get_detector_modules(detector)

    all_fishers = predict_fisher(events, modules, model)
    inverted = invert_fishers(all_fishers)
    
    return inverted, all_fishers

end

function predict_cov(events::Vector{<:Event}, detector, model::FisherSurrogateModelPerLine; abs_scale, sca_scale)

    lines = get_detector_lines(detector)
    all_fishers = predict_fisher(events, lines, model, abs_scale=abs_scale, sca_scale=sca_scale)
    inverted = invert_fishers(all_fishers)
    return inverted, all_fishers
end


function predict_cov_improvement(events::Vector{<:Event}, new_targets, prev_fisher, model::FisherSurrogateModelPerLine; abs_scale, sca_scale)
    
    all_fishers = predict_fisher(events, [new_targets], model, abs_scale=abs_scale, sca_scale=sca_scale)
    
    for (m, m_prev) in zip(eachslice(all_fishers, dims=1), eachslice(prev_fisher, dims=1))
        m .= m + m_prev
    end

    inverted = invert_fishers(all_fishers)

    return inverted, all_fishers
end

end