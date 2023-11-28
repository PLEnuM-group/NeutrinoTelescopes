module FisherSurrogate

using Flux
using BSON
using ..NeuralFlowSurrogate: _calc_flow_input!, apply_feature_transform, Normalizer, HyperParams, create_mlp_embedding
using ..SurrogateModelHits
using ...EventGeneration
using LinearAlgebra
using MLUtils
using PhysicsTools

export FisherSurrogateModel
export FisherSurrogateModelParams
export predict_cov


Base.@kwdef struct FisherSurrogateModelParams <: HyperParams
    mlp_layers::Int64 = 2
    mlp_layer_size::Int64 = 512
    lr::Float64 = 0.001
    dropout::Float64 = 0.1
    non_linearity::Function = relu
    l2_norm_alpha = 0.0
end

function Base.Dict(hp::FisherSurrogateModelParams) 
    d = Dict(
        "mlp_layers" => hp.mlp_layers,
        "mlp_layer_size" => hp.mlp_layer_size,
        "lr" => hp.lr,
        "dropout" => hp.dropout,
        "l2_norm_alpha" => hp.l2_norm_alpha,
        "non_linearity" => string(hp.non_linearity)
    )
    return d
end


struct FisherSurrogateModel
    model::Chain
    tf_in::Vector{Normalizer}
    tf_out::Vector{Normalizer}
    max_valid_distance::Float64
    input_buffer::Matrix{Float32}    
end

function FisherSurrogateModel(fname::String, max_particles=500, max_targets=70*20)
    m = BSON.load(fname)
    Flux.testmode!(m[:model])
    buffer = zeros(Float32, 8, max_targets*max_particles)
    return FisherSurrogateModel(m[:model], m[:tf_in], m[:tf_out], 200., buffer)
end

Flux.gpu(s::FisherSurrogateModel) = FisherSurrogateModel(gpu(s.model), s.tf_in, s.tf_out, s.max_valid_distance, s.input_buffer)
Flux.cpu(s::FisherSurrogateModel) = FisherSurrogateModel(cpu(s.model), s.tf_in, s.tf_out, s.max_valid_distance, s.input_buffer)


function setup_training(hparams::FisherSurrogateModelParams)

    # Instantiate the optimizer
    opt = hparams.l2_norm_alpha > 0 ? Flux.Optimiser(WeightDecay(hparams.l2_norm_alpha), ADAM(hparams.lr)) : ADAM(hparams.lr)

    cov_model = create_mlp_embedding(
        hidden_structure=[hparams.mlp_layer_size for _ in 1:hparams.mlp_layers],
        n_in=8,
        n_out=21,
        dropout=hparams.dropout,
        non_linearity=hparams.non_linearity,
    ) |> gpu

    optim = Flux.setup(opt, cov_model)
    
    return cov_model, optim
end

function make_dataloaders(data)

    train_data, test_data = splitobs(data, at=0.8, shuffle=true)
    train_loader = Flux.DataLoader(train_data, batchsize=2048, shuffle=true)
    test_loader = Flux.DataLoader(test_data, batchsize=size(test_data[1], 2), shuffle=false)

    return train_loader, test_loader
end


function predict_cov(particles, targets, model::FisherSurrogateModel)

    modules_range_mask = get_modules_in_range(particles, targets, model.max_valid_distance)

    if !any(modules_range_mask)
        return nothing
    end

    targets_range = targets[modules_range_mask]

    inv_y_tf = inv.(model.tf_out)
    # particles is the inner loop
    inp = @view _calc_flow_input!(particles, targets_range, model.tf_in, model.input_buffer)[:, 1:(length(particles) * length(targets))]
    
    triu_pred = apply_feature_transform(cpu(model.model(gpu(inp))), inv_y_tf).^3

    fisher_pred_sum = zeros(6,6)
    triu = zeros(6,6)
    triu_ixs = triu!((trues(6,6)))
    @inbounds for trl in eachcol(triu_pred)      
        triu[triu_ixs] .= trl
        fisher_pred_sum .+= (triu'*triu)
    end
    cov_pred_sum = inv(fisher_pred_sum)
    return cov_pred_sum
end


function predict_cov(events::Vector{<:Event}, targets, model::FisherSurrogateModel)
    all_particles::Vector{Particle{Float32}} = [(first(event[:particles])) for event in events]
    
    n_events = length(events)

    normalizers::Vector{Normalizer{Float32}} = model.tf_out

    inv_y_tf = inv.(normalizers)
    # particles is the inner loop

    inp = @view _calc_flow_input!(all_particles, targets, model.tf_in, model.input_buffer)[:, 1:(length(all_particles) * length(targets))]

    triu_pred = cpu(apply_feature_transform(model.model(gpu(inp)), inv_y_tf).^3)
    triu_pred = reshape(triu_pred, (size(triu_pred, 1), n_events, length(targets)))

    triu = zeros(6,6)
    triu_ixs = triu!((trues(6,6)))

    all_fishers = zeros(n_events, 6, 6)

    # Loop over events
    for (i, tp) in enumerate(eachslice(triu_pred, dims=2))

        modules_range_mask = get_modules_in_range(events[i][:particles], targets, model.max_valid_distance)

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
    inverted = Matrix[]
    for m in eachslice(all_fishers, dims=1)
        if isapprox(det(m), 0)
            continue
        end
        push!(inverted, inv(m))
    end

    cov_pred_sum = inverted

end



end