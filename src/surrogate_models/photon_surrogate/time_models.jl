
export NNRQNormFlow
export RQNormFlowHParams, AbsScaRQNormFlowHParams
export create_model_input!


"""
NNRQNormFlow(
    embedding::Chain
    K::Integer,
    range_min::Number,
    range_max::Number,
    )

1-D rq-spline normalizing flow with expected counts prediction.

The rq-spline requires 3 * K + 1 parameters, where `K` is the number of knots. These are
parametrized by an embedding (MLP).

# Arguments
- embedding: Flux model
- range_min: Lower bound of the spline transformation
- range_max: Upper bound of the spline transformation
"""
struct NNRQNormFlow <: RQSplineModel
    embedding::Chain
    K::Integer
    range_min::Float64
    range_max::Float64
    transformations::Vector{Normalizer{Float32}}
end

# Make embedding parameters trainable
Flux.@functor NNRQNormFlow (embedding,)

Base.@kwdef struct RQNormFlowHParams <: HyperParams
    K::Int64 = 10
    batch_size::Int64 = 5000
    mlp_layers::Int64 = 2
    mlp_layer_size::Int64 = 512
    lr::Float64 = 0.001
    lr_min::Float64 = 1E-5
    epochs::Int64 = 50
    dropout::Float64 = 0.1
    non_linearity::String = "relu"
    seed::Int64 = 31338
    l2_norm_alpha = 0.0
    adam_beta_1 = 0.9
    adam_beta_2 = 0.999
    resnet = false
end

Base.@kwdef struct AbsScaRQNormFlowHParams <: HyperParams
    K::Int64 = 10
    batch_size::Int64 = 5000
    mlp_layers::Int64 = 2
    mlp_layer_size::Int64 = 512
    lr::Float64 = 0.001
    lr_min::Float64 = 1E-5
    epochs::Int64 = 50
    dropout::Float64 = 0.1
    non_linearity::String = "relu"
    seed::Int64 = 31338
    l2_norm_alpha = 0.0
    adam_beta_1 = 0.9
    adam_beta_2 = 0.999
    resnet = false
end


"""
    setup_model(hparams::AbsScaRQNormFlowHParams)

Create and initialize a model for the arrival time distribution with medium perturbations.

# Arguments
- `hparams::AbsScaRQNormFlowHParams`: Hyperparameters for the model.
- `transformations::AbstractVector{<:Normalizer}`: Feature normalizers.

# Returns
- `model`: The initialized time prediction model.
- `log_likelihood`: The log-likelihood function of the model.

"""
function setup_model(hparams::AbsScaRQNormFlowHParams, transformations::AbstractVector{<:Normalizer})
    hidden_structure = fill(hparams.mlp_layer_size, hparams.mlp_layers)

    non_lins = Dict("relu" => relu, "tanh" => tanh)
    non_lin = non_lins[hparams.non_linearity]

    # 3 K + 1 for spline, 1 for shift, 1 for scale
    n_spline_params = 3 * hparams.K + 1
    n_out = n_spline_params + 2

    # 3 Rel. Position, 3 Direction, 1 Energy, 1 distance, 1 abs 1 sca
    n_in = 8 + 16 + 2

    embedding = create_mlp_embedding(
        hidden_structure=hidden_structure,
        n_in=n_in,
        n_out=n_out,
        dropout=hparams.dropout,
        non_linearity=non_lin,
        split_final=false)

    model = NNRQNormFlow(embedding, hparams.K, -30.0, 200.0, transformations)
    return model, arrival_time_log_likelihood
end

"""
    setup_model(hparams::RQNormFlowHParams)

Create and initialize a model for the arrival time distribution without medium perturbations.

# Arguments
- `hparams::RQNormFlowHParams`: Hyperparameters for the model.
- `transformations::AbstractVector{<:Normalizer}`: Feature normalizers.

# Returns
- `model`: The initialized neural network model.
- `log_likelihood`: The log-likelihood function of the model.

"""
function setup_model(hparams::RQNormFlowHParams, transformations::AbstractVector{<:Normalizer})
    hidden_structure = fill(hparams.mlp_layer_size, hparams.mlp_layers)

    non_lins = Dict("relu" => relu, "tanh" => tanh)
    non_lin = non_lins[hparams.non_linearity]

    # 3 K + 1 for spline, 1 for shift, 1 for scale
    n_spline_params = 3 * hparams.K + 1
    n_out = n_spline_params + 2

    # 3 Rel. Position, 3 Direction, 1 Energy, 1 distance
    n_in = 8 + 16

    embedding = create_mlp_embedding(
        hidden_structure=hidden_structure,
        n_in=n_in,
        n_out=n_out,
        dropout=hparams.dropout,
        non_linearity=non_lin,
        split_final=false)

    model = NNRQNormFlow(embedding, hparams.K, -30.0, 200.0, transformations)
    return model, arrival_time_log_likelihood
end

function create_model_input!(
    model::NNRQNormFlow,
    particles::AbstractVector{<:Particle},
    targets::AbstractVector{<:PhotonTarget},
    output;
    abs_scale=1., sca_scale=1.)

    n_pmt = get_pmt_count(eltype(targets))
    out_ix = LinearIndices((1:n_pmt, eachindex(particles), eachindex(targets)))
    
    flen = length(model.transformations)

    for (p_ix, t_ix) in product(eachindex(particles), eachindex(targets))
        particle = particles[p_ix]
        target = targets[t_ix]

        for pmt_ix in 1:n_pmt
            ix = out_ix[pmt_ix, p_ix, t_ix]
            outview = @view output[1:flen, ix]
            create_model_input!(particle, target, outview, model.transformations, abs_scale=abs_scale, sca_scale=sca_scale)
        end

        ix = out_ix[1:n_pmt, p_ix, t_ix]
        output[flen+1:flen+n_pmt, ix] .= Matrix(one(eltype(output)) * I, n_pmt, n_pmt)
        
    end

    return output
end



