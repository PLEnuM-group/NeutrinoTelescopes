export AbsScaPoissonExpModelParams, PoissonExpModelParams

struct PoissonExpModel <: AmplitudeSurrogate
    embedding::Chain
    transformations::Vector{Normalizer{Float32}}
end

# Make embedding parameters trainable
Flux.@functor PoissonExpModel (embedding,)

function (m::PoissonExpModel)(cond)
    output = m.embedding(cond)
    return output
end



Base.@kwdef struct AbsScaPoissonExpModelParams <: HyperParams
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

Base.@kwdef struct PoissonExpModelParams <: HyperParams
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
    setup_model(hparams::PoissonExpModel)

Constructs a model for photon amplitude prediction using a Poisson model.

# Arguments
- `hparams::PoissonExpModelParams`: Hyperparameters for the model.
- `transformations::AbstractVector{<:Normalizer}`: Feature normalizers.

# Returns
- `embedding`: The embedding layer of the model.
- `log_poisson_likelihood`: The log-likelihood function for the Poisson distribution.

"""
function setup_model(hparams::PoissonExpModelParams, transformations::AbstractVector{<:Normalizer})
    hidden_structure = fill(hparams.mlp_layer_size, hparams.mlp_layers)

    non_lins = Dict("relu" => relu, "tanh" => tanh)
    non_lin = non_lins[hparams.non_linearity]

    n_in = 8 
    n_out = 16

    embedding = create_mlp_embedding(
        hidden_structure=hidden_structure,
        n_in=n_in,
        n_out=n_out,
        dropout=hparams.dropout,
        non_linearity=non_lin,
        split_final=false)

    return PoissonExpModel(embedding, transformations), log_poisson_likelihood
    
end

"""
    setup_model(hparams::AbsScaPoissonExpModelParams)

    Constructs a model for photon amplitude prediction using a Poisson model with medium perturbations.

# Arguments
- `hparams::AbsScaPoissonExpModel`: The hyperparameters for the model.
- `transformations::AbstractVector{<:Normalizer}`: Feature normalizers.

# Returns
- `embedding`: The embedding layer of the model.
- `log_poisson_likelihood`: The log Poisson likelihood function.

"""
function setup_model(hparams::AbsScaPoissonExpModelParams, transformations::AbstractVector{<:Normalizer})
    hidden_structure = fill(hparams.mlp_layer_size, hparams.mlp_layers)

    non_lins = Dict("relu" => relu, "tanh" => tanh)
    non_lin = non_lins[hparams.non_linearity]

    n_in = 10 
    n_out = 16

    embedding = create_mlp_embedding(
        hidden_structure=hidden_structure,
        n_in=n_in,
        n_out=n_out,
        dropout=hparams.dropout,
        non_linearity=non_lin,
        split_final=false)

    
    return PoissonExpModel(embedding, transformations), log_poisson_likelihood
end

function create_model_input!(
    model::PoissonExpModel,
    particles::AbstractVector{<:Particle},
    targets::AbstractVector{<:PhotonTarget},
    output;
    abs_scale,
    sca_scale)

   return create_model_input!(particles, targets, output, model.transformations, abs_scale=abs_scale, sca_scale=sca_scale)
end