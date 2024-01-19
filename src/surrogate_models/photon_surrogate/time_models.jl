
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

# Returns
- `model`: The initialized time prediction model.
- `log_likelihood`: The log-likelihood function of the model.

"""
function setup_model(hparams::AbsScaRQNormFlowHParams)
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

    model = NNRQNormFlow(embedding, hparams.K, -30.0, 200.0)
    return model, arrival_time_log_likelihood
end

"""
    setup_model(hparams::RQNormFlowHParams)

Create and initialize a model for the arrival time distribution without medium perturbations.

# Arguments
- `hparams::RQNormFlowHParams`: Hyperparameters for the model.

# Returns
- `model`: The initialized neural network model.
- `log_likelihood`: The log-likelihood function of the model.

"""
function setup_model(hparams::RQNormFlowHParams)
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

    model = NNRQNormFlow(embedding, hparams.K, -30.0, 200.0)
    return model, arrival_time_log_likelihood
end

