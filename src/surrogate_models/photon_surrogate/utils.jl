export Normalizer
export create_mlp_embedding, create_resnet_embedding
export arrival_time_log_likelihood, log_likelihood_with_poisson


struct Normalizer{T}
    mean::T
    σ::T
end

Normalizer(x::AbstractVector) = Normalizer(mean(x), std(x))
(norm::Normalizer)(x::Number) = (x - norm.mean) / norm.σ
Base.inv(n::Normalizer) = x -> x*n.σ + n.mean

Base.convert(::Type{Normalizer{T}}, n::Normalizer) where {T<:Real} = Normalizer(T(n.mean), T(n.σ))

function fit_normalizer!(x::AbstractVector)
    tf = Normalizer(x)
    x .= tf.(x)
    return x, tf
end


"""
    create_mlp_embedding(; hidden_structure::AbstractVector{<:Integer}, n_in, n_out, dropout=0, non_linearity=relu, split_final=false)

Create a multi-layer perceptron (MLP) embedding model.

# Arguments
- `hidden_structure`: An abstract vector of integers representing the number of hidden units in each layer of the MLP.
- `n_in`: The number of input units.
- `n_out`: The number of output units.
- `dropout`: The dropout rate (default: 0).
- `non_linearity`: The activation function to use (default: relu).
- `split_final`: Whether to split the final layer into two separate layers (default: false).

# Returns
A Chain model representing the MLP embedding.

"""
function create_mlp_embedding(;
    hidden_structure::AbstractVector{<:Integer},
    n_in,
    n_out,
    dropout=0,
    non_linearity=relu,
    split_final=false)
    model = []
    push!(model, Dense(n_in => hidden_structure[1], non_linearity))
    push!(model, Dropout(dropout))

    hs_h = hidden_structure[2:end]
    hs_l = hidden_structure[1:end-1]

    for (l, h) in zip(hs_l, hs_h)
        push!(model, Dense(l => h, non_linearity))
        push!(model, Dropout(dropout))
    end

    if split_final
        final = Parallel(vcat,
            Dense(hidden_structure[end] => n_out - 1),
            Dense(hidden_structure[end] => 1)
        )
    else
        #zero_init(out, in) = vcat(zeros(out-3, in), zeros(1, in), ones(1, in), fill(1/in, 1, in))
        final = Dense(hidden_structure[end] => n_out)
    end
    push!(model, final)
    return Chain(model...)
end

"""
    create_resnet_embedding(; hidden_structure::AbstractVector{<:Integer}, n_in, n_out, non_linearity=relu, dropout=0)

Create a ResNet embedding model.

# Arguments
- `hidden_structure`: An abstract vector of integers representing the structure of the hidden layers. All hidden layers must have the same width.
- `n_in`: The number of input features.
- `n_out`: The number of output features.
- `non_linearity`: The non-linearity function to be used in the dense layers. Default is `relu`.
- `dropout`: The dropout rate. Default is 0.

# Returns
A ResNet embedding model.

"""
function create_resnet_embedding(;
    hidden_structure::AbstractVector{<:Integer},
    n_in,
    n_out,
    non_linearity=relu,
    dropout=0
)

    if !all(hidden_structure[1] .== hidden_structure)
        error("For resnet, all hidden layers have to be of same width")
    end

    layer_width = hidden_structure[1]

    model = []
    push!(model, Dense(n_in => layer_width, non_linearity))
    push!(model, Dropout(dropout))

    for _ in 2:length(hidden_structure)
        layer = Dense(layer_width => layer_width, non_linearity)
        drp = Dropout(dropout)
        layer = Chain(layer, drp)
        push!(model, SkipConnection(layer, +))
    end
    push!(model, Dense(layer_width => n_out))

    return Chain(model...)
end


"""
    log_likelihood_with_poisson(x::NamedTuple, model::RQNormFlow)

Evaluate model and return sum of logpdfs of normalizing flow and poisson
"""
function log_likelihood_with_poisson(x::NamedTuple, model::ArrivalTimeSurrogate)

    logpdf_eval, log_expec = model(x[:tres], x[:label])
    non_zero_mask = x[:nhits] .> 0
    logpdf_eval = logpdf_eval .* non_zero_mask

    # poisson: log(exp(-lambda) * lambda^k)
    poiss_f = x[:nhits] .* log_expec .- exp.(log_expec) .- loggamma.(x[:nhits] .+ 1.0)

    # sets correction to nhits of nhits > 0 and to 0 for nhits == 0
    # avoids nans
    correction = x[:nhits] .+ (.!non_zero_mask)

    # correct for overcounting the poisson factor
    poiss_f = poiss_f ./ correction

    return -(sum(logpdf_eval) + sum(poiss_f)) / length(x[:tres])
end


"""
log_likelihood(x::NamedTuple, model::ArrivalTimeSurrogate)

Evaluate model and return sum of logpdfs of normalizing flow
"""
function arrival_time_log_likelihood(x::NamedTuple, model::ArrivalTimeSurrogate)
    logpdf_eval = model(x[:tres], x[:label])
    return -sum(logpdf_eval) / length(x[:tres])
end


function log_poisson_likelihood(x::NamedTuple, model::ArrivalTimeSurrogate)
    
    # one expectation per PMT (16 x batch_size)
    log_expec = model(x[:labels])
    poiss_f = x[:nhits] .* log_expec .- exp.(log_expec) .- loggamma.(x[:nhits] .+ 1.0)

    return -sum(poiss_f) / size(x[:labels], 2)
end