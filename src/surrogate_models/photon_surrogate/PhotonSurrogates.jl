module PhotonSurrogates

using Flux
using BSON: @save, load
using StructTypes

include("utils.jl")

abstract type ArrivalTimeSurrogate end
abstract type RQSplineModel <: ArrivalTimeSurrogate end

"""
    (m::RQSplineModel)(x, cond)

Evaluate normalizing flow at values `x` with conditional values `cond`.

Returns logpdf
"""
function (m::RQSplineModel)(x, cond)
    params = m.embedding(cond)
    logpdf_eval = eval_transformed_normal_logpdf(x, params, m.range_min, m.range_max)
    return logpdf_eval
end



abstract type PhotonSurrogate end

"""
    struct PhotonSurrogateWithPerturb <: PhotonSurrogate

PhotonSurrogateWithPerturb is a struct that represents a photon surrogate model with medium perturbations.

# Fields
- `amp_model::Chain`: The amplitude model of the surrogate.
- `amp_transformations::Vector{Normalizer}`: The amplitude transformations applied to the surrogate.
- `time_model::RQSplineModel`: The time model of the surrogate.
- `time_transformations::Vector{Normalizer}`: The time transformations applied to the surrogate.
"""
struct PhotonSurrogateWithPerturb <: PhotonSurrogate
    amp_model::Chain
    amp_transformations::Vector{Normalizer}
    time_model::RQSplineModel
    time_transformations::Vector{Normalizer}
end

"""
    struct PhotonSurrogateWithoutPerturb <: PhotonSurrogate

The `PhotonSurrogateWithoutPerturb` struct represents a photon surrogate model without medium perturbations.

# Fields
- `amp_model::Chain`: The amplitude model for the surrogate.
- `amp_transformations::Vector{Normalizer}`: The amplitude transformations applied to the surrogate.
- `time_model::RQSplineModel`: The time model for the surrogate.
- `time_transformations::Vector{Normalizer}`: The time transformations applied to the surrogate.
"""
struct PhotonSurrogateWithoutPerturb <: PhotonSurrogate
    amp_model::Chain
    amp_transformations::Vector{Normalizer}
    time_model::RQSplineModel
    time_transformations::Vector{Normalizer}
end

"""
    PhotonSurrogate(fname_amp, fname_time)

Constructs a photon surrogate model using the given file names for amplitude and time models.
The type of model (`PhotonSurrogateWithoutPerturb` or PhotonSurrogateWithPerturb`) is automatically inferred using the size of the model input layer.

# Arguments
- `fname_amp`: File name of the amplitude model.
- `fname_time`: File name of the time model.

# Returns
- The constructed photon surrogate model.
"""
function PhotonSurrogate(fname_amp, fname_time)

    b1 = load(fname_amp)
    b2 = load(fname_time)

    time_model = b2[:model]
    amp_model = b1[:model]

    Flux.testmode!(time_model)
    Flux.testmode!(amp_model)

    inp_size_time = size(time_model.embedding.layers[1].weights, 2) 
    inp_size_amp = size(amp_model.embedding.layers[1].weights, 2) 

    if inp_size_time == 26 && inp_size_amp == 10
        mtype = PhotonSurrogateWithPerturb
    elseif inp_size_time == 24 && inp_size_amp == 8
        mtype = PhotonSurrogateWithoutPerturb
    else
        error("Cannot parse model inputs.")
    end

    return mtype(b1[:model], b1[:tf_vec], time_model, b2[:tf_vec])
end

Flux.gpu(s::T) where {T <: PhotonSurrogate} = T(gpu(s.amp_model), s.amp_transformations, gpu(s.time_model), s.time_transformations)
Flux.cpu(s::T) where {T <: PhotonSurrogate} = T(cpu(s.amp_model), s.amp_transformations, cpu(s.time_model), s.time_transformations)

abstract type HyperParams end

StructTypes.StructType(::Type{<:HyperParams}) = StructTypes.Struct()


end