module NeuralFlowSurrogate
using ArraysOfArrays
using MLUtils
using Flux
using TensorBoardLogger
using SpecialFunctions
using Logging
using ProgressLogging
using Random
using EarlyStopping
using DataFrames
using CategoricalArrays
using HDF5
using StatsBase
using LinearAlgebra
using Flux.Optimise
using BSON: @save, load
using Base.Iterators
using PoissonRandom
using LogExpFunctions
using NonNegLeastSquares
using PhysicsTools
using PhotonPropagation
using ParameterSchedulers
using ParameterSchedulers: Scheduler
using StructTypes
using StaticArrays
using CUDA
import Base.GC: gc


using ..RQSplineFlow: eval_transformed_normal_logpdf, sample_flow!
using ...Processing

export ArrivalTimeSurrogate, RQSplineModel, PhotonSurrogate
export kfold_train_model
export get_log_amplitudes, unfold_energy_losses, t_first_likelihood
export track_likelihood_fixed_losses, single_cascade_likelihood, multi_particle_likelihood, track_likelihood_energy_unfolding
export lightsabre_muon_likelihood
export sample_cascade_event, evaluate_model, sample_multi_particle_event
export create_pmt_table, preproc_labels, read_pmt_hits, fit_trafo_pipeline, log_likelihood_with_poisson, read_pmt_number_of_hits
export read_pmt_hits!, read_pmt_number_of_hits!
export calc_flow_input!
export train_model!, RQNormFlowHParams, PoissonExpModel
export setup_optimizer, setup_model, setup_dataloaders
export Normalizer



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


abstract type ArrivalTimeSurrogate end
abstract type RQSplineModel <: ArrivalTimeSurrogate end

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

struct PhotonSurrogate
    amp_model::Chain
    amp_transformations::Vector{Normalizer}
    time_model::RQSplineModel
    time_transformations::Vector{Normalizer}
end

function PhotonSurrogate(fname_amp, fname_time)

    b1 = load(fname_amp)
    b2 = load(fname_time)

    time_model = b2[:model]
    Flux.testmode!(time_model)

    return PhotonSurrogate(b1[:model], b1[:tf_vec], time_model, b2[:tf_vec])

end

Flux.gpu(s::PhotonSurrogate) = PhotonSurrogate(gpu(s.amp_model), s.amp_transformations, gpu(s.time_model), s.time_transformations)
Flux.cpu(s::PhotonSurrogate) = PhotonSurrogate(cpu(s.amp_model), s.amp_transformations, cpu(s.time_model), s.time_transformations)

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
    (m::RQSplineModel)(x, cond)

Evaluate normalizing flow at values `x` with conditional values `cond`.

Returns logpdf and log-expectation
"""
function (m::RQSplineModel)(x, cond)
    params = m.embedding(cond)
    logpdf_eval = eval_transformed_normal_logpdf(x, params, m.range_min, m.range_max)
    return logpdf_eval
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
function log_likelihood(x::NamedTuple, model::ArrivalTimeSurrogate)
    logpdf_eval = model(x[:tres], x[:label])
    return -sum(logpdf_eval) / length(x[:tres])
end


function log_poisson_likelihood(x::NamedTuple, model)
    
    # one expectation per PMT (16 x batch_size)
    log_expec = model(x[:labels])
    poiss_f = x[:nhits] .* log_expec .- exp.(log_expec) .- loggamma.(x[:nhits] .+ 1.0)

    return -sum(poiss_f) / size(x[:labels], 2)
end

abstract type HyperParams end


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


Base.@kwdef struct PoissonExpModel <: HyperParams
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

StructTypes.StructType(::Type{<:HyperParams}) = StructTypes.Struct()

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
    return model, log_likelihood
end

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
    return model, log_likelihood
end



function setup_model(hparams::PoissonExpModel)
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

    
    return embedding, log_poisson_likelihood
end


function setup_dataloaders(train_data, test_data, seed::Integer, batch_size::Integer)
    rng = Random.MersenneTwister(seed)
    train_loader = DataLoader(
        train_data,
        batchsize=batch_size,
        shuffle=true,
        rng=rng)

    test_loader = DataLoader(
        test_data,
        batchsize=50000,
        shuffle=false)

    return train_loader, test_loader
end


function setup_dataloaders(train_data, test_data, hparams::HyperParams)
    setup_dataloaders(train_data, test_data, hparams.seed, hparams.batch_size)   
end

function setup_dataloaders(data, args...)
    train_data, test_data = splitobs(data, at=0.7, shuffle=true)
    return setup_dataloaders(train_data, test_data, args...)
end



# Function to get dictionary of model parameters
function fill_param_dict!(dict, m, prefix)
    if m isa Chain
        for (i, layer) in enumerate(m.layers)
            fill_param_dict!(dict, layer, prefix * "layer_" * string(i) * "/" * string(layer) * "/")
        end
    else
        for fieldname in fieldnames(typeof(m))
            val = getfield(m, fieldname)
            if val isa AbstractArray
                val = vec(val)
            end
            dict[prefix*string(fieldname)] = val
        end
    end
end

sqnorm(x) = sum(abs2, x)


function setup_optimizer(hparams, n_batches)

    opt = Adam(hparams.lr, (hparams.adam_beta_1, hparams.adam_beta_2))
    if hparams.l2_norm_alpha > 0
        opt = OptimiserChain(WeightDecay(hparams.l2_norm_alpha), opt)
    end
    schedule = Interpolator(CosAnneal(λ0=hparams.lr_min, λ1=hparams.lr, period=hparams.epochs), n_batches)
    return Scheduler(schedule, opt)
end




function train_model!(;
    optimizer,
    train_loader,
    test_loader,
    model,
    loss_function,
    hparams,
    logger,
    device,
    use_early_stopping,
    checkpoint_path=nothing)


    model = model |> device
    pars = Flux.params(model)

    if use_early_stopping
        stopper = EarlyStopper(Warmup(Patience(5); n=3), InvalidValue(), NumberSinceBest(n=5), verbosity=1)
    else
        stopper = EarlyStopper(Never(), verbosity=1)
    end

    local loss
    local total_test_loss

    best_test = Inf
    best_test_epoch = 0

    t = time()
    @progress for epoch in 1:hparams.epochs
        Flux.trainmode!(model)

        total_train_loss = 0.0
        for d in train_loader
            d = d |> device
            gs = gradient(pars) do
                loss = loss_function(d, model)

                return loss
            end
            total_train_loss += loss
            Flux.update!(optimizer, pars, gs)
        end

        total_train_loss /= length(train_loader)

        Flux.testmode!(model)
        total_test_loss = 0
        for d in test_loader
            d = d |> device
            total_test_loss += loss_function(d, model)
        end
        total_test_loss /= length(test_loader)

        #param_dict = Dict{String,Any}()
        #fill_param_dict!(param_dict, model, "")

        if !isnothing(logger)
            with_logger(logger) do
                @info "loss" train = total_train_loss test = total_test_loss
                @info "hparams" lr = optimizer.optim.eta log_step_increment = 0
                #@info "model" params = param_dict log_step_increment = 0
            end

        end
        println("Epoch: $epoch, Train: $total_train_loss Test: $total_test_loss")

        if !isnothing(checkpoint_path) && epoch > 5 && total_test_loss < best_test
            @save checkpoint_path * "_BEST.bson" model
            best_test = total_test_loss
            best_test_epoch = epoch
        end

        done!(stopper, total_test_loss) && break

    end
    return model, total_test_loss, best_test, best_test_epoch, time() - t
end





function dataframe_to_matrix(df)
    feature_matrix = Matrix{Float64}(undef, 9, nrow(df))
    feature_matrix[1, :] .= log.(df[:, :distance])
    feature_matrix[2, :] .= log.(df[:, :energy])

    feature_matrix[3:5, :] .= reduce(hcat, sph_to_cart.(df[:, :dir_theta], df[:, :dir_phi]))
    feature_matrix[6:8, :] .= reduce(hcat, sph_to_cart.(df[:, :pos_theta], df[:, :pos_phi]))
    feature_matrix[9, :] .= df[:, :pmt_id]
    return feature_matrix
end

function apply_feature_transform(m, tf_vec)
    # Not mutating version...
    tf_matrix = mapreduce(
        t -> permutedims(t[2].(t[1])),
        vcat,
        zip(eachrow(m), tf_vec)
    )
    return tf_matrix
end

function apply_feature_transform!(m, tf_vec, output)
    # Mutating version...
    for (in_row, out_row, tf) in zip(eachrow(m), eachrow(output), tf_vec)
        out_row .= tf.(in_row)
    end
    return output
end

function initialize_normalizers(feature_matrix)
    tf_vec = Vector{Normalizer{Float64}}(undef, size(feature_matrix, 1))
    for (row, ix) in zip(eachrow(feature_matrix), eachindex(tf_vec))
        tf = Normalizer(row)
        tf_vec[ix] = tf
    end

    return tf_vec
end

function preproc_labels!(feature_matrix::AbstractMatrix, output, tf_vec=nothing)
    if isnothing(tf_vec)
        tf_vec = initialize_normalizers(feature_matrix)       
    end
    feature_matrix = apply_feature_transform!(feature_matrix, tf_vec, output)
    return feature_matrix, tf_vec
end



function append_onehot_pmt!(output, pmt_ixs)
    lev = 1:16
    output[end-15:end, :] .= (lev .== permutedims(pmt_ixs))
    return output
end


function count_hit_per_pmt(grp)

    feature_vector = zeros(Float64, 8)
    hit_vector = zeros(Float64, 16)

    grp_attrs = attrs(grp)
    
    feature_vector[1] = log.(grp_attrs["distance"])
    feature_vector[2] = log.(grp_attrs["energy"])

    feature_vector[3:5] = reduce(hcat, sph_to_cart.(grp_attrs["dir_theta"], grp_attrs["dir_phi"]))
    feature_vector[6:8] = reduce(hcat, sph_to_cart.(grp_attrs["pos_theta"], grp_attrs["pos_phi"]))

    if size(grp, 1) == 0
        return feature_vector, hit_vector
    end

    hits = DataFrame(grp[:, :], [:tres, :pmt_id, :total_weight])

    hits_per_pmt = combine(groupby(hits, :pmt_id), :total_weight => sum => :weight_sum)
    pmt_id_ix = Int.(hits_per_pmt[:, :pmt_id])
    hit_vector[pmt_id_ix] .= hits_per_pmt[:, :weight_sum]

    return feature_vector, hit_vector
end

function create_pmt_table(grp, limit=nothing)

    grp_data = grp[:, :]
    grp_attrs = attrs(grp)

    grplen = size(grp_data, 1)

    sumw = sum(grp_data[:, 3])
    weights = FrequencyWeights(grp_data[:, 3], sumw)
    sampled = sample(1:grplen, weights, ceil(Int64, sumw), replace=true)

    grp_data = grp_data[sampled, :]

    out_length = !isnothing(limit) ? min(limit, size(grp, 1)) : size(grp, 1)

    if haskey(grp_attrs, "abs_scale")
        feature_dim = 10
    else
        feature_dim = 8
    end


    feature_matrix = zeros(Float64, feature_dim, out_length)
    hit_times = zeros(Float64, out_length)

    feature_matrix[1, :] .= log.(grp_attrs["distance"])
    feature_matrix[2, :] .= log.(grp_attrs["energy"])

    feature_matrix[3:5, :] .= permutedims(reduce(hcat, sph_to_cart.(grp_attrs["dir_theta"], grp_attrs["dir_phi"])))
    feature_matrix[6:8, :] .= permutedims(reduce(hcat, sph_to_cart.(grp_attrs["pos_theta"], grp_attrs["pos_phi"])))

    if haskey(grp_attrs, "abs_scale")
        feature_matrix[9, :] .= grp_attrs["abs_scale"]
        feature_matrix[10, :] .= grp_attrs["sca_scale"]
    end


    data_mat = grp[1:out_length, :]

    hit_times = data_mat[:, 1]
    pmt_ixs = data_mat[:, 2]


    return feature_matrix, pmt_ixs, hit_times

end

function read_pmt_number_of_hits!(fnames, nhits_buffer, features_buffer, nsel_frac=0.8, rng=default_rng())
    ix = 1
    for fname in fnames
        h5open(fname, "r") do fid
            if !isnothing(rng)
                datasets = shuffle(rng, keys(fid["pmt_hits"]))
            else
                datasets = keys(fid["pmt_hits"])
            end

            if nsel_frac == 1
                index_end = length(datasets)
            else
                index_end = Int(ceil(length(datasets) * nsel_frac))
            end

            for grpn in datasets[1:index_end]
                grp = fid["pmt_hits"][grpn]
                f,h = count_hit_per_pmt(grp)

                nhits_buffer[:, ix] .= h
                features_buffer[:, ix] .= f
                ix += 1
            end

        end
    end

    features_buffer_view = @view features_buffer[:, 1:ix]
    nhits_buffer_view = @view nhits_buffer[:, 1:ix]

    features_buffer_view, tf_vec = preproc_labels!(features_buffer_view, features_buffer_view)

    return nhits_buffer_view, features_buffer_view, tf_vec
end


function read_pmt_hits!(fnames, hit_buffer, pmt_ixs_buffer, features_buffer, nsel_frac=0.8, rng=default_rng())

    ix = 1
    for fname in fnames
        h5open(fname, "r") do fid
            if !isnothing(rng)
                datasets = shuffle(rng, keys(fid["pmt_hits"]))
            else
                datasets = keys(fid["pmt_hits"])
            end

            if nsel_frac == 1
                index_end = length(datasets)
            else
                index_end = Int(ceil(length(datasets) * nsel_frac))
            end

            for grpn in datasets[1:index_end]
                grp = fid["pmt_hits"][grpn]
                if size(grp, 1) == 0
                    continue
                end
                f, pix, h = create_pmt_table(grp, 1000)
                nhits = length(h)

                if ix+nhits-1 > length(hit_buffer)
                    error("Buffer overflow")
                end

                hit_buffer[ix:ix+nhits-1] .= h
                features_buffer[1:size(f, 1), ix:ix+nhits-1] .= f
                pmt_ixs_buffer[ix:ix+nhits-1] .= pix
                ix += nhits
            end
        end
    end

    pmt_ixs_buffer_view = @view pmt_ixs_buffer[1:ix]
    features_buffer_view = @view features_buffer[:, 1:ix]
    hit_buffer_view = @view hit_buffer[1:ix]

    features_buffer, tf_vec = preproc_labels!(features_buffer_view, features_buffer_view)
    append_onehot_pmt!(features_buffer_view, pmt_ixs_buffer_view)
    return hit_buffer_view, features_buffer_view, tf_vec
end


function _calc_flow_input!(
    particle_pos,
    particle_dir,
    particle_energy,
    target_pos,
    tf_vec::AbstractVector,
    output)

    rel_pos = particle_pos .- target_pos
    dist = norm(rel_pos)
    normed_rel_pos = rel_pos ./ dist

    @inbounds begin
        output[1] = tf_vec[1](log(dist))
        output[2] = tf_vec[2](log(particle_energy))
        output[3] = tf_vec[3](particle_dir[1])
        output[4] = tf_vec[4](particle_dir[2])
        output[5] = tf_vec[5](particle_dir[3])
        output[6] = tf_vec[6](normed_rel_pos[1])
        output[7] = tf_vec[7](normed_rel_pos[2])
        output[8] = tf_vec[8](normed_rel_pos[3])
        #features, _ = preproc_labels(features, tf_vec)
        #apply_feature_transform!(output, tf_vec, output)
    end

    return output
end



function _calc_flow_input!(
    particle::Particle,
    target::PhotonTarget,
    tf_vec::AbstractVector,
    output)

    if particle_shape(particle) == Track()
        particle = shift_to_closest_approach(particle, target.shape.position)
    end

    return _calc_flow_input!(particle.position, particle.direction, particle.energy, target.shape.position, tf_vec, output)
    
end



"""
    _calc_flow_input!(particles::AbstractVector{<:Particle}, targets::AbstractVector{<:PhotonTarget}, tf_vec, output)

Mutating version. Flow input is written into output
"""
function _calc_flow_input!(particles::AbstractVector{<:Particle}, targets::AbstractVector{<:PhotonTarget}, tf_vec, output)
  

    out_ix = LinearIndices((eachindex(particles), eachindex(targets)))
    
    for (p_ix, t_ix) in product(eachindex(particles), eachindex(targets))
        particle = particles[p_ix]
        target = targets[t_ix]

        ix = out_ix[p_ix, t_ix]
        _calc_flow_input!(particle, target, tf_vec, @view output[:, ix])
        
    end

    return output
end




"""
    calc_flow_input(particle::Particle, target::PhotonTarget, tf_vec::AbstractVector)

Transform a particle and a target into input features for a neural netork.

The resulting feature matrix contains the log distance of particle and target,
the log of the particle energy, the particle direction (in cartesian coordinates) and the
direction from target to particle.
These columns are repeated for the number of pmts on the target. The final column is the
pmt index.
The resulting feature matrix is then transformed column by column using the transformations
included in `tf_vec`.
"""
function calc_flow_input(particle::Particle, target::PhotonTarget, tf_vec::AbstractVector) 
    f = _calc_flow_input(particle, target, tf_vec)
    n_pmt = get_pmt_count(target)
    features::Matrix{eltype(f)} = repeat(f, 1, n_pmt)
    features = append_onehot_pmt(features, 1:n_pmt)
    return features
end



"""
    calc_flow_input(particles::AbstractVector{<:Particle}, targets::AbstractVector{<:PhotonTarget}, tf_vec)

For vectors of particles and targets, the resulting feature matrices for each combination
are catted together.
The product of `particles` and `targets` is packed densely into a single dimension with
embedding structure: [(p_1, t_1), (p_2, t_1), ... (p_1, t_end), ... (p_end, t_end)]
"""
function calc_flow_input(particles::AbstractVector{<:Particle}, targets::AbstractVector{<:PhotonTarget}, tf_vec)

    res = mapreduce(hcat, product(particles, targets)) do pt

        particle = pt[1]
        target = pt[2]
        
        return calc_flow_input(particle, target, tf_vec)       
    end

    return res
end

"""
    calc_flow_input!(particle::P, target::PhotonTarget, tf_vec::AbstractVector, output)
Mutating version of `calc_flow_input`.
"""
function calc_flow_input!(particle::Particle, target::PhotonTarget, tf_vec::AbstractVector, output)
    _calc_flow_input!(particle, target, tf_vec, @view output[1:8])
    
    n_pmt = get_pmt_count(target)
    lev = 1:n_pmt
    output[9:9+n_pmt-1, :] .= (lev .== permutedims(pmt_ixs))
    return output
end


"""
    calc_flow_input!(particles::AbstractVector{<:Particle}, targets::AbstractVector{<:PhotonTarget}, tf_vec, output)

Mutating version. Flow input is written into output
"""
function calc_flow_input!(particles::AbstractVector{<:Particle}, targets::AbstractVector{<:PhotonTarget}, tf_vec, output)
    n_pmt = get_pmt_count(eltype(targets))
    out_ix = LinearIndices((1:n_pmt, eachindex(particles), eachindex(targets)))
    
    for (p_ix, t_ix) in product(eachindex(particles), eachindex(targets))
        particle = particles[p_ix]
        target = targets[t_ix]

        for pmt_ix in 1:n_pmt
            ix = out_ix[pmt_ix, p_ix, t_ix]
            _calc_flow_input!(particle, target, tf_vec, @view output[1:8, ix])
        end

        ix = out_ix[1:n_pmt, p_ix, t_ix]
        output[9:9+n_pmt-1, ix] .= Matrix(one(eltype(output)) * I, n_pmt, n_pmt)
        
    end

    return output
end


function poisson_logpmf(n, log_lambda)
    return n * log_lambda - exp(log_lambda) - loggamma(n + 1.0)
end

"""
    create_non_uniform_ranges(n_per_split::AbstractVector)

Create a vector of UnitRanges, where each range consecutively select n from n_per_split
"""
function create_non_uniform_ranges(n_per_split::AbstractVector)
    vtype = Union{UnitRange{Int64},Missing}
    output = Vector{vtype}(undef, length(n_per_split))
    ix = 1
    for (i, n) in enumerate(n_per_split)
        output[i] = n > 0 ? (ix:ix+n-1) : missing
        ix += n
    end
    return output
end


function _reshape_and_timeshift_data!(times, particles, targets, medium, n_hits_per_pmt_source, output_buffer)

    n_pmt = get_pmt_count(eltype(targets))
    n_hits_per_source = vec(n_hits_per_pmt_source)

    # Create range selectors into times for each particle-pmt pair
    selectors = reshape(
        create_non_uniform_ranges(n_hits_per_source),
        n_pmt, length(particles), length(targets)
    )
    times = flatview(times)

    for (pmt_ix, t_ix) in product(1:n_pmt, eachindex(targets))
        target = targets[t_ix]

        @views n_hits_this_target = sum(n_hits_per_pmt_source[pmt_ix, :, t_ix])

        if n_hits_this_target == 0
            push!(output_buffer, Float64[])
            continue
        end

        data_this_target = Vector{Float64}(undef, n_hits_this_target)
        data_selectors = create_non_uniform_ranges(n_hits_per_pmt_source[pmt_ix, :, t_ix])

        for p_ix in eachindex(particles)
            particle = particles[p_ix]
            this_n_hits = n_hits_per_pmt_source[pmt_ix, p_ix, t_ix]
            if this_n_hits > 0
                t_geo = calc_tgeo(particle, target, medium)
                times_sel = selectors[pmt_ix, p_ix, t_ix]
                data_this_target[data_selectors[p_ix]] = times[times_sel] .+ t_geo .+ particle.time
            end
        end

        push!(output_buffer, data_this_target)
    end
    return output_buffer
end


function _prepare_flow_inputs(particles, targets, model, feat_buffer, device, rng; oversample=1)
    n_pmt = get_pmt_count(eltype(targets))

    _, log_expec_per_src_pmt_rs = get_log_amplitudes(particles, targets, model; feat_buffer=feat_buffer, device=device)

    if !isnothing(feat_buffer)
        shape_buffer = @view feat_buffer[:, 1:length(particles)*length(targets)*n_pmt]
        calc_flow_input!(particles, targets, model.time_transformations, shape_buffer)
        input = shape_buffer
    else
        input = calc_flow_input(particles, targets, model.time_transformations)
    end
    
    # The embedding for all the parameters is
    # [(p_1, t_1, pmt_1), (p_1, t_1, pmt_2), ... (p_2, t_1, pmt_1), ... (p_1, t_end, pmt_1), ... (p_end, t_end, pmt_end)]

    flow_params = cpu(model.time_model.embedding(device(input)))

    expec_per_source_rs = log_expec_per_src_pmt_rs .= exp.(log_expec_per_src_pmt_rs) .* oversample

    n_hits_per_source_rs = pois_rand.(rng, Float64.(expec_per_source_rs))
   
    return flow_params, n_hits_per_source_rs
end

function _sample_times_for_particle(particles, targets, model, output_buffer, rng=Random.default_rng(); oversample=1, feat_buffer=nothing, device=gpu)
    
    flow_params, n_hits_per_source_rs = _prepare_flow_inputs(particles, targets, model, feat_buffer, device, rng, oversample=oversample)
    
    n_hits_per_source = vec(n_hits_per_source_rs)
    mask = n_hits_per_source .> 0
    non_zero_hits = n_hits_per_source[mask]

    # Only sample times for particle-target pairs that have at least one hit
    times = sample_flow!(flow_params[:, mask], model.time_model.range_min, model.time_model.range_max, non_zero_hits, output_buffer, rng=rng)
    return times, n_hits_per_source_rs
end

"""
    sample_multi_particle_event(particles, targets, model, medium, rng=nothing; oversample=1, feat_buffer=nothing, output_buffer=nothing)

Sample arrival times at `targets` for `particles` using `model`.
"""
function sample_multi_particle_event(
    particles,
    targets,
    model,
    medium,
    rng=Random.default_rng();
    oversample=1,
    feat_buffer=nothing,
    output_buffer=nothing,
    device=gpu)

    # We currently cannot reshape VectorOfArrays. For now just double allocate
    temp_output_buffer = VectorOfArrays{Float64, 1}()
    n_pmts = get_pmt_count(eltype(targets))*length(targets)
    sizehint!(temp_output_buffer, n_pmts, (100, ))

    times, n_hits_per_pmt_source = _sample_times_for_particle(particles, targets, model, temp_output_buffer, rng, oversample=oversample, feat_buffer=feat_buffer, device=device)

    if isnothing(output_buffer)
        output_buffer = VectorOfArrays{Float64, 1}()
        sizehint!(output_buffer, n_pmts, (100, ))
    end
    empty!(output_buffer)


    _reshape_and_timeshift_data!(times, particles, targets, medium, n_hits_per_pmt_source, output_buffer)

    return output_buffer
end


"""
    sample_many_multi_particle_events(particles, targets, model, medium, rng=nothing; oversample=1, feat_buffer=nothing, output_buffer=nothing)

Sample arrival times at `targets` for `particles` using `model`.
"""
function sample_many_multi_particle_events(
    vec_particles::VectorOfArrays,
    targets,
    model,
    medium,
    rng=Random.default_rng();
    oversample=1,
    feat_buffer=nothing,
    output_buffer=nothing,
    device=gpu)

    # We currently cannot reshape VectorOfArrays. For now just double allocate
    temp_output_buffer = VectorOfArrays{Float64, 1}()
    n_pmts = get_pmt_count(eltype(targets))*length(targets)
    sizehint!(temp_output_buffer, n_pmts, (100, ))

    particles_flat = flatview(vec_particles)
    times, n_hits_per_pmt_source = _sample_times_for_particle(particles_flat, targets, model, temp_output_buffer, rng, oversample=oversample, feat_buffer=feat_buffer, device=device)


    if isnothing(output_buffer)
        output_buffer = VectorOfArrays{Float64, 1}()
        sizehint!(output_buffer, n_pmts, (100, ))
    end
    empty!(output_buffer)


    _reshape_and_timeshift_data!(times, particles, targets, medium, n_hits_per_pmt_source, output_buffer)

    return output_buffer
end

"""
    sample_cascade_event(energy, dir_theta, dir_phi, position, time; targets, model, tf_vec, medium, rng=nothing)
Sample photon times at `targets` for a cascade.
"""
function sample_cascade_event(energy, dir_theta, dir_phi, position, time; targets, model, medium, rng=nothing)

    dir = sph_to_cart(dir_theta, dir_phi)
    particle = Particle(position, dir, time, energy, 0.0, PEMinus)
    return sample_multi_particle_event([particle], targets, model, medium, rng)
end

"""
get_log_amplitudes(particles, targets, model::PhotonSurrogate; feat_buffer=nothing)

Evaluate `model` for `particles` and `targets`

Returns:
    -log_expec_per_pmt: Log of expected photons per pmt. Shape: [n_pmt, 1, n_targets]
    -log_expec_per_src_pmt_rs: Log of expected photons per pmt and per particle. Shape [n_pmt, n_particles, n_targets]
"""
function get_log_amplitudes(particles, targets, model::PhotonSurrogate; feat_buffer=nothing, device=gpu)
    n_pmt = get_pmt_count(eltype(targets))

    tf_vec = model.amp_transformations

    if isnothing(feat_buffer)
        input = _calc_flow_input(particles, targets, tf_vec)
    else
        amp_buffer = @view feat_buffer[1:8, 1:length(targets)*length(particles)]
        _calc_flow_input!(particles, targets, tf_vec, amp_buffer)
        input = amp_buffer
    end

    input = permutedims(input)'

    log_expec_per_src_trg::Matrix{eltype(input)} = cpu(model.amp_model(device(input)))

    log_expec_per_src_pmt_rs = reshape(
        log_expec_per_src_trg,
        n_pmt, length(particles), length(targets))

    log_expec_per_pmt = LogExpFunctions.logsumexp(log_expec_per_src_pmt_rs, dims=2)

    return log_expec_per_pmt, log_expec_per_src_pmt_rs
end


function shape_llh_generator(data; particles, targets, flow_params, rel_log_expec, model, medium)

    n_pmt = get_pmt_count(eltype(targets))
    data_ix = LinearIndices((1:n_pmt, eachindex(targets)))
    ix = LinearIndices((1:n_pmt, eachindex(particles), eachindex(targets)))

    shape_llh_gen = (
        length(data[data_ix[pmt_ix, t_ix]]) > 0 ?
        # Reduce over the particle dimension to create the mixture
        sum(LogExpFunctions.logsumexp(
            # Evaluate the flow for each time and each particle and stack result
            reduce(
                hcat,
                # Mixture weights
                rel_log_expec[pmt_ix, p_ix, t_ix] .+
                # Returns vector of logl for each time in data
                eval_transformed_normal_logpdf(
                    data[data_ix[pmt_ix, t_ix]] .- calc_tgeo(particles[p_ix], targets[t_ix], medium) .- particles[p_ix].time,
                    flow_params[:, ix[pmt_ix, p_ix, t_ix]],
                    model.range_min,
                    model.range_max
                )
                for p_ix in eachindex(particles)
            ),
            dims=2
        ))
        : 0.0
        for (pmt_ix, t_ix) in product(1:n_pmt, eachindex(targets)))
    return shape_llh_gen
end

function shape_llh(data; particles, targets, flow_params, rel_log_expec, model, medium)

    n_pmt = get_pmt_count(eltype(targets))
    data_ix = LinearIndices((1:n_pmt, eachindex(targets)))
    ix = LinearIndices((1:n_pmt, eachindex(particles), eachindex(targets)))
    out_ix = LinearIndices((1:n_pmt, eachindex(targets)))

    T = eltype(flow_params)
    shape_llh = zeros(eltype(flow_params), n_pmt*length(targets))

    @inbounds for (pmt_ix, t_ix) in product(1:n_pmt, eachindex(targets))
        this_data_len = length(data[data_ix[pmt_ix, t_ix]])
        if this_data_len == 0 
            continue
        end

        acc = fill(T(-Inf), this_data_len)

        for p_ix in eachindex(particles)
            # Mixture weights 
            rel_expec_per_part = rel_log_expec[pmt_ix, p_ix, t_ix]

            this_flow_params = @views flow_params[:, ix[pmt_ix, p_ix, t_ix]]

            # Mixture Pdf
            shape_pdf = eval_transformed_normal_logpdf(
                    data[data_ix[pmt_ix, t_ix]] .- calc_tgeo(particles[p_ix], targets[t_ix], medium) .- particles[p_ix].time,
                    this_flow_params,
                    model.range_min,
                    model.range_max
            )

            acc = @. logaddexp(acc, rel_expec_per_part + shape_pdf)
        end

        shape_llh[out_ix[pmt_ix, t_ix]] = sum(acc)
    end

    return shape_llh
end


function evaluate_model(
    particles::AbstractVector{<:Particle},
    data, targets, model::PhotonSurrogate, medium; feat_buffer=nothing, device=gpu)

    log_expec_per_pmt, log_expec_per_src_pmt_rs = get_log_amplitudes(particles, targets, model; feat_buffer=feat_buffer, device=device)
    rel_log_expec = log_expec_per_src_pmt_rs .= log_expec_per_src_pmt_rs .- log_expec_per_pmt

    hits_per_target = length.(data)
    # Flattening log_expec_per_pmt with [:] will let the first dimension be the inner one
    poiss_llh = poisson_logpmf.(hits_per_target, vec(log_expec_per_pmt[:]))

    npmt = get_pmt_count(eltype(targets))


    if isnothing(feat_buffer)
        input = calc_flow_input(particles, targets, model.time_transformations)
    else
        input = @view feat_buffer[:, 1:length(targets)*length(particles)*npmt]
        calc_flow_input!(particles, targets, model.time_transformations, feat_buffer)
    end

    flow_params::Matrix{eltype(input)} = cpu(model.time_model.embedding(device(input)))
    #flow_params = model.time_model.embedding(device(input))


    #sllh = shape_llh(data; particles=particles, targets=targets, flow_params=flow_params, rel_log_expec=rel_log_expec, model=model.time_model, medium=medium)
    sllh = shape_llh_generator(data; particles=particles, targets=targets, flow_params=flow_params, rel_log_expec=rel_log_expec, model=model.time_model, medium=medium)
    
    
    return poiss_llh, sllh, log_expec_per_pmt
end


function multi_particle_likelihood(
    particles::AbstractVector{<:Particle};
    data::AbstractVector{<:AbstractVector{<:Real}},
    targets::AbstractVector{<:PhotonTarget},
    model::PhotonSurrogate, medium, feat_buffer=nothing, amp_only=false, device=gpu)

    n_pmt = get_pmt_count(eltype(targets))
    @assert length(targets) * n_pmt == length(data)
    pois_llh, shape_llh, _ = evaluate_model(particles, data, targets, model, medium, feat_buffer=feat_buffer, device=device)
    if amp_only
        return sum(pois_llh)
    else
        return sum(pois_llh) + sum(shape_llh, init=0.)
    end


end

#=
function noise_likelihood(data; particles, targets, medium, noise_window_length)
    n_pmt = get_pmt_count(eltype(targets))
    data_ix = LinearIndices((1:n_pmt, eachindex(targets)))
    ix = LinearIndices((1:n_pmt, eachindex(particles), eachindex(targets)))

    data_ix[pmt_ix, t_ix]

    llh_w_noise = logsumexp(log(p_noise)- log((noise_window_length)) , log(1-p_noise) + log(s_lh))

    llh_with_noise = log_poisson_w_nois + log()


end
=#


function single_cascade_likelihood(
    logenergy::Real,
    dir_theta::Real,
    dir_phi::Real,
    position::AbstractArray{<:Real},
    time::Real; data, targets, model, medium, feat_buffer=nothing)
    
    T = promote_type(typeof(dir_theta), typeof(dir_phi))
    
    dir::SVector{3, T} = sph_to_cart(dir_theta, dir_phi)
    energy = 10. .^logenergy
    particles = [Particle(position, dir, time, energy, 0.0, PEMinus)]
    return multi_particle_likelihood(particles, data=data, targets=targets, model=model, medium=medium, feat_buffer=feat_buffer)
end

function lightsabre_muon_likelihood(
    logenergy::Real,
    dir_theta::Real,
    dir_phi::Real,
    position::AbstractArray{<:Real},
    time::Real; data, targets, model, medium, feat_buffer=nothing)
    
    T = promote_type(typeof(dir_theta), typeof(dir_phi))
    
    dir::SVector{3, T} = sph_to_cart(dir_theta, dir_phi)
    energy = 10. .^logenergy
    particles = [Particle(position, dir, time, energy, 100000, PMuMinus)]
    return multi_particle_likelihood(particles, data=data, targets=targets, model=model, medium=medium, feat_buffer=feat_buffer)
end



function track_likelihood_fixed_losses(
    logenergy, dir_theta, dir_phi, position;
    losses, muon_energy, data, targets, model, tf_vec, medium, feat_buffer=nothing, amp_only=false)

    energy = 10^logenergy
    dir = sph_to_cart(dir_theta, dir_phi)
    dist_along = norm.([p.position .- position for p in losses])


    new_loss_positions = [position .+ d .* dir for d in dist_along]
    new_loss_energies = [p.energy / muon_energy * energy for p in losses]

    times = [p.time for p in losses]

    new_losses = Particle.(new_loss_positions, [dir], times, new_loss_energies, 0.0, [PEMinus])

    return multi_particle_likelihood(
        new_losses, data=data, targets=targets, model=model, tf_vec=tf_vec, medium=medium, feat_buffer=feat_buffer, amp_only=amp_only)
end


function t_first_likelihood(particles; data, targets, model, tf_vec, medium, feat_buffer=nothing)

    log_expec_per_pmt, log_expec_per_src_pmt_rs, flow_params = get_log_amplitudes(particles, targets, model, tf_vec; feat_buffer=feat_buffer)
    rel_log_expec = log_expec_per_src_pmt_rs .- log_expec_per_pmt

    # data masking
    t_first = [length(d) > 0 ? minimum(d) : -inf]


    llh_tfirst = shape_llh_generator(t_first; particles=particles, targets=targets, flow_params=flow_params, rel_log_expec=rel_log_expec, model=model, medium=medium)

    upper = t_first .+ 200.0

    p_later = integral_norm_flow(flow_params, t_first, upper, model.range_min, model.range_max)

    llh = llh_tfirst .+ (exp.(log_expec_per_pmt) .- 1) .* log((1 .- p_later))

    return sum(llh)
end




function build_loss_vector(position, direction, energy, time, spacing, length)

    dist_along = (-length/2):spacing:(length/2)

    loss_positions = [position .+ direction .* d for d in dist_along]
    loss_times = time .+ dist_along ./ c_vac_m_ns

    new_losses = Particle.(loss_positions, [direction], loss_times, energy, 0.0, [PEMinus])

    return new_losses
end

function unfold_energy_losses(position, direction, time; data, targets, model, tf_vec, spacing, plength=500.0)
    e_base = 5E4
    lvec = build_loss_vector(position, direction, e_base, time, spacing, plength)

    n_pmt = get_pmt_count(eltype(targets))
    feat_buffer = zeros(9, n_pmt * length(targets) * length(lvec))
    _, log_amp_per_src_pmt = get_log_amplitudes(lvec, targets, gpu(model), tf_vec; feat_buffer=feat_buffer)
    log_amp_per_src_module = LogExpFunctions.logsumexp(log_amp_per_src_pmt, dims=1)[1, :, :]

    data_rs = reshape(data, n_pmt, Int(length(data) / n_pmt))
    hits_per_pmt = length.(data)
    hits_per_module = sum(length.(data_rs), dims=1)[:]

    #=
    hit_mask = hits_per_pmt .>= 1
    hits_per_pmt_masked = hits_per_pmt[hit_mask]
    perm = permutedims(log_amp_per_src_pmt, (1, 3, 2))
    source_exp = exp.(reshape(perm, prod(size(perm)[1:2]), size(perm)[3]))[hit_mask, :]
    escales = nonneg_lsq(source_exp, Float64.(hits_per_pmt_masked), alg=:nnls)[:]
    =#

    source_exp = permutedims(exp.(log_amp_per_src_module))
    escales = nonneg_lsq(source_exp, Float64.(hits_per_module), alg=:nnls)[:]


    non_zero = escales .> 0
    lvec_masked = lvec[non_zero]

    for (l, escale) in zip(lvec_masked, escales[non_zero])
        l.energy = escale * e_base
    end
    return lvec_masked
end


function track_likelihood_energy_unfolding(dir_theta, dir_phi, position, time; spacing, data, targets, model, tf_vec, medium, amp_only=false)

    dir = sph_to_cart(dir_theta, dir_phi)
    losses = unfold_energy_losses(position, dir, time; data=data, targets=targets, model=model, tf_vec=tf_vec, spacing=spacing)
    if length(losses) == 0
        return -Inf64
    end
    return multi_particle_likelihood(losses, data=data, targets=targets, model=model, tf_vec=tf_vec, medium=medium, amp_only=amp_only)
end

function kfold_train_model(data, outpath, model_name, tf_vec, n_folds, hparams::HyperParams)
    
    logdir = joinpath(@__DIR__, "../../tensorboard_logs/$model_name")

    model_stats = []

    for (model_num, (train_data, val_data)) in enumerate(kfolds(data; k=n_folds))
        gc()
        lg = TBLogger(logdir)
        model, loss_f = setup_model(hparams)
        chk_path = joinpath(outpath, "$(model_name)_$(model_num)")

        train_loader, test_loader = setup_dataloaders(train_data, val_data, hparams)
        opt = setup_optimizer(hparams, length(train_loader))
        device = gpu
        model, final_test_loss, best_test_loss, best_test_epoch, time_elapsed = train_model!(
            optimizer=opt,
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            loss_function=loss_f,
            hparams=hparams,
            logger=lg,
            device=device,
            use_early_stopping=true,
            checkpoint_path=chk_path)

        model_path = joinpath(outpath, "$(model_name)_$(model_num)_FNL.bson")
        model = cpu(model)
        @save model_path model hparams tf_vec

        push!(model_stats, (model_num=model_num, final_test_loss=final_test_loss))
    end

    model_stats = DataFrame(model_stats)
    return model_stats

end


end

# backwards compat for old bson files
ExtendedCascadeModel = NeuralFlowSurrogate