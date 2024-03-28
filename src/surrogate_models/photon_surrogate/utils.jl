export Normalizer, fit_normalizer!
export create_mlp_embedding, create_resnet_embedding
export arrival_time_log_likelihood, log_likelihood_with_poisson
export create_model_input!
export apply_normalizer
export kfold_train_model
export sample_multi_particle_event
export get_log_amplitudes
export evaluate_model
export multi_particle_likelihood

using ArraysOfArrays
using Flux
using Random
using ProgressLogging
using PhysicsTools
using PhotonPropagation
using MLUtils
using ParameterSchedulers
using ParameterSchedulers: Scheduler, Stateful, next!
using EarlyStopping
using Logging
using ProgressLogging
using SpecialFunctions
import Base.GC: gc
using Base.Iterators
using TensorBoardLogger
using LinearAlgebra
using LogExpFunctions
using PoissonRandom
using ..RQSplineFlow


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

function apply_normalizer(m, tf_vec)
    # Not mutating version...
    tf_matrix = mapreduce(
        t -> permutedims(t[2].(t[1])),
        vcat,
        zip(eachrow(m), tf_vec)
    )
    return tf_matrix
end

function apply_normalizer(m, tf_vec, output)
    # Mutating version...
    for (in_row, out_row, tf) in zip(eachrow(m), eachrow(output), tf_vec)
        out_row .= tf.(in_row)
    end
    return output
end

function fit_normalizer!(x::AbstractMatrix)
    tf_vec = Vector{Normalizer{Float64}}(undef, size(x, 1))
    for (row, ix) in zip(eachrow(x), eachindex(tf_vec))
        row, tf = fit_normalizer!(row)
        tf_vec[ix] = tf
    end

    return x, tf_vec
end


export ArrivalTimeSurrogate, RQSplineModel
export PhotonSurrogate, PhotonSurrogateWithPerturb, PhotonSurrogateWithoutPerturb
export HyperParams
export setup_model

abstract type SurrogateModel end
abstract type ArrivalTimeSurrogate <: SurrogateModel end
abstract type RQSplineModel <: ArrivalTimeSurrogate end
abstract type AmplitudeSurrogate <: SurrogateModel end


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
- `amp_model::AmplitudeSurrogate`: The amplitude model of the surrogate.
- `amp_transformations::Vector{Normalizer}`: The amplitude transformations applied to the surrogate.
- `time_model::ArrivalTimeSurrogate`: The time model of the surrogate.
- `time_transformations::Vector{Normalizer}`: The time transformations applied to the surrogate.
"""
struct PhotonSurrogateWithPerturb{A<:AmplitudeSurrogate, T<:ArrivalTimeSurrogate} <: PhotonSurrogate
    amp_model::A
    time_model::T
end

"""
    struct PhotonSurrogateWithoutPerturb <: PhotonSurrogate

The `PhotonSurrogateWithoutPerturb` struct represents a photon surrogate model without medium perturbations.

# Fields
- `amp_model::AmplitudeSurrogate`: The amplitude model for the surrogate.
- `amp_transformations::Vector{Normalizer}`: The amplitude transformations applied to the surrogate.
- `time_model::ArrivalTimeSurrogate`: The time model for the surrogate.
- `time_transformations::Vector{Normalizer}`: The time transformations applied to the surrogate.
"""
struct PhotonSurrogateWithoutPerturb{A<:AmplitudeSurrogate, T<:ArrivalTimeSurrogate} <: PhotonSurrogate
    amp_model::A
    time_model::T
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

    inp_size_time = size(time_model.embedding.layers[1].weight, 2) 
    inp_size_amp = size(amp_model.embedding.layers[1].weight, 2) 

    if inp_size_time == 26 && inp_size_amp == 10
        mtype = PhotonSurrogateWithPerturb
    elseif inp_size_time == 24 && inp_size_amp == 8
        mtype = PhotonSurrogateWithoutPerturb
    else
        error("Cannot parse model inputs.")
    end

    return mtype(b1[:model], time_model)
end

Flux.gpu(s::T) where {T <: PhotonSurrogate} = T(gpu(s.amp_model), gpu(s.time_model))
Flux.cpu(s::T) where {T <: PhotonSurrogate} = T(cpu(s.amp_model), cpu(s.time_model))

abstract type HyperParams end

StructTypes.StructType(::Type{<:HyperParams}) = StructTypes.Struct()

setup_model(hparams::HyperParams) = error("Not implemented for type $(typeof(hparams))")


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


"""
    log_poisson_likelihood(x::NamedTuple, model::AmplitudeSurrogate)

Compute the log-likelihood of a Poisson distribution given the observed number of hits and the expected number of hits.

# Arguments
- `x::NamedTuple`: A named tuple containing the labels (input data) and observed number of hits.
- `model::ArrivalTimeSurrogate`: The surrogate model used to predict the expected number of hits.

# Returns
- `Float64`: The negative log-likelihood of the Poisson distribution.

"""
function log_poisson_likelihood(x::NamedTuple, model::AmplitudeSurrogate)
    
    # one expectation per PMT (16 x batch_size)
    log_expec = model(x[:labels])
    poiss_f = x[:nhits] .* log_expec .- exp.(log_expec) .- loggamma.(x[:nhits] .+ 1.0)

    return -sum(poiss_f) / size(x[:labels], 2)
end


"""
    setup_dataloaders(train_data, test_data, seed::Integer, batch_size::Integer)

Create and return data loaders for training and testing.

# Arguments
- `train_data`: The training data.
- `test_data`: The testing data.
- `seed`: The seed for random number generation.
- `batch_size`: The batch size for the data loaders.

# Returns
- `train_loader`: The data loader for training.
- `test_loader`: The data loader for testing.
"""
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


"""
    setup_dataloaders(train_data, test_data, hparams::HyperParams)

# Arguments
- `train_data`: The training data.
- `test_data`: The testing data.
- `hparams`: The hyperparameters.

"""
function setup_dataloaders(train_data, test_data, hparams::HyperParams)
    return setup_dataloaders(train_data, test_data, hparams.seed, hparams.batch_size)   
end

function setup_dataloaders(data, args...)
    train_data, test_data = splitobs(data, at=0.7, shuffle=true)
    return setup_dataloaders(train_data, test_data, args...)
end

"""
    fill_param_dict!(dict, m, prefix)

Fill a dictionary with the parameters of a given model `m` by recursively traversing its fields.
The dictionary `dict` is updated with the parameter names as keys and their corresponding values as values.
The `prefix` argument is used to specify a prefix for the parameter names in the dictionary.

# Arguments
- `dict::Dict{String, Any}`: The dictionary to be filled with parameter names and values.
- `m`: The model whose parameters will be added to the dictionary.
- `prefix::String`: The prefix to be added to the parameter names in the dictionary.

"""
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

function setup_optimizer(hparams)
    opt = Adam(hparams.lr, (hparams.adam_beta_1, hparams.adam_beta_2))
    if hparams.l2_norm_alpha > 0
        opt = OptimiserChain(WeightDecay(hparams.l2_norm_alpha), opt)
    end

    return opt
end

"""
    train_model!(;
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

Train a model using the specified optimizer, data loaders, model architecture, loss function, hyperparameters, logger, device, and early stopping criteria.

# Arguments
- `optimizer`: The optimizer used for training the model.
- `train_loader`: The data loader for the training dataset.
- `test_loader`: The data loader for the testing dataset.
- `model`: The model architecture to be trained.
- `loss_function`: The loss function used for training.
- `hparams`: The hyperparameters for training the model.
- `logger`: The logger for recording training progress.
- `device`: The device (e.g., CPU or GPU) on which the model will be trained.
- `use_early_stopping`: A boolean indicating whether to use early stopping during training.
- `checkpoint_path`: The path to save the best model checkpoint.

# Returns
- `model`: The trained model.
- `total_test_loss`: The total test loss after training.
- `best_test`: The best test loss achieved during training.
- `best_test_epoch`: The epoch at which the best test loss was achieved.
- `training_time`: The total training time in seconds.
"""

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
    checkpoint_path=nothing,
    schedule=nothing)

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
    @show length(train_loader)
    @progress for epoch in 1:hparams.epochs
        Flux.trainmode!(model)

        total_train_loss = 0.0
        for d in train_loader
            d = d |> device
            loss, grads = Flux.withgradient(model) do m
                loss = loss_function(d, m)
                return loss
            end
            total_train_loss += loss
            Flux.update!(optimizer, model, grads[1])
        end

        total_train_loss /= length(train_loader)

        if !isnothing(schedule)
            eta = next!(schedule)
            Flux.adjust!(optimizer, eta)
        end

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

function create_model_input!(
    particle_pos,
    particle_dir,
    particle_energy,
    target_pos,
    output,
    tf_vec;
    abs_scale,
    sca_scale
)
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

        if length(tf_vec) == 10
            output[9] = tf_vec[9](abs_scale)
            output[10] = tf_vec[10](sca_scale)
        end
    end

    return output
end

function create_model_input!(
    particle::Particle,
    target::PhotonTarget,
    output,
    tf_vec;
    abs_scale,
    sca_scale)

    if particle_shape(particle) == Track()
        particle = shift_to_closest_approach(particle, target.shape.position)
    end

    return create_model_input!(particle.position, particle.direction, particle.energy, target.shape.position, output, tf_vec, abs_scale=abs_scale, sca_scale=sca_scale)
end

function create_model_input!(
    particles::AbstractVector{<:Particle},
    targets::AbstractVector{<:PhotonTarget},
    output,
    tf_vec;
    abs_scale,
    sca_scale)
    out_ix = LinearIndices((eachindex(particles), eachindex(targets)))

    for (p_ix, t_ix) in product(eachindex(particles), eachindex(targets))
        particle = particles[p_ix]
        target = targets[t_ix]

        ix = out_ix[p_ix, t_ix]

        outview = @view output[:, ix]

        create_model_input!(particle, target, outview, tf_vec, abs_scale=abs_scale, sca_scale=sca_scale)
        
    end
    return output
end


function kfold_train_model(data, outpath, model_name, tf_vec, n_folds, hparams::HyperParams, logdir)
       

    model_stats = []

    for (model_num, (train_data, val_data)) in enumerate(kfolds(shuffleobs(data); k=n_folds))
        gc()
        lg = TBLogger(logdir)
        model, loss_f = setup_model(hparams, tf_vec)
        model = gpu(model)
        chk_path = joinpath(outpath, "$(model_name)_$(model_num)")

        train_loader, test_loader = setup_dataloaders(train_data, val_data, hparams)
        opt = setup_optimizer(hparams)
        schedule = Stateful(CosAnneal(λ0=hparams.lr_min, λ1=hparams.lr, period=hparams.epochs))

        opt_state = Flux.setup(opt, model)

        device = gpu
        model, final_test_loss, best_test_loss, best_test_epoch, time_elapsed = train_model!(
            optimizer=opt_state,
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            loss_function=loss_f,
            hparams=hparams,
            logger=lg,
            device=device,
            use_early_stopping=false,
            checkpoint_path=chk_path,
            schedule=schedule)

        model_path = joinpath(outpath, "$(model_name)_$(model_num)_FNL.bson")
        model = cpu(model)
        @save model_path model hparams tf_vec

        push!(model_stats, (model_num=model_num, final_test_loss=final_test_loss))
    end

    model_stats = DataFrame(model_stats)
    return model_stats

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
    device=gpu,
    abs_scale=1.,
    sca_scale=1.,
    noise_rate=1E4,
    time_window=1E4)

    # We currently cannot reshape VectorOfArrays. For now just double allocate
    temp_output_buffer = VectorOfArrays{Float64, 1}()
    n_pmts = get_pmt_count(eltype(targets))*length(targets)
    sizehint!(temp_output_buffer, n_pmts, (100, ))

    # Calculate flow parameters
    n_pmt = get_pmt_count(eltype(targets))
    _, log_expec_per_src_pmt_rs = get_log_amplitudes(particles, targets, model; feat_buffer=feat_buffer, device=device, abs_scale=abs_scale, sca_scale=sca_scale)

    shape_buffer = @view feat_buffer[:, 1:length(particles)*length(targets)*n_pmt]
    create_model_input!(model.time_model, particles, targets, shape_buffer, abs_scale=abs_scale, sca_scale=sca_scale)
    input = shape_buffer

    # The embedding for all the parameters is
    # [(p_1, t_1, pmt_1), (p_1, t_1, pmt_2), ... (p_2, t_1, pmt_1), ... (p_1, t_end, pmt_1), ... (p_end, t_end, pmt_end)]

    flow_params = cpu(model.time_model.embedding(device(input)))
    expec_per_source_rs = log_expec_per_src_pmt_rs .= exp.(log_expec_per_src_pmt_rs) .* oversample
    n_hits_per_pmt_source = pois_rand.(rng, Float64.(expec_per_source_rs))

    n_hits_per_source = vec(n_hits_per_pmt_source)
    mask = n_hits_per_source .> 0
    non_zero_hits = n_hits_per_source[mask]

    # Only sample times for particle-target pairs that have at least one hit
    times = sample_flow!(flow_params[:, mask], model.time_model.range_min, model.time_model.range_max, non_zero_hits, temp_output_buffer, rng=rng)


    if isnothing(output_buffer)
        output_buffer = VectorOfArrays{Float64, 1}()
        sizehint!(output_buffer, n_pmts, (100, ))
    end

    empty!(output_buffer)

    
    # Create range selectors into times for each particle-pmt pair
    selectors = reshape(
        create_non_uniform_ranges(n_hits_per_source),
        n_pmt, length(particles), length(targets)
    )
    times = flatview(times)

    noise_rate *= 1E-9

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

        n_noise_hits = pois_rand(noise_rate * time_window)
        # println("Adding $(n_noise_hits)")
        if n_noise_hits > 0
            noise_times = rand(n_noise_hits) .*time_window .- time_window/2
            data_this_target = vcat(data_this_target, noise_times)
        end

        push!(output_buffer, data_this_target)
    end
    return output_buffer

end

"""
get_log_amplitudes(particles, targets, model::PhotonSurrogate; feat_buffer=nothing)

Evaluate `model` for `particles` and `targets`

Returns:
    -log_expec_per_pmt: Log of expected photons per pmt. Shape: [n_pmt, 1, n_targets]
    -log_expec_per_src_pmt_rs: Log of expected photons per pmt and per particle. Shape [n_pmt, n_particles, n_targets]
"""
function get_log_amplitudes(particles, targets, model::PhotonSurrogate; feat_buffer, device=gpu, abs_scale, sca_scale)

    n_pmt = get_pmt_count(eltype(targets))

    #TODO: get rid of this
    input_size = size(model.amp_model.embedding.layers[1].weight, 2)

    amp_buffer = @view feat_buffer[1:input_size, 1:length(targets)*length(particles)]
    create_model_input!(model.amp_model, particles, targets, amp_buffer, abs_scale=abs_scale, sca_scale=sca_scale)

    input = amp_buffer

    input = permutedims(input)'

    log_expec_per_src_trg::Matrix{eltype(input)} = cpu(model.amp_model(device(input)))

    log_expec_per_src_pmt_rs = reshape(
        log_expec_per_src_trg,
        n_pmt, length(particles), length(targets))

    log_expec_per_pmt = LogExpFunctions.logsumexp(log_expec_per_src_pmt_rs, dims=2)

    return log_expec_per_pmt, log_expec_per_src_pmt_rs
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

function evaluate_model(
    particles::AbstractVector{<:Particle};
    data,
    targets,
    model::PhotonSurrogate,
    medium,
    feat_buffer=nothing,
    device=gpu,
    noise_rate=1E4,
    time_window=1E4,
    abs_scale,
    sca_scale)


    log_expec_per_pmt, log_expec_per_src_pmt_rs = get_log_amplitudes(particles, targets, model; feat_buffer=feat_buffer, device=device, abs_scale=abs_scale, sca_scale=sca_scale)
    

    hits_per_target = length.(data)
    # Flattening log_expec_per_pmt with [:] will let the first dimension be the inner one
    poiss_llh = poisson_logpmf.(hits_per_target, vec(log_expec_per_pmt[:]))

    npmt = get_pmt_count(eltype(targets))


    input = @view feat_buffer[:, 1:length(targets)*length(particles)*npmt]
    create_model_input!(model.time_model, particles, targets, input, abs_scale=abs_scale, sca_scale=sca_scale)


    flow_params::Matrix{eltype(input)} = cpu(model.time_model.embedding(device(input)))

    #sllh = shape_llh(data; particles=particles, targets=targets, flow_params=flow_params, rel_log_expec=rel_log_expec, model=model.time_model, medium=medium)
    sllh = shape_llh_generator(
        particles,
        data=data,
        targets=targets,
        flow_params=flow_params,
        log_expec_per_pmt=log_expec_per_pmt,
        log_expec_per_src_pmt=log_expec_per_src_pmt_rs,
        model=model.time_model,
        medium=medium,
        time_window=time_window,
        noise_rate=noise_rate)
        
    return poiss_llh, sllh, log_expec_per_pmt
end

"""
    shape_llh_generator(particles::AbstractVector{<:Particle};
                       data::AbstractVector{<:AbstractVector{<:Real}},
                       targets::AbstractVector{<:PhotonTarget},
                       flow_params,
                       log_expec_per_pmt::AbstractArray{<:Real, 3},
                       log_expec_per_src_pmt::AbstractArray{<:Real, 3},
                       model,
                       medium,
                       noise_rate=1E4)

This function generates the shape likelihood for a given set of particles. It calculates the shape likelihood by evaluating the noise and signal likelihoods for each PMT and target combination,
and then combining them using log-sum-exp. The shape likelihood is returned as a generator.

# Arguments
- `particles::AbstractVector{<:Particle}`: A vector of particles.
- `data::AbstractVector{<:AbstractVector{<:Real}}`: A vector of data vectors.
- `targets::AbstractVector{<:PhotonTarget}`: A vector of photon targets.
- `flow_params`: The flow parameters.
- `log_expec_per_pmt::AbstractArray{<:Real, 3}`: An array of log expectations per PMT.
- `log_expec_per_src_pmt::AbstractArray{<:Real, 3}`: An array of log expectations per source PMT.
- `model`: The model.
- `medium`: The medium.
- `noise_rate=1E4`: The noise rate.

# Returns
- `shape_llh_gen`: The shape likelihood generator.
"""
function shape_llh_generator(
    particles::AbstractVector{<:Particle};
    data::AbstractVector{<:AbstractVector{<:Real}},
    targets::AbstractVector{<:PhotonTarget},
    flow_params,
    log_expec_per_pmt::AbstractArray{<:Real, 3},
    log_expec_per_src_pmt::AbstractArray{<:Real, 3},
    model,
    medium,
    time_window,
    noise_rate)

    rel_log_expec = log_expec_per_src_pmt .= log_expec_per_src_pmt .- log_expec_per_pmt

    n_pmt = get_pmt_count(eltype(targets))
    data_ix = LinearIndices((1:n_pmt, eachindex(targets)))
    ix = LinearIndices((1:n_pmt, eachindex(particles), eachindex(targets)))

    noise_rate = noise_rate * 1E-9

    if noise_rate == 0
        noise_rate = 1E-300
    end

    noise_expec_per_pmt = time_window * noise_rate 
    rel_noise_expec = noise_expec_per_pmt ./ (noise_expec_per_pmt .+ exp.(log_expec_per_pmt))
    
    # Uniform: l = 1/(b-a)
    noise_llh = -log(time_window)

    shape_llh_gen = (
        length(data[data_ix[pmt_ix, t_ix]]) > 0 ?
        sum(
            LogExpFunctions.logaddexp.(
                # Noise likelihood
                Ref(log.(rel_noise_expec[data_ix[pmt_ix, t_ix]]) + noise_llh),

                # Signal Likelihood
                # Reduce over the particle dimension to create the mixture
                log.(1 .-rel_noise_expec[data_ix[pmt_ix, t_ix]]) .+ 

                LogExpFunctions.logsumexp(
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
                )
            )
        )
        : 0.0
        for (pmt_ix, t_ix) in product(1:n_pmt, eachindex(targets)))
    return shape_llh_gen
end

#=
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
=#

"""
    multi_particle_likelihood(particles::AbstractVector{<:Particle};
                             data::AbstractVector{<:AbstractVector{<:Real}},
                             targets::AbstractVector{<:PhotonTarget},
                             model::PhotonSurrogate,
                             medium::MediumProperties;
                             feat_buffer=nothing,
                             amp_only=false,
                             device=gpu,
                             noise_rate=1E4)

Compute the likelihood of multiple particles given the data and model.

# Arguments
- `particles`: An abstract vector of particles.
- `data`: An abstract vector of abstract vectors of real numbers representing the data.
- `targets`: An abstract vector of photon targets.
- `model`: A photon surrogate model.
- `medium`: Medium properties.
- `feat_buffer`: (optional) A buffer for storing intermediate feature calculations.
- `amp_only`: (optional) A boolean indicating whether to compute only the amplitude likelihood.
- `device`: (optional) The device to use for computation.
- `noise_rate`: (optional) The noise rate (Hz)
- `time_window`: (optional) The timewindow length (ns)

# Returns
- The likelihood of the particles given the data and model.

"""
function multi_particle_likelihood(
    particles::AbstractVector{<:Particle};
    data::AbstractVector{<:AbstractVector{<:Real}},
    targets::AbstractVector{<:PhotonTarget},
    model::PhotonSurrogate,
    medium::MediumProperties,
    feat_buffer=nothing,
    amp_only=false,
    device=gpu,
    noise_rate=1E4,
    time_window=1E4,
    abs_scale=1.,
    sca_scale=1.
)

    n_pmt = get_pmt_count(eltype(targets))
    @assert length(targets) * n_pmt == length(data)
    pois_llh, shape_llh, _ = evaluate_model(
        particles,
        data=data,
        targets=targets,
        model=model,
        medium=medium,
        feat_buffer=feat_buffer,
        device=device,
        noise_rate=noise_rate,
        time_window=time_window,
        abs_scale=abs_scale,
        sca_scale=sca_scale,
)
    if amp_only
        return sum(pois_llh)
    else
        return sum(pois_llh) + sum(shape_llh, init=0.)
    end
end

function poisson_logpmf(n, log_lambda)
    return n * log_lambda - exp(log_lambda) - loggamma(n + 1.0)
end

