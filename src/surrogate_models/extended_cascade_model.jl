module ExtendedCascadeModel

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
using BSON: @save
using Base.Iterators
using PoissonRandom
using LogExpFunctions
using NonNegLeastSquares
using PhysicsTools
using PhotonPropagation
using ParameterSchedulers
using ParameterSchedulers: Scheduler

using ..RQSplineFlow: eval_transformed_normal_logpdf, sample_flow
using ...Processing

export kfold_train_model
export get_log_amplitudes, unfold_energy_losses, t_first_likelihood
export track_likelihood_fixed_losses, single_cascade_likelihood, multi_particle_likelihood, track_likelihood_energy_unfolding
export sample_cascade_event, evaluate_model, sample_multi_particle_event
export create_pmt_table, preproc_labels, read_pmt_hits, calc_flow_input, fit_trafo_pipeline, log_likelihood_with_poisson
export train_time_expectation_model, train_model!, RQNormFlowHParams
export setup_optimizer, setup_time_expectation_model, setup_dataloaders
export Normalizer


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


function create_mlp_embedding(;
    hidden_structure::AbstractVector{<:Integer},
    n_out,
    dropout=0,
    non_linearity=relu,
    split_final=false)
    model = []
    push!(model, Dense(24 => hidden_structure[1], non_linearity))
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
    n_out,
    non_linearity=relu,
    dropout=0
)

    if !all(hidden_structure[1] .== hidden_structure)
        error("For resnet, all hidden layers have to be of same width")
    end

    layer_width = hidden_structure[1]

    model = []
    push!(model, Dense(24 => layer_width, non_linearity))
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
    params = m.embedding(Float64.(cond))

    spline_params = params[1:end-1, :]
    logpdf_eval = eval_transformed_normal_logpdf(x, spline_params, m.range_min, m.range_max)
    log_expec = params[end, :]
    return logpdf_eval, log_expec

end


"""
    log_likelihood_with_poisson(x::NamedTuple, model::RQNormFlow)

Evaluate model and return sum of logpdfs of normalizing flow and poisson
"""
function log_likelihood_with_poisson(x::NamedTuple, model::ArrivalTimeSurrogate)
    logpdf_eval, log_expec = model(x[:tres], x[:label])

    # poisson: log(exp(-lambda) * lambda^k)
    poiss_f = x[:nhits] .* log_expec .- exp.(log_expec) .- loggamma.(x[:nhits] .+ 1.0)

    # correct for overcounting the poisson factor
    poiss_f = poiss_f ./ x[:nhits]

    return -(sum(logpdf_eval) + sum(poiss_f)) / length(x[:tres])
end


"""
    log_likelihood_with_poisson(x::NamedTuple, model::RQNormFlow)

Evaluate model and return sum of logpdfs of normalizing flow and poisson
"""
function log_likelihood(x::NamedTuple, model::ArrivalTimeSurrogate)
    logpdf_eval = model(x[:tres], x[:label], false)
    return -sum(logpdf_eval) / length(x[:tres])
end


Base.@kwdef struct RQNormFlowHParams
    K::Int64 = 10
    batch_size::Int64 = 5000
    mlp_layers::Int64 = 2
    mlp_layer_size::Int64 = 512
    lr::Float64 = 0.001
    lr_min::Float64 = 1E-5
    epochs::Int64 = 50
    dropout::Float64 = 0.1
    non_linearity::Symbol = :relu
    seed::Int64 = 31338
    l2_norm_alpha = 0.0
    adam_beta_1 = 0.9
    adam_beta_2 = 0.999
    resnet = false
end

function setup_time_expectation_model(hparams::RQNormFlowHParams)
    hidden_structure = fill(hparams.mlp_layer_size, hparams.mlp_layers)

    non_lins = Dict(:relu => relu, :tanh => tanh)
    non_lin = non_lins[hparams.non_linearity]

    # 3 K + 1 for spline, 1 for shift, 1 for scale, 1 for log-expectation
    n_spline_params = 3 * hparams.K + 1
    n_out = n_spline_params + 3

    embedding = create_mlp_embedding(
        hidden_structure=hidden_structure,
        n_out=n_out,
        dropout=hparams.dropout,
        non_linearity=non_lin,
        split_final=false)

    model = NNRQNormFlow(embedding, hparams.K, -20.0, 100.0)
    return model
end

function setup_dataloaders(train_data, test_data, hparams::RQNormFlowHParams)

    rng = Random.MersenneTwister(hparams.seed)
    train_loader = DataLoader(
        train_data,
        batchsize=hparams.batch_size,
        shuffle=true,
        rng=rng)

    test_loader = DataLoader(
        test_data,
        batchsize=50000,
        shuffle=false)

    return train_loader, test_loader
end

function setup_dataloaders(data, hparams::RQNormFlowHParams)
    train_data, test_data = splitobs(data, at=0.7)
    return setup_dataloaders(train_data, test_data, hparams)
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
        opt = Optimiser(WeightDecay(hparams.l2_norm_alpha), opt)
    end
    schedule = Interpolator(CosAnneal(λ0=hparams.lr_min, λ1=hparams.lr, period=hparams.epochs), n_batches)
    return Scheduler(schedule, opt)
end

function train_time_expectation_model(data, use_gpu=true, use_early_stopping=true, checkpoint_path=nothing, model_name="RQNormFlow"; hyperparams...)

    hparams = RQNormFlowHParams(; hyperparams...)
    model = setup_time_expectation_model(hparams)

    logdir = joinpath(@__DIR__, "../../tensorboard_logs/$model_name")
    lg = TBLogger(logdir)

    train_loader, test_loader = setup_dataloaders(data, hparams)

    opt = setup_optimizer(hparams, length(train_loader))

    device = use_gpu ? gpu : cpu
    model, final_test_loss, best_test_loss, best_test_epoch, time_elapsed = train_model!(
        optimizer=opt,
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        loss_function=log_likelihood_with_poisson,
        hparams=hparams,
        logger=lg,
        device=device,
        use_early_stopping=use_early_stopping,
        checkpoint_path=checkpoint_path)

    return model, final_test_loss, best_test_loss, best_test_epoch, hparams, opt, time_elapsed
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

        with_logger(logger) do
            @info "loss" train = total_train_loss test = total_test_loss
            @info "hparams" lr = optimizer.optim.eta log_step_increment = 0
            #@info "model" params = param_dict log_step_increment = 0

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


function create_pmt_table(grp, limit=true)
    hits = DataFrame(grp[:, :], [:tres, :pmt_id])
    for (k, v) in attrs(grp)
        hits[!, k] .= v
    end

    hits = DataFrames.transform!(groupby(hits, :pmt_id), nrow => :hits_per_pmt)

    if limit && (nrow(hits) > 200)
        hits = hits[1:200, :]
    end

    return hits
end

struct Normalizer{T}
    mean::T
    σ::T
end

Normalizer(x::AbstractVector) = Normalizer(mean(x), std(x))
(norm::Normalizer)(x::Number) = (x - norm.mean) / norm.σ

function fit_normalizer!(x::AbstractVector)
    tf = Normalizer(x)
    x .= tf.(x)
    return x, tf
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

function apply_feature_transform(m, tf_vec, n_pmt)

    lev = 1:n_pmt
    one_hot = (lev .== permutedims(m[9, :]))

    tf_matrix = mapreduce(
        t -> permutedims(t[2].(t[1])),
        vcat,
        zip(eachrow(m), tf_vec)
    )

    return vcat(one_hot, tf_matrix)
end


function preproc_labels(df, n_pmt, tf_vec=nothing)

    feature_matrix = dataframe_to_matrix(df)

    if isnothing(tf_vec)
        tf_vec = Vector{Normalizer{Float64}}(undef, 8)
        for (row, ix) in zip(eachrow(feature_matrix), eachindex(tf_vec))
            tf = Normalizer(row)
            tf_vec[ix] = tf
        end
    end

    feature_matrix = apply_feature_transform(feature_matrix, tf_vec, n_pmt)

    return feature_matrix, tf_vec
end

function read_pmt_hits(fnames, nsel_frac=0.8, rng=nothing)

    all_hits = []
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
                hits = create_pmt_table(grp)
                push!(all_hits, hits)
            end
        end
    end

    rnd_ixs = shuffle(rng, 1:length(all_hits))

    all_hits = all_hits[rnd_ixs]

    hits_df = reduce(vcat, all_hits)

    tres = hits_df[:, :tres]
    nhits = hits_df[:, :hits_per_pmt]
    cond_labels, tf_dict = preproc_labels(hits_df, 16)
    return tres, nhits, cond_labels, tf_dict
end

read_hdf(fname::String, nsel, rng) = read_hdf([fname], nsel, rng)


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

    particle_pos = particle.position
    particle_dir = particle.direction
    particle_energy = particle.energy
    target_pos = target.position

    rel_pos = particle_pos .- target_pos
    dist = norm(rel_pos)
    normed_rel_pos = rel_pos ./ dist

    n_pmt = get_pmt_count(target)

    feature_matrix::Matrix{typeof(dist)} = repeat(
        [
            log(dist)
            log(particle_energy)
            particle_dir
            normed_rel_pos
        ],
        1, n_pmt)

    feature_matrix = vcat(feature_matrix, permutedims(1:n_pmt))

    return apply_feature_transform(feature_matrix, tf_vec, n_pmt)

end

"""
calc_flow_input(particles::AbstractVector{<:Particle}, targets::AbstractVector{<:PhotonTarget}, tf_vec)

For vectors of particles and targets, the resulting feature matrices for each combination
are catted together.
The product of `particles` and `targets` is packed densely into a single dimension with
embedding structure: [(p_1, t_1), (p_2, t_1), ... (p_1, t_end), ... (p_end, t_end)]
"""
function calc_flow_input(particles::AbstractVector{<:Particle}, targets::AbstractVector{<:PhotonTarget}, tf_vec)

    res = mapreduce(
        t -> calc_flow_input(t[1], t[2], tf_vec),
        hcat,
        product(particles, targets))

    return res
end


function calc_flow_input(particles::AbstractVector{<:Particle}, targets::AbstractVector{<:PhotonTarget}, tf_vec, output)

    n_pmt = get_pmt_count(eltype(targets))

    for (p_ix, t_ix) in product(eachindex(particles), eachindex(targets))
        particle = particles[p_ix]
        target = targets[t_ix]
        rel_pos = particle.position .- target.position
        dist = norm(rel_pos)
        normed_rel_pos = rel_pos ./ dist

        out_ix = LinearIndices((1:n_pmt, eachindex(particles), eachindex(targets)))

        for pmt_ix in 1:n_pmt

            ix = out_ix[pmt_ix, p_ix, t_ix]

            output[1, ix] = log(dist)
            output[2, ix] = log(particle.energy)
            output[3:5, ix] = particle.direction
            output[6:8, ix] = normed_rel_pos
            output[9, ix] = pmt_ix
        end
    end

    feature_matrix = copy(output)

    return apply_feature_transform(feature_matrix, tf_vec, n_pmt)
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

"""
    sample_multi_particle_event(particles, targets, model, tf_vec, c_n, rng=nothing; oversample=1)

Sample arrival times at `targets` for `particles` using `model`.
"""
function sample_multi_particle_event(particles, targets, model, tf_vec, c_n, rng=nothing; oversample=1)

    n_pmt = get_pmt_count(eltype(targets))
    input = calc_flow_input(particles, targets, tf_vec)
    output = model.embedding(input)

    # The embedding for all the parameters is
    # [(p_1, t_1, pmt_1), (p_1, t_1, pmt_2), ... (p_2, t_1, pmt_1), ... (p_1, t_end, pmt_1), ... (p_end, t_end, pmt_end)]

    flow_params = output[1:end-1, :]
    log_expec_per_source = output[end, :]
    expec_per_source = exp.(log_expec_per_source) .* oversample

    n_hits_per_source = pois_rand.(rng, expec_per_source)
    n_hits_per_source_rs = reshape(
        n_hits_per_source,
        n_pmt,
        length(particles), length(targets))

    mask = n_hits_per_source .> 0
    non_zero_hits = n_hits_per_source[mask]

    # Only sample times for partice-target pairs that have at least one hit
    times = sample_flow(flow_params[:, mask], model.range_min, model.range_max, non_zero_hits, rng=rng)

    # Create range selectors into times for each particle-pmt pair
    selectors = reshape(
        create_non_uniform_ranges(n_hits_per_source),
        n_pmt, length(particles), length(targets)
    )


    data = Vector{Vector{Float64}}(undef, n_pmt * length(targets))
    data_index = LinearIndices((1:n_pmt, eachindex(targets)))

    for (pmt_ix, t_ix) in product(1:n_pmt, eachindex(targets))
        target = targets[t_ix]

        n_hits_this_target = sum(n_hits_per_source_rs[pmt_ix, :, t_ix])

        if n_hits_this_target == 0
            data[data_index[pmt_ix, t_ix]] = []
            continue
        end

        data_this_target = Vector{Float64}(undef, n_hits_this_target)
        data_selectors = create_non_uniform_ranges(n_hits_per_source_rs[pmt_ix, :, t_ix])

        for p_ix in eachindex(particles)
            particle = particles[p_ix]
            this_n_hits = n_hits_per_source_rs[pmt_ix, p_ix, t_ix]
            if this_n_hits > 0
                t_geo = calc_tgeo(norm(particle.position .- target.position) - target.radius, c_n)
                times_sel = selectors[pmt_ix, p_ix, t_ix]
                data_this_target[data_selectors[p_ix]] = times[times_sel] .+ t_geo .+ particle.time
            end
        end
        data[data_index[pmt_ix, t_ix]] = data_this_target
    end
    return data
end



function sample_cascade_event(energy, dir_theta, dir_phi, position, time; targets, model, tf_vec, c_n, rng=nothing)

    dir = sph_to_cart(dir_theta, dir_phi)
    particle = Particle(position, dir, time, energy, 0.0, PEMinus)
    return sample_multi_particle_event([particle], targets, model, tf_vec, c_n, rng)
end

"""
    get_log_amplitudes(particles, targets, model, tf_vec; feat_buffer=nothing)

Evaluate `model` for `particles` and ´targets`

Returns:
    -log_expec_per_pmt: Log of expected photons per pmt. Shape: [n_pmt, 1, n_targets]
    -log_expec_per_src_pmt_rs: Log of expected photons per pmt and per particle. Shape [n_pmt, n_particles, n_targets]
    -flow_params: Flattened input parameters to the normalizing flow
"""
function get_log_amplitudes(particles, targets, model, tf_vec; feat_buffer=nothing)
    n_pmt = get_pmt_count(eltype(targets))

    if isnothing(feat_buffer)
        input = calc_flow_input(particles, targets, tf_vec)
    else
        input = calc_flow_input(particles, targets, tf_vec, feat_buffer)
    end
    output::Matrix{eltype(input)} = cpu(model.embedding(gpu(input)))

    flow_params = output[1:end-1, :]
    log_expec_per_src_trg = output[end, :]

    log_expec_per_src_pmt_rs = reshape(
        log_expec_per_src_trg,
        n_pmt, length(particles), length(targets))

    log_expec_per_pmt = LogExpFunctions.logsumexp(log_expec_per_src_pmt_rs, dims=2)

    return log_expec_per_pmt, log_expec_per_src_pmt_rs, flow_params
end


function shape_llh_generator(data; particles, targets, flow_params, rel_log_expec, model, c_n)

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
                    data[data_ix[pmt_ix, t_ix]] .- calc_tgeo(particles[p_ix], targets[t_ix], c_n) .- particles[p_ix].time,
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


function evaluate_model(particles, data, targets, model, tf_vec, c_n; feat_buffer=nothing)

    log_expec_per_pmt, log_expec_per_src_pmt_rs, flow_params = get_log_amplitudes(particles, targets, model, tf_vec; feat_buffer=feat_buffer)

    rel_log_expec = log_expec_per_src_pmt_rs .- log_expec_per_pmt

    hits_per_target = length.(data)
    # Flattening log_expec_per_pmt with [:] will let the first dimension be the inner one
    poiss_llh = poisson_logpmf.(hits_per_target, log_expec_per_pmt[:])

    shape_llh_gen = shape_llh_generator(data; particles=particles, targets=targets, flow_params=flow_params, rel_log_expec=rel_log_expec, model=model, c_n=c_n)
    return poiss_llh, shape_llh_gen, log_expec_per_pmt
end


function multi_particle_likelihood(particles; data, targets, model, tf_vec, c_n, feat_buffer=nothing, amp_only=false)
    n_pmt = get_pmt_count(eltype(targets))
    @assert length(targets) * n_pmt == length(data)
    pois_llh, shape_llh, _ = evaluate_model(particles, data, targets, model, tf_vec, c_n, feat_buffer=feat_buffer)
    if amp_only
        return sum(pois_llh)
    else
        return sum(pois_llh) + sum(shape_llh)
    end
end


function single_cascade_likelihood(logenergy, dir_theta, dir_phi, position, time; data, targets, model, tf_vec, c_n, feat_buffer=nothing)
    dir = sph_to_cart(dir_theta, dir_phi)
    energy = 10^logenergy
    particles = [Particle(position, dir, time, energy, 0.0, PEMinus)]
    return multi_particle_likelihood(particles, data=data, targets=targets, model=model, tf_vec=tf_vec, c_n=c_n, feat_buffer=feat_buffer)
end

function track_likelihood_fixed_losses(
    logenergy, dir_theta, dir_phi, position;
    losses, muon_energy, data, targets, model, tf_vec, c_n, feat_buffer=nothing, amp_only=false)

    energy = 10^logenergy
    dir = sph_to_cart(dir_theta, dir_phi)
    dist_along = norm.([p.position .- position for p in losses])


    new_loss_positions = [position .+ d .* dir for d in dist_along]
    new_loss_energies = [p.energy / muon_energy * energy for p in losses]

    times = [p.time for p in losses]

    new_losses = Particle.(new_loss_positions, [dir], times, new_loss_energies, 0.0, [PEMinus])

    return multi_particle_likelihood(
        new_losses, data=data, targets=targets, model=model, tf_vec=tf_vec, c_n=c_n, feat_buffer=feat_buffer, amp_only=amp_only)
end


function t_first_likelihood(particles; data, targets, model, tf_vec, c_n, feat_buffer=nothing)

    log_expec_per_pmt, log_expec_per_src_pmt_rs, flow_params = get_log_amplitudes(particles, targets, model, tf_vec; feat_buffer=feat_buffer)
    rel_log_expec = log_expec_per_src_pmt_rs .- log_expec_per_pmt

    # data masking
    t_first = [length(d) > 0 ? minimum(d) : -inf]


    llh_tfirst = shape_llh_generator(t_first; particles=particles, targets=targets, flow_params=flow_params, rel_log_expec=rel_log_expec, model=model, c_n=c_n)

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


function track_likelihood_energy_unfolding(dir_theta, dir_phi, position, time; spacing, data, targets, model, tf_vec, c_n, amp_only=false)

    dir = sph_to_cart(dir_theta, dir_phi)
    losses = unfold_energy_losses(position, dir, time; data=data, targets=targets, model=model, tf_vec=tf_vec, spacing=spacing)
    if length(losses) == 0
        return -Inf64
    end
    return multi_particle_likelihood(losses, data=data, targets=targets, model=model, tf_vec=tf_vec, c_n=c_n, amp_only=amp_only)
end

function kfold_train_model(data, model_name, tf_vec, k=5; hyperparams...)
    hparams = RQNormFlowHParams(; hyperparams...)

    logdir = joinpath(@__DIR__, "../../tensorboard_logs/$model_name")

    for (model_num, (train_data, val_data)) in enumerate(kfolds(data; k=k))
        lg = TBLogger(logdir)
        model = setup_time_expectation_model(hparams)
        chk_path = joinpath(@__DIR__, "../data/$(model_name)_$(model_num)")

        train_loader, test_loader = setup_dataloaders(train_data, val_data, hparams)
        opt = setup_optimizer(hparams, length(train_loader))
        device = gpu
        model, final_test_loss, best_test_loss, best_test_epoch, time_elapsed = train_model!(
            optimizer=opt,
            train_loader=train_loader,
            test_loader=test_loader,
            model=model,
            loss_function=log_likelihood_with_poisson,
            hparams=hparams,
            logger=lg,
            device=device,
            use_early_stopping=true,
            checkpoint_path=chk_path)

        model_path = joinpath(@__DIR__, "../data/$(model_name)_$(model_num)_FNL.bson")
        model = cpu(model)
        @save model_path model hparams tf_vec
    end
end


end
