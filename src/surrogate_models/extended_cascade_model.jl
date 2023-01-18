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

using ..RQSplineFlow: eval_transformed_normal_logpdf, sample_flow
using ...Types
using ...Utils
using ...PhotonPropagation.Detection
using ...PhotonPropagation.PhotonPropagationCuda

export track_likelihood_fixed_losses, single_cascade_likelihood, multi_particle_likelihood
export sample_cascade_event, evaluate_model, sample_multi_particle_event
export create_pmt_table, preproc_labels, read_pmt_hits, calc_flow_input, fit_trafo_pipeline, log_likelihood_with_poisson
export train_time_expectation_model, train_model!, RQNormFlowHParams, setup_time_expectation_model, setup_dataloaders
export Normalizer


abstract type ArrivalTimeSurrogate end

"""
    RQNormFlow(K::Integer,
                      range_min::Number,
                      range_max::Number,
                      hidden_structure::AbstractVector{<:Integer};
                      dropout::Real=0.3,
                      non_linearity=relu)

1-D rq-spline normalizing flow with expected counts prediction.

The rq-spline requires 3 * K + 1 parameters, where `K` is the number of knots. These are
parametrized by an embedding (MLP).

# Arguments
- K: Number of knots
- range_min:: Lower bound of the spline transformation
- range_max:: Upper bound of the spline transformation
- hidden_structure:  Number of nodes per MLP layer
- dropout: Dropout value (between 0 and 1) used in training (default=0.3)
- non_linearity: Non-linearity used in MLP (default=relu)
- add_log_expec: Also predict log-expectation
- split_final=false: Split the final layer into one for predicting the spline params and one for the log_expec
"""
struct RQNormFlow <: ArrivalTimeSurrogate
    embedding::Chain
    K::Integer
    range_min::Float64
    range_max::Float64
    has_log_expec::Bool
end

# Make embedding parameters trainable
Flux.@functor RQNormFlow (embedding,)

function RQNormFlow(K::Integer,
    range_min::Number,
    range_max::Number,
    hidden_structure::AbstractVector{<:Integer};
    dropout=0.3,
    non_linearity=relu,
    add_log_expec=false,
    split_final=false
)

    model = []
    push!(model, Dense(24 => hidden_structure[1], non_linearity))
    push!(model, Dropout(dropout))
    for ix in 2:length(hidden_structure[2:end])
        push!(model, Dense(hidden_structure[ix-1] => hidden_structure[ix], non_linearity))
        push!(model, Dropout(dropout))
    end

    # 3 K + 1 for spline, 1 for shift, 1 for scale, 1 for log-expectation
    n_spline_params = 3 * K + 1
    n_flow_params = n_spline_params + 2


    if add_log_expec && split_final
        final = Parallel(vcat,
            Dense(hidden_structure[end] => n_flow_params),
            Dense(hidden_structure[end] => 1)
        )
    elseif add_log_expec && !split_final
        #zero_init(out, in) = vcat(zeros(out-3, in), zeros(1, in), ones(1, in), fill(1/in, 1, in))
        final = Dense(hidden_structure[end] => n_flow_params + 1)
    else
        final = Dense(hidden_structure[end] => n_flow_params)
    end
    push!(model, final)

    return RQNormFlow(Chain(model...), K, range_min, range_max, add_log_expec)
end

"""
    (m::RQNormFlow)(x, cond)

Evaluate normalizing flow at values `x` with conditional values `cond`.

Returns logpdf and log-expectation
"""
function (m::RQNormFlow)(x, cond, pred_log_expec=false)
    params = m.embedding(Float64.(cond))

    @assert !pred_log_expec || (pred_log_expec && m.has_log_expec) "Requested to return log expectation, but model doesn't provide.
    "
    if pred_log_expec
        spline_params = params[1:end-1, :]
        logpdf_eval = eval_transformed_normal_logpdf(x, spline_params, m.range_min, m.range_max)
        log_expec = params[end, :]

        return logpdf_eval, log_expec
    else
        logpdf_eval = eval_transformed_normal_logpdf(x, params, m.range_min, m.range_max)
        return logpdf_eval
    end
end


"""
    log_likelihood_with_poisson(x::NamedTuple, model::RQNormFlow)

Evaluate model and return sum of logpdfs of normalizing flow and poisson
"""
function log_likelihood_with_poisson(x::NamedTuple, model::RQNormFlow)
    logpdf_eval, log_expec = model(x[:tres], x[:label], true)

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
function log_likelihood(x::NamedTuple, model::RQNormFlow)
    logpdf_eval = model(x[:tres], x[:label], false)
    return -sum(logpdf_eval) / length(x[:tres])
end


Base.@kwdef struct RQNormFlowHParams
    K::Int64 = 10
    batch_size::Int64 = 5000
    mlp_layers::Int64 = 2
    mlp_layer_size::Int64 = 512
    lr::Float64 = 0.001
    epochs::Int64 = 50
    dropout::Float64 = 0.1
    non_linearity::Symbol = :relu
    seed::Int64 = 31338
    l2_norm_alpha = 0.0
end

function setup_time_expectation_model(hparams::RQNormFlowHParams)
    hidden_structure = fill(hparams.mlp_layer_size, hparams.mlp_layers)

    non_lins = Dict(:relu => relu, :tanh => tanh)
    non_lin = non_lins[hparams.non_linearity]

    model = RQNormFlow(
        hparams.K, -20.0, 100.0, hidden_structure, dropout=hparams.dropout, non_linearity=non_lin,
        add_log_expec=true
    )
    return model
end

function setup_dataloaders(data, hparams::RQNormFlowHParams)
    train_data, test_data = splitobs(data, at=0.7)
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


function train_time_expectation_model(data, use_gpu=true, use_early_stopping=true, checkpoint_path=nothing; hyperparams...)

    hparams = RQNormFlowHParams(; hyperparams...)

    model = setup_time_expectation_model(hparams)

    if hparams.l2_norm_alpha > 0
        opt = Optimiser(WeightDecay(hparams.l2_norm_alpha), Adam(hparams.lr))
    else
        opt = Adam(hparams.lr)
    end

    logdir = joinpath(@__DIR__, "../../tensorboard_logs/RQNormFlow")
    lg = TBLogger(logdir)

    train_loader, test_loader = setup_dataloaders(data, hparams)

    device = use_gpu ? gpu : cpu
    model, final_test_loss = train_model!(opt, train_loader, test_loader, model, log_likelihood_with_poisson, hparams, lg, device, use_early_stopping, checkpoint_path)

    return model, final_test_loss, hparams, opt
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

function train_model!(opt, train, test, model, loss_function, hparams, logger, device, use_early_stopping, checkpoint_path=nothing)
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

    @progress for epoch in 1:hparams.epochs

        Flux.trainmode!(model)

        total_train_loss = 0.0
        for d in train
            d = d |> device
            gs = gradient(pars) do
                loss = loss_function(d, model)

                return loss
            end
            total_train_loss += loss
            Flux.update!(opt, pars, gs)
        end


        total_train_loss /= length(train)

        Flux.testmode!(model)
        total_test_loss = 0
        for d in test
            d = d |> device
            total_test_loss += loss_function(d, model)
        end
        total_test_loss /= length(test)

        param_dict = Dict{String,Any}()
        fill_param_dict!(param_dict, model, "")


        with_logger(logger) do
            @info "loss" train = total_train_loss test = total_test_loss
            @info "model" params = param_dict log_step_increment = 0

        end
        println("Epoch: $epoch, Train: $total_train_loss Test: $total_test_loss")

        if !isnothing(checkpoint_path) && epoch > 5 && total_test_loss < best_test
            @save checkpoint_path * "_BEST.bson" model
            best_test = total_test_loss
        end

        done!(stopper, total_test_loss) && break

    end
    return model, total_test_loss
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

    n_hits_per_source = pois_rand.(expec_per_source)
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
    particle = Particle(position, dir, time, energy, 0., PEMinus)
    return sample_multi_particle_event([particle], targets, model, tf_vec, c_n, rng)
end


function evaluate_model(particles, data, targets, model, tf_vec, c_n; feat_buffer=nothing)
    n_pmt = get_pmt_count(eltype(targets))
    @assert length(targets) * n_pmt == length(data)

    if isnothing(feat_buffer)
        input = calc_flow_input(particles, targets, tf_vec)
    else
        input = calc_flow_input(particles, targets, tf_vec, feat_buffer)
    end


    output::Matrix{eltype(input)} = cpu(model.embedding(gpu(input)))

    # The embedding for all the parameters is
    # [(p_1, t_1, pmt_1), (p_1, t_1, pmt_2), ... (p_2, t_1, pmt_1), ... (p_1, t_end, pmt_1), ... (p_end, t_end, pmt_end)]
    flow_params = output[1:end-1, :]
    log_expec_per_src_trg = output[end, :]

    log_expec_per_src_pmt_rs = reshape(
        log_expec_per_src_trg,
        n_pmt, length(particles), length(targets))

    log_expec_per_pmt = LogExpFunctions.logsumexp(log_expec_per_src_pmt_rs, dims=2)
    rel_log_expec = log_expec_per_src_pmt_rs .- log_expec_per_pmt

    hits_per_target = length.(data)
    # Flattening log_expec_per_pmt with [:] will let the first dimension be the inner one
    poiss_llh = poisson_logpmf.(hits_per_target, log_expec_per_pmt[:])

    data_ix = LinearIndices((1:n_pmt, eachindex(targets)))
    ix = LinearIndices((1:n_pmt, eachindex(targets), eachindex(particles)))

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
                    flow_params[:, ix[pmt_ix, t_ix, p_ix]],
                    model.range_min,
                    model.range_max
                )
                for p_ix in eachindex(particles)
            ),
            dims=2
        ))
        : 0.0
        for (pmt_ix, t_ix) in product(1:n_pmt, eachindex(targets)))
    return poiss_llh, shape_llh_gen, log_expec_per_pmt
end


function multi_particle_likelihood(particles; data, targets, model, tf_vec, c_n, feat_buffer=nothing)
    n_pmt = get_pmt_count(eltype(targets))
    @assert length(targets) * n_pmt == length(data)
    pois_llh, shape_llh, _ = evaluate_model(particles, data, targets, model, tf_vec, c_n, feat_buffer=feat_buffer)
    return sum(pois_llh) + sum(shape_llh)
end


function single_cascade_likelihood(logenergy, dir_theta, dir_phi, position, time; data, targets, model, tf_vec, c_n, feat_buffer=nothing)
    dir = sph_to_cart(dir_theta, dir_phi)
    energy = 10^logenergy
    particles = [Particle(position, dir, time, energy, 0., PEMinus)]
    return multi_particle_likelihood(particles, data=data, targets=targets, model=model, tf_vec=tf_vec, c_n=c_n, feat_buffer=feat_buffer)
end

function track_likelihood_fixed_losses(logenergy, dir_theta, dir_phi, position, time; losses, muon_energy, data, targets, model, tf_vec, c_n, feat_buffer=nothing)

    energy = 10^logenergy
    dir = sph_to_cart(dir_theta, dir_phi)
    dist_along = norm.([p.position .- position for p in losses])

    new_loss_positions = [position .+ d .* dir for d in dist_along]
    new_loss_times = time .+ dist_along .* 0.3

    new_loss_energies = [p.energy / muon_energy * energy for p in losses]

    new_losses = Particle.(new_loss_positions, [dir], new_loss_times, new_loss_energies, 0., [PEMinus])

    return multi_particle_likelihood(new_losses, data=data, targets=targets, model=model, tf_vec=tf_vec, c_n=c_n, feat_buffer=feat_buffer)

end


end
