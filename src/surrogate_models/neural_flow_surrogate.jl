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

export ArrivalTimeSurrogate, RQSplineModel, PhotonSurrogate, AbsScaRQNormFlowHParams, AbsScaPoissonExpModel
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




sqnorm(x) = sum(abs2, x)








function dataframe_to_matrix(df)
    feature_matrix = Matrix{Float64}(undef, 9, nrow(df))
    feature_matrix[1, :] .= log.(df[:, :distance])
    feature_matrix[2, :] .= log.(df[:, :energy])

    feature_matrix[3:5, :] .= reduce(hcat, sph_to_cart.(df[:, :dir_theta], df[:, :dir_phi]))
    feature_matrix[6:8, :] .= reduce(hcat, sph_to_cart.(df[:, :pos_theta], df[:, :pos_phi]))
    feature_matrix[9, :] .= df[:, :pmt_id]
    return feature_matrix
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


function count_hit_per_pmt(grp, with_perturb=true)

    grp_attrs = attrs(grp)

    if haskey(grp_attrs, "abs_scale")
        feature_dim = 10
    else
        feature_dim = 8
    end


    feature_vector = zeros(Float64, feature_dim)
    hit_vector = zeros(Float64, 16)

    feature_vector[1] = log.(grp_attrs["distance"])
    feature_vector[2] = log.(grp_attrs["energy"])

    feature_vector[3:5] = reduce(hcat, sph_to_cart.(grp_attrs["dir_theta"], grp_attrs["dir_phi"]))
    feature_vector[6:8] = reduce(hcat, sph_to_cart.(grp_attrs["pos_theta"], grp_attrs["pos_phi"]))

    if with_perturb
        feature_vector[9] = grp_attrs["abs_scale"]
        feature_vector[10] = grp_attrs["sca_scale"]
    end


    if size(grp, 1) == 0
        return feature_vector, hit_vector
    end

    hits = DataFrame(grp[:, :], [:tres, :pmt_id, :total_weight])

    hits_per_pmt = combine(groupby(hits, :pmt_id), :total_weight => sum => :weight_sum)
    pmt_id_ix = Int.(hits_per_pmt[:, :pmt_id])
    hit_vector[pmt_id_ix] .= hits_per_pmt[:, :weight_sum]

    return feature_vector, hit_vector
end

function create_pmt_table(grp, limit=nothing,  with_perturb=true)

    grp_data = grp[:, :]
    grp_attrs = attrs(grp)

    grplen = size(grp_data, 1)

    sumw = sum(grp_data[:, 3])
    weights = FrequencyWeights(grp_data[:, 3], sumw)
    sampled = sample(1:grplen, weights, ceil(Int64, sumw), replace=true)

    grp_data = grp_data[sampled, :]

    out_length = !isnothing(limit) ? min(limit, size(grp, 1)) : size(grp, 1)

    if with_perturb
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

    if with_perturb
        feature_matrix[9, :] .= grp_attrs["abs_scale"]
        feature_matrix[10, :] .= grp_attrs["sca_scale"]
    end


    data_mat = grp[1:out_length, :]

    hit_times = data_mat[:, 1]
    pmt_ixs = data_mat[:, 2]

    return feature_matrix, pmt_ixs, hit_times

end

function read_pmt_number_of_hits!(fnames, nhits_buffer, features_buffer, nsel_frac=0.8, rng=default_rng(), with_perturb=true)
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
                f,h = count_hit_per_pmt(grp, with_perturb)

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


function read_pmt_hits!(fnames, hit_buffer, pmt_ixs_buffer, features_buffer, nsel_frac=0.8, rng=default_rng(), with_perturb=true)

    ix = 1

    feature_length = with_perturb ? 10 : 8

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
                f, pix, h = create_pmt_table(grp, 100, with_perturb)
                nhits = length(h)

                if ix+nhits-1 > length(hit_buffer)
                    @warn "Input buffer full, might now read every file"
                end

                hit_buffer[ix:ix+nhits-1] .= h
                features_buffer[1:feature_length, ix:ix+nhits-1] .= f
                pmt_ixs_buffer[ix:ix+nhits-1] .= pix
                ix += nhits
            end
        end
    end

    pmt_ixs_buffer_view = @view pmt_ixs_buffer[1:ix]
    features_buffer_view = @view features_buffer[:, 1:ix]
    hit_buffer_view = @view hit_buffer[1:ix]

    preproc_view = @view features_buffer_view[1:feature_length, :]

    features_buffer, tf_vec = preproc_labels!(preproc_view, preproc_view)
    append_onehot_pmt!(features_buffer_view, pmt_ixs_buffer_view)
    return hit_buffer_view, features_buffer_view, tf_vec
end


function _calc_flow_input!(
    particle_pos,
    particle_dir,
    particle_energy,
    target_pos,
    tf_vec::AbstractVector,
    output;
    abs_scale=1.,
    sca_scale=1.)

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

function _calc_flow_input!(
    particle::Particle,
    target::PhotonTarget,
    tf_vec::AbstractVector,
    output;
    abs_scale=1.,
    sca_scale=1.)

    if particle_shape(particle) == Track()
        particle = shift_to_closest_approach(particle, target.shape.position)
    end

    return _calc_flow_input!(particle.position, particle.direction, particle.energy, target.shape.position, tf_vec, output, abs_scale=abs_scale, sca_scale=sca_scale)
    
end



"""
    _calc_flow_input!(particles::AbstractVector{<:Particle}, targets::AbstractVector{<:PhotonTarget}, tf_vec, output)

Mutating version. Flow input is written into output
"""
function _calc_flow_input!(particles::AbstractVector{<:Particle}, targets::AbstractVector{<:PhotonTarget}, tf_vec, output; abs_scale=1., sca_scale=1.)
  

    out_ix = LinearIndices((eachindex(particles), eachindex(targets)))
    
    for (p_ix, t_ix) in product(eachindex(particles), eachindex(targets))
        particle = particles[p_ix]
        target = targets[t_ix]

        ix = out_ix[p_ix, t_ix]

        outview = @view output[:, ix]

        _calc_flow_input!(particle, target, tf_vec, outview, abs_scale=abs_scale, sca_scale=sca_scale)
        
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
function calc_flow_input!(particle::Particle, target::PhotonTarget, tf_vec::AbstractVector, output; abs_scale=1., sca_scale=1.)
    flen = length(tf_vec)
    outview = @view output[1:flen]
    _calc_flow_input!(particle, target, tf_vec, outview, abs_scale=abs_scale, sca_scale=sca_scale)
    
    

    n_pmt = get_pmt_count(target)
    lev = 1:n_pmt
    output[flen+1:flen+n_pmt, :] .= (lev .== permutedims(pmt_ixs))
    return output
end


"""
    calc_flow_input!(particles::AbstractVector{<:Particle}, targets::AbstractVector{<:PhotonTarget}, tf_vec, output)

Mutating version. Flow input is written into output
"""
function calc_flow_input!(particles::AbstractVector{<:Particle}, targets::AbstractVector{<:PhotonTarget}, tf_vec, output; abs_scale=1., sca_scale=1.)
    n_pmt = get_pmt_count(eltype(targets))
    out_ix = LinearIndices((1:n_pmt, eachindex(particles), eachindex(targets)))
    
    flen = length(tf_vec)
    @show flen

    for (p_ix, t_ix) in product(eachindex(particles), eachindex(targets))
        particle = particles[p_ix]
        target = targets[t_ix]

        for pmt_ix in 1:n_pmt
            ix = out_ix[pmt_ix, p_ix, t_ix]
            outview = @view output[1:flen, ix]
            _calc_flow_input!(particle, target, tf_vec, outview, abs_scale=abs_scale, sca_scale=sca_scale)
        end

        ix = out_ix[1:n_pmt, p_ix, t_ix]
        output[flen+1:flen+n_pmt, ix] .= Matrix(one(eltype(output)) * I, n_pmt, n_pmt)
        
    end

    return output
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
function get_log_amplitudes(particles, targets, model::PhotonSurrogate; feat_buffer, device=gpu, abs_scale=1., sca_scale=1.)
    n_pmt = get_pmt_count(eltype(targets))

    tf_vec = model.amp_transformations

    input_size = size(model.amp_model.layers[1].weight, 2)

    amp_buffer = @view feat_buffer[1:input_size, 1:length(targets)*length(particles)]
    _calc_flow_input!(particles, targets, tf_vec, amp_buffer)

    if input_size == 10
        amp_buffer[9, :] .= abs_scale
        amp_buffer[10, :] .= sca_scale
    end


    input = amp_buffer

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



end

# backwards compat for old bson files
ExtendedCascadeModel = NeuralFlowSurrogate