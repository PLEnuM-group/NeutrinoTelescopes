using HDF5
using PhysicsTools
using DataFrames
using Random
using StatsBase

export read_amplitudes_from_hdf5!, read_times_from_hdf5!


function append_onehot_pmt!(output, pmt_ixs)
    lev = 1:16
    output[end-15:end, :] .= (lev .== permutedims(pmt_ixs))
    return output
end


"""
    _convert_grp_attrs_to_features!(grp_attrs, out_matrix::AbstractMatrix)

Converts HDF group attributes to features and stores them in the output vector.

# Arguments
- `grp_attrs`: A dictionary containing group attributes.
- `out_matrix`: An abstract matrix to store the converted features.
"""
function _convert_grp_attrs_to_features!(grp_attrs, out_matrix::AbstractMatrix)
    
    out_matrix[1, :] .= log(grp_attrs["distance"])
    out_matrix[2, :] .= log(grp_attrs["energy"])

    out_matrix[3:5, :] .= sph_to_cart(grp_attrs["dir_theta"], grp_attrs["dir_phi"])
    out_matrix[6:8, :] .= sph_to_cart(grp_attrs["pos_theta"], grp_attrs["pos_phi"])

    if size(out_matrix, 1) == 10 || size(out_matrix, 1) == 26
        out_matrix[9, :] .= grp_attrs["abs_scale"]
        out_matrix[10, :] .= grp_attrs["sca_scale"]
    end
    return out_matrix
end

function _convert_grp_attrs_to_features!(grp_attrs, out_vector::AbstractVector)
    return _convert_grp_attrs_to_features!(grp_attrs, reshape(out_vector, length(out_vector), 1))[:, 1]
end


"""
    count_hits_per_pmt!(grp, feature_vector::AbstractArray, target_vector::AbstractArray)

Count the number of hits per PMT, calculate input features in store them in the target vectors.

# Arguments
- `grp`: The group of hits.
- `feature_vector`: The feature vector.
- `target_vector`: The target vector.

# Returns
- `feature_vector`: The updated feature vector.
- `target_vector`: The updated target vector.
"""
function count_hits_per_pmt!(grp, feature_vector::AbstractArray, target_vector::AbstractArray)

    grp_attrs = attrs(grp)
     _convert_grp_attrs_to_features!(grp_attrs, feature_vector)

    if length(grp) == 0
        target_vector .= 0
    else

        hits = DataFrame(grp[:, :], [:tres, :pmt_id, :total_weight])

        hits_per_pmt = combine(groupby(hits, :pmt_id), :total_weight => sum => :weight_sum)
        pmt_id_ix = Int.(hits_per_pmt[:, :pmt_id])
        target_vector[pmt_id_ix] .= hits_per_pmt[:, :weight_sum]
    end
    return feature_vector, target_vector
end



"""
    read_hit_times_per_pmt!(grp, feature_matrix::AbstractArray, target_vector::AbstractArray, limit=200)

Reads hit times per PMT from a group `grp` and populates `feature_matrix` and `target_vector` with the corresponding data.

## Arguments
- `grp`: A group containing hit time data.
- `feature_matrix`: An abstract array to store the features extracted from the hit time data.
- `target_vector`: An abstract array to store the target values (hit times) corresponding to the features.
- `limit`: An optional argument specifying the maximum number of data points to read. Default is 200.

## Returns
- `feature_matrix`: The feature matrix containing the extracted features.
- `target_vector`: The target vector containing the hit times.

"""
function read_hit_times_per_pmt!(grp, feature_matrix::AbstractArray, target_vector::AbstractArray, limit=200)

    grp_data = grp[:, :]
    grp_attrs = attrs(grp)

    grplen = size(grp_data, 1)

    sumw = sum(grp_data[:, 3])

    out_length = !isnothing(limit) ? min(limit, grplen) : grplen

    weights = FrequencyWeights(grp_data[:, 3], sumw)
    sampled = sample(1:grplen, weights, ceil(Int64, out_length), replace=true)

    grp_data = grp_data[sampled, :]
    
    @views _convert_grp_attrs_to_features!(grp_attrs, feature_matrix[:, 1:out_length])
    target_vector[1:out_length] .= grp_data[:, 1]
    @views append_onehot_pmt!(feature_matrix[:, 1:out_length], grp_data[:, 2])

    return @views feature_matrix[:, 1:out_length], target_vector[1:out_length]

end

"""
    read_amplitudes_from_hdf5!(fnames, nhits_buffer, features_buffer, nsel_frac=0.8, rng=Random.default_rng())

Reads expected photon counts from HDF5 files and populates the `nhits_buffer` and `features_buffer` matrices.

# Arguments
- `fnames`: An array of file names to read from.
- `nhits_buffer`: A matrix to store the hit counts per PMT.
- `features_buffer`: A matrix to store the features.
- `nsel_frac`: The fraction of datasets to select from each file (default: 0.8).
- `rng`: The random number generator to use for shuffling the datasets (default: `Random.default_rng()`).

# Returns
- `nhits_buffer_view`: A view of the `nhits_buffer` matrix containing the read data.
- `features_buffer_view`: A view of the `features_buffer` matrix containing the read data.
"""
function read_amplitudes_from_hdf5!(fnames, nhits_buffer, features_buffer, nsel_frac=0.8, rng=Random.default_rng())
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
                nhits_buffer[:, ix] .= 0.
                @views count_hits_per_pmt!(grp, features_buffer[:, ix], nhits_buffer[:, ix])
                ix += 1
                
            end
        end
    end

    features_buffer_view = @view features_buffer[:, 1:ix-1]
    nhits_buffer_view = @view nhits_buffer[:, 1:ix-1]

    return nhits_buffer_view, features_buffer_view
end



"""
    read_times_from_hdf5!(fnames, hit_buffer, features_buffer, nsel_frac=0.8, rng=Random.default_rng())

Reads photon hit times from HDF5 files and stores them in the hit_buffer and features_buffer arrays.

# Arguments
- `fnames`: An array of file names to read from.
- `hit_buffer`: An array to store the hit times.
- `features_buffer`: An array to store the features.
- `nsel_frac`: The fraction of datasets to select from each file (default is 0.8).
- `rng`: The random number generator to use for shuffling the datasets (default is Random.default_rng()).

# Returns
- `hit_buffer_view`: A view of the hit_buffer array containing the read hit times.
- `features_buffer_view`: A view of the features_buffer array containing the read features.
"""
function read_times_from_hdf5!(fnames, hit_buffer, features_buffer, nsel_frac=0.8, rng=Random.default_rng())

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
                _, h = @views read_hit_times_per_pmt!(grp, features_buffer[:, ix:end], hit_buffer[ix:end], 100)
                nhits = length(h)
                ix += nhits
            end
        end
    end

    println("Read $(ix-1) hits")

    features_buffer_view = @view features_buffer[:, 1:ix-1]
    hit_buffer_view = @view hit_buffer[1:ix-1]
    return hit_buffer_view, features_buffer_view
end
