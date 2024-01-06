
export calc_center_of_mass, calculate_inertia_matrix

function _weights_per_module(targets, data)
    n_pmt = get_pmt_count(eltype(targets))
    @assert length(targets) * n_pmt == length(data)

    weights = length.(data)
    weights_rs = reshape(weights, n_pmt, length(targets))
    weights_per_mod = sum(weights_rs, dims=1)[1, :]

    return weights_per_mod
end


function calc_center_of_mass(targets, data)
    weights_per_mod = _weights_per_module(targets, data)
    com = 1 / sum(weights_per_mod) .* sum(weights_per_mod .* [t.position for t in targets])
    return com
end



function calculate_inertia_matrix(targets, data, com)
    com = calc_center_of_mass(targets, data)
    delta_r = [t.position .- com for t in targets]
    delta_r_ssc = ssc.(delta_r)

    weights_per_mod = _weights_per_module(targets, data)
    im = -sum(weights_per_mod .* delta_r_ssc .^ 2)

    return im
end

function calculate_inertia_matrix(targets, data)
    return calculate_inertia_matrix(target, data, calc_center_of_mass(target, data))
end
