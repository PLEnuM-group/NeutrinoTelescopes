using Glob
using LinearAlgebra
using JLD2
using NeutrinoTelescopes

function load_data_from_dir(path, type, nfiles=nothing)

    data_files = glob("training_data_$(type)_*.jld2", path)
    
    if !isnothing(nfiles)
        data_files = data_files[1:nfiles]
    end

    training_data = []
    for file in data_files
        push!(training_data, jldopen(file)["data"])
    end
    training_data = reduce(vcat, training_data)

    raw_input = reduce(hcat, training_data.raw_input)
    chol_upper = reduce(hcat, training_data.chol_upper)
    chol_upper_cbrt = cbrt.(chol_upper)

    #min_val = minimum(chol_upper)

    #chol_upper_shift_log = log.(chol_upper .+ (abs(min_val) +1))

    raw_input_tf, tf_in = fit_normalizer!(raw_input)
    chol_upper_tf, tf_out = fit_normalizer!(chol_upper_cbrt)


    return ((raw_input_tf, chol_upper_tf), tf_in, tf_out)
end