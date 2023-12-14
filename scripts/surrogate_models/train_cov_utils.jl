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

    tf_vec = NeuralFlowSurrogate.initialize_normalizers(raw_input) 
    raw_input_tf = NeuralFlowSurrogate.apply_feature_transform(raw_input, tf_vec)
    tf_vec_out = NeuralFlowSurrogate.initialize_normalizers(chol_upper_cbrt) 
    chol_upper_tf = NeuralFlowSurrogate.apply_feature_transform(chol_upper_cbrt, tf_vec_out)

    return ((raw_input_tf, chol_upper_tf), tf_vec, tf_vec_out)
end