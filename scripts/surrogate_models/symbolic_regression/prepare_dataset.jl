using ArgParse
using JLD2
using DataFrames
using TOML
using Random

include("utils.jl")


function prepare_datasets(config_path::String, output_dir::String)
    # Load the configuration file
    config = TOML.parsefile(config_path)

    # Load the input data
    input_file = config["input"]["filename"]
    data = JLD2.load(input_file)["data"]

    # Apply selection criteria
    if haskey(config, "selection")
        selection = config["selection"]
        selected_data = apply_selection(data, selection)
    else
        selected_data = data
    end


    indices = shuffle(1:nrow(selected_data))

    train_size = config["input"]["train_size"]

    # Split the data into training and testing sets
    if nrow(selected_data) > 2*train_size
        train_data = selected_data[indices[1:train_size], :]
        test_data = selected_data[indices[train_size+1:end], :]
    else
        train_cut = ceil(Int, 0.7 * nrow(selected_data))
        train_data = selected_data[indices[1:train_cut], :]
        test_data = selected_data[indices[train_cut+1:end], :]
    end

    # Save the training and testing sets
    train_file = joinpath(output_dir, "train.jld2")
    test_file = joinpath(output_dir, "test.jld2")
    jldsave(train_file, train=train_data)
    jldsave(test_file, test=test_data)
end

function main()

    s = ArgParseSettings()
    @add_arg_table s begin
        "--config"
        help = "Path to the configuration file"
        required = true
        arg_type = String
        "--output"
        help = "Output directory"
        required = true
        arg_type = String
    end

    args = parse_args(ARGS, s)
    config_path = args["config"]
    output_dir = args["output"]

    if !isdir(output_dir)
        mkdir(output_dir)
    end

    prepare_datasets(config_path, output_dir)
end

main()