using NeutrinoTelescopes
using HDF5
using DataFrames
using Base.Iterators
using Random
using Glob
using ArgParse
using JLD2
using PhotonSurrogateModel

s = ArgParseSettings()
@add_arg_table s begin
    "-i"
    help = "Input files"
    nargs = '+'
    action => :store_arg
    "-o"
    help = "Output file"
    required = true
    "--perturb_medium"
    help = "Train a model with medium perturbation"
    action = :store_true
end
parsed_args = parse_args(ARGS, s; as_symbols=true)

fnames = parsed_args[:i]
outfile = parsed_args[:o]

feature_length = parsed_args[:perturb_medium] ? 8 + 2 : 8

hit_buffer = Matrix{Float32}(undef, 16, Int64(1E8))
features_buffer = Matrix{Float32}(undef, feature_length, Int64(1E8))
hits, features = read_amplitudes_from_hdf5!(fnames, hit_buffer, features_buffer, 1, nothing)
jldsave(outfile, hits=hits, features=features)