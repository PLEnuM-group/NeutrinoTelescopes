using NeutrinoTelescopes
using HDF5
using DataFrames
using Base.Iterators
using Random
using Glob
using ArgParse
using JLD2

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

hit_buffer = Vector{Float32}(undef, Int64(1E8))
feature_length = parsed_args[:perturb_medium] ? 24 + 2 : 24
features_buffer = Matrix{Float32}(undef, feature_length, Int64(1E8))

rng = MersenneTwister(31338)
nsel_frac = 0.9
hits, features = read_times_from_hdf5!(fnames, hit_buffer, features_buffer, nsel_frac, rng)

jldsave(outfile, hits=hits, features=features)