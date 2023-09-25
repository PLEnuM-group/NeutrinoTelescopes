using DataFrames
using HDF5
using NeutrinoTelescopes
using PhotonPropagation
using Random
using JLD2
using MLUtils
using StaticArrays
p = POM(SA[0., 0., 0.], 1)

@show DataFrame(p.pmt_coordinates',[:theta, :phi])

fnames = [
    "/home/saturn/capn/capn100h/snakemake/photon_tables/lightsabre/hits/photon_table_lightsabre_0_hits.hd5",
    "/home/saturn/capn/capn100h/snakemake/photon_tables/lightsabre/hits/photon_table_lightsabre_1_hits.hd5"
]


hit_buffer = Vector{Float64}(undef, Int64(1E8))
pmt_ixs_buffer = Vector{Int64}(undef, Int64(1E8))
features_buffer = Matrix{Float64}(undef, 24, Int64(1E8))

rng = MersenneTwister(31338)
nsel_frac = 0.9
hits, features, tf_vec = read_pmt_hits!([fname], hit_buffer, pmt_ixs_buffer, features_buffer, nsel_frac, rng)

length(hits) / 1E8

parsed_args = Dict(:s => 0.1)

if parsed_args[:s] > 0
    @inbounds for i in eachindex(hits)
        hits[i] = randn() * parsed_args[:s]
    end
end

@allocated data = (tres=hits, label=features)


kf = kfolds(data; k=5)

first(kf)[1]
