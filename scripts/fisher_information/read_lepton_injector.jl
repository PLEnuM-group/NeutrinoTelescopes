using HDF5
using DataFrames
using Random
using StaticArrays
using PhysicsTools
using NeutrinoTelescopes
using PhotonPropagation

using JLD2

fname = "/home/hpc/capn/capn100h/.julia/dev/NeutrinoTelescopes/scripts/fisher_information/test.hd5"

cylinder = Cylinder(SA[0., 0, 0], 1000., 500.)

inj = LIInjector(fname, drop_starting=true, volume=cylinder)

extrema(reduce(hcat, inj.states[:, :Position_final1])[3, :])


save("test.jld2", Dict("inj" => inj))

medium = make_cascadia_medium_properties(0.95f0)

targets = make_n_hex_cluster_detector(7, 200, 20, 50, z_start=475)
d = Detector(targets, medium)
get_bounding_cylinder(d).radius

Base.rand(inj)