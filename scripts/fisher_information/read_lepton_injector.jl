using HDF5
using DataFrames
using Random
using StaticArrays
using PhysicsTools
using NeutrinoTelescopes

using JLD2

fname = "/home/hpc/capn/capn100h/.julia/dev/NeutrinoTelescopes/scripts/fisher_information/test.hd5"

cylinder = Cylinder(SA[0., 0, 0], 1000., 500.)

inj = LIInjector(fname, drop_starting=true, volume=cylinder)

save("test.jld2", Dict("inj" => inj))



Base.rand(inj)