using NeutrinoTelescopes
using StaticArrays
using PhysicsTools
using DataFrames
using HDF5
using JLD2
ec = EventCollection()
e = Event()
e[:particles] = [Particle(SA[0, 0, 0], SA[0, 0, 0], 0, 0, 0, PEMinus)]
e[:hits] = DataFrame(time=[1., 2.], pmt_id=[1, 1])
push!(ec, e)

e[:hits]


h5open("/tmp/test.hd5", "w") do hdl
    hdl["e"] = e[:hits]
end


f = h5open("/tmp/test.hd5", "r")
f["e"]
close(f)