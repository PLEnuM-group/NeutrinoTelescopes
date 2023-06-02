using HDF5
using CairoMakie
using NeutrinoTelescopes
using PhotonPropagation
using Rotations
using LinearAlgebra
using DataFrames
using StaticArrays
using PhysicsTools
using Distributions
using PairPlots
using StatsBase


pmt_pos = get_pmt_positions(make_pone_module(SA[0., 0., 0.], 1), RotMatrix3(I))

fid = h5open(fname, "r")

fid["pmt_hits"]["dataset_1041"][:, :]

fname = joinpath(ENV["WORK"], "photon_tables/extended/photon_table_extended_0.hd5")

stats = []

for ds in fid["photons"]
    atts = attrs(ds)
    #df = DataFrame(ds[:, :], [:time, :pmt_ix])
    df = DataFrame(ds[:, :],  [:tres, :pos_x, :pos_y, :pos_z, :total_weight])

    nhits = sum(df[:, "total_weight"])

    direction = sph_to_cart(acos(atts["dir_costheta"]), atts["dir_phi"])
    view_angle = rad2deg.(acos(dot([0, 0, -atts["distance"]], direction) / atts["distance"]))



    push!(stats, (distance= atts["distance"], nhits=nrow(df), view_angle=view_angle))

    # per_pmt = combine(groupby(df, :pmt_ix), nrow)
    # pmt_ix_max = per_pmt[argmax(combine(groupby(df, :pmt_ix), nrow)[:, :nrow]), :pmt_ix]
    # pmt_pos_max = pmt_pos[Int(pmt_ix_max)]
end

stats = DataFrame(stats)
stats[!, :loghits] = log10.(stats[:, :nhits])
pairplot(stats[:, [:loghits, :distance, :view_angle]])


h = fit(Histogram,
    (stats[:, :view_angle], log10.(stats[:, :distance])), 
    FrequencyWeights(stats[:, :nhits]),
    (0:0.2:1, 1:0.1:2)
)



heatmap(h.edges[1], h.edges[2], h.weights, axis=(xlabel="cos(θ)", ylabel="log(distance)"))


fig = Figure()
ax = Axis(fig[1, 1], yscale=log10, limits=(0, 200, 1, 1E5))
hexbin!(ax, distances, nhits)



fig


sort(nhits)

close(fid)