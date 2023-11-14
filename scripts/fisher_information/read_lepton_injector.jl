using HDF5
using DataFrames
using Random
using StaticArrays
using PhysicsTools
using NeutrinoTelescopes
using PhotonPropagation
using JLD2
using StatsBase
using CairoMakie
using KernelDensityEstimate


fname = "/home/saturn/capn/capn100h/snakemake/leptoninjector-lightsabre-0.hd5"

hdl = h5open(fname)

table = hdl["RangedInjector0/properties"][:]
energy_at_dist_kde = load("/home/hpc/capn/capn100h/.julia/dev/NeutrinoTelescopes/scripts/fisher_information/muon_table/test_muons_hist.jld2", "table")

targets = make_n_hex_cluster_detector(7, 50, 20, 50, z_start=475)
medium = make_cascadia_medium_properties(0.95f0)
det = Detector(targets, medium)
cyl = get_bounding_cylinder(det, padding_top=100, padding_side=100)

mask = Bool[]
muons = Particle[]


distance_table = DataFrame(nu_energy=>Float64[], mu_energy=>Float64[], distance=>Float64[])
for row in table
    p = Particle(SA_F64[row.x, row.y, row.z], sph_to_cart(row.zenith, row.azimuth), 0., row.totalEnergy * (1-row.finalStateY), 0., ptype_for_code(row.finalType1))
    isec = get_intersection(cyl, p)
    isecs = !isnothing(isec.first) && (isec.first > 0)
    if !isecs
        continue
    end
    push!(distance_table, (nu_energy=row.totalEnergy, mu_energy=p.energy, distance=isec.))


end

muons



function get_df_bin(df, initial_energy, distance)
    sorted_keys = [k.initial_energy for k in sort(keys(groupby(df, :initial_energy)))]
    bix = searchsortedfirst(sorted_keys, initial_energy) -1
    if( bix == 0) |( bix == lastindex(sorted_keys))
        return nothing
    end
    key1 = sorted_keys[bix]

    sorted_keys = [k.distance for k in sort(keys(groupby(df, :distance)))]
    bix = searchsortedfirst(sorted_keys, distance) -1
    if (bix == 0) | (bix == lastindex(sorted_keys))
        return nothing
    end
    key2 = sorted_keys[bix]

    return groupby(df, [:initial_energy, :distance])[(initial_energy=key1, distance=key2)]

end

muon = deepcopy(muons[4])
isec = get_intersection(cyl, muon)
dist = isec.first

e_lost = first(get_df_bin(energy_at_dist_kde, muon.energy, dist)[:, :kde_log_final_energy])

xs = 0:0.01:7
lines(xs, e_lost(collect(xs)))



cylinder = Cylinder(SA[0., 0, 0], 1000., 500.)

inj = LIInjector(fname, drop_starting=true, volume=cylinder)

extrema(reduce(hcat, inj.states[:, :Position_final1])[3, :])


save("test.jld2", Dict("inj" => inj))

medium = make_cascadia_medium_properties(0.95f0)

targets = make_n_hex_cluster_detector(7, 200, 20, 50, z_start=475)
d = Detector(targets, medium)
get_bounding_cylinder(d).radius

Base.rand(inj)