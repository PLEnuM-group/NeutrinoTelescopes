using CSV
using DataFrames
using CairoMakie
using PhysicsTools
using LinearAlgebra
using PhotonPropagation
using NeutrinoTelescopes
using PhysicsTools
using StaticArrays
using Glob
using HDF5
using Interpolations
using Distributions
using PhysicalConstants.CODATA2018
using Unitful




begin
    coords = Matrix{Float64}(undef, 2, 16)
    # upper 
    coords[1, 1:4] .= deg2rad(90 - 25)
    coords[2, 1:4] = (range(0; step=π / 2, length=4))

    # upper 2
    coords[1, 5:8] .= deg2rad(90 - 57.5)
    coords[2, 5:8] = (range(π / 4; step=π / 2, length=4))

    # lower 2
    coords[1, 9:12] .= deg2rad(90 + 25)
    coords[2, 9:12] = [π/2, 0, 3*π/2, π]

    # lower
    coords[1, 13:16] .= deg2rad(90 + 57.5)
    coords[2, 13:16] = [π/4, 7/4*π, 5/4*π, 3/4*π]
end


function calc_coordinates!(df)
    pos_in = Matrix{Float64}(df[:, ["in_x", "in_y", "in_z"]])
    norm_in = norm.(eachrow(pos_in))
    pos_in_normed = pos_in ./ norm_in
   
    df[!, :in_norm_x] .= pos_in_normed[:, 1]
    df[!, :in_norm_y] .= pos_in_normed[:, 2]
    df[!, :in_norm_z] .= pos_in_normed[:, 3]

    in_p_cart = Matrix{Float64}((df[:, [:in_px, :in_py, :in_pz]]))

    norm_p = norm.(eachrow(in_p_cart))
    in_p_cart_norm = in_p_cart ./ norm_p

    df[!, :in_p_norm_x] .= in_p_cart_norm[:, 1]
    df[!, :in_p_norm_y] .= in_p_cart_norm[:, 2]
    df[!, :in_p_norm_z] .= in_p_cart_norm[:, 3]
   return df
end

#sim_path = joinpath(ENV["WORK"], "geant4_pmt")
#sim_path = "/home/chrhck/geant4_sims/P-OM photons 30 cm sphere/"
sim_path = joinpath(ENV["WORK"], "geant4_pmt/30cm_sphere")
files = glob("*.csv", sim_path)
coords_cart = reduce(hcat, sph_to_cart.(eachcol(coords)))


all_hc_1 = Vector{Float64}[]
all_hc_2 = Vector{Float64}[]

all_hc = [all_hc_1, all_hc_2]

wavelengths = Float64[]
total_acc_1 = Float64[]
total_acc_2 = Float64[]

pmt_grp_1 = collect(1:16)[div.(0:15,  4) .% 2 .== 0]
pmt_grp_2 = collect(1:16)[div.(0:15,  4) .% 2 .== 1]


for f in files

    df = DataFrame(CSV.File(f))
    wl = round(ustrip(u"nm", PlanckConstant * SpeedOfLightInVacuum ./ ( df[1, :in_E]u"eV")))
    push!(wavelengths, wl)
    calc_coordinates!(df)

   
    for pmt_ix in 1:16
        pmt_grp = get_pom_pmt_group(pmt_ix)

        pmt_coords = coords_cart[:, pmt_ix]

        in_pos_cart_norm = Matrix{Float64}(df[!, [:in_norm_x, :in_norm_y, :in_norm_z]])
        rel_costheta = dot.(eachrow(in_pos_cart_norm), Ref(pmt_coords))

        hit_pmt::BitVector = (
            (df[:, :out_VolumeName] .== "photocathode" .|| df[:, :out_VolumeName] .== "photocathodeTube") .&&
            df[:, :out_Volume_CopyNo] .== (pmt_ix-1) .&& 
            df[:, :out_ProcessName] .== "OpAbsorption"
        )

        rel_theta = acos.(rel_costheta[hit_pmt])

        push!(all_hc[pmt_grp], rel_theta)

    end
    
    any_hit = ((df[:, :out_VolumeName] .== "photocathode" .|| df[:, :out_VolumeName] .== "photocathodeTube") .&&
          df[:, :out_ProcessName] .== "OpAbsorption"
    )

    mask_grp_1 = any_hit .&& (div.(df[:, :out_Volume_CopyNo],  4) .% 2 .== 0)
    mask_grp_2 = any_hit .&& (div.(df[:, :out_Volume_CopyNo],  4) .% 2 .== 1)

    # mask_grp_1 / mask_grp_2 are the probabilities to hit any pmt from the PMT group
    # have to account the number of PMTs per group to get the probability for a specific pmt
    push!(total_acc_1, sum(mask_grp_1) / nrow(df) / length(pmt_grp_1))
    push!(total_acc_2, sum(mask_grp_2) / nrow(df) / length(pmt_grp_2))

end

d1 = Distributions.fit(Rayleigh, (reduce(vcat, all_hc_1))) 
d2 = Distributions.fit(Rayleigh, (reduce(vcat, all_hc_2))) 

fname = joinpath(@__DIR__, "../assets/pmt_acc.hd5")
h5open(fname, "w") do fid
    fid["acc_pmt_grp_1"] = total_acc_1
    fid["acc_pmt_grp_2"] = total_acc_2
    fid["wavelengths"] = wavelengths
    fid["sigma_grp_1"] = d1.σ 
    fid["sigma_grp_2"] = d2.σ
end


fname = joinpath(@__DIR__, "../../PhotonPropagation/assets/pmt_acc.hd5")
h5open(fname, "r") do fid
    fig = Figure()
    ax = Axis(fig[1, 1], xticks = WilkinsonTicks(8), xminorticks = IntervalsBetween(10), xminorticksvisible=true)
    lines!(ax, fid["wavelengths"][:], fid["acc_pmt_grp_1"][:])
    lines!(ax, fid["wavelengths"][:], fid["acc_pmt_grp_2"][:])
    fig
end