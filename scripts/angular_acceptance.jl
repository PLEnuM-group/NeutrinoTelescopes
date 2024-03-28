using CSV
using DataFrames
using CairoMakie
using PhysicsTools
using LinearAlgebra
using PhotonPropagation
using NeutrinoTelescopes
using StaticArrays
using Glob
using HDF5
using Interpolations
using Distributions
using PhysicalConstants.CODATA2018
using Unitful
using Rotations
using LinearAlgebra
using JSON3
using Healpix

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
sim_path = joinpath(ENV["ECAPSTOR"], "geant4_pmt/30cm_sphere")
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
            df[:, :out_Volume_CopyNo] .== (pmt_ix-1)
            #df[:, :out_ProcessName] .== "OpAbsorption"
        )

        rel_theta = acos.(rel_costheta[hit_pmt])

        push!(all_hc[pmt_grp], rel_theta)

    end
    
    any_hit = ((df[:, :out_VolumeName] .== "photocathode" .|| df[:, :out_VolumeName] .== "photocathodeTube")
          #df[:, :out_ProcessName] .== "OpAbsorption"
    )

    mask_grp_1 = any_hit .&& (div.(df[:, :out_Volume_CopyNo],  4) .% 2 .== 0)
    mask_grp_2 = any_hit .&& (div.(df[:, :out_Volume_CopyNo],  4) .% 2 .== 1)

    # mask_grp_1 / mask_grp_2 are the probabilities to hit any pmt from the PMT group
    # have to account the number of PMTs per group to get the probability for a specific pmt
    push!(total_acc_1, sum(mask_grp_1) / nrow(df))
    push!(total_acc_2, sum(mask_grp_2) / nrow(df))
end


azimuth_hit = []

m_accepted = HealpixMap{Float64, RingOrder}(32)
m_all = HealpixMap{Float64, RingOrder}(32)
m_all[:] .= UNSEEN
m_accepted[:] .= UNSEEN

coords_rot = []
for col in eachcol(coords)
    R = calc_rot_matrix(SA[0.0, 0.0, 1.0], SA[1.0, 0.0, 0.0])
    cart = sph_to_cart(col[1], col[2])
    col = cart_to_sph((R * cart)...)
    push!(coords_rot, col)
    
end
coords_rot = Matrix(reduce(hcat, coords_rot))

for f in files

    df = DataFrame(CSV.File(f))
    calc_coordinates!(df)

    in_dir_cart = Matrix{Float64}(df[!, [:in_norm_x, :in_norm_y, :in_norm_z]])
    R = calc_rot_matrix(SA[0.0, 0.0, 1.0], SA[1.0, 0.0, 0.0])
    in_dir_cart = Ref(R) .* eachrow(in_dir_cart)
    
    in_dir_sph = reduce(hcat, cart_to_sph.(in_dir_cart))
    
    any_hit = ((df[:, :out_VolumeName] .== "photocathode" .|| df[:, :out_VolumeName] .== "photocathodeTube")
          #df[:, :out_ProcessName] .== "OpAbsorption"
    )

    pix_id = map(x -> ang2pix(m, vec2ang(x...)...), in_dir_cart)
    pix_id_accepted = pix_id[any_hit]
    mask = m_accepted[pix_id_accepted] .== UNSEEN
    m_accepted[pix_id_accepted[mask]] .= 0
    m_accepted[pix_id_accepted] .+= 1

    mask = m_all[pix_id] .== UNSEEN
    m_all[pix_id[mask]] .= 0
    m_all[pix_id] .+= 1

    push!(azimuth_hit, in_dir_sph[2, any_hit])
end

any_unseen = (m_accepted .== UNSEEN) .|| (m_all .== UNSEEN)
m_ratio = m_accepted / m_all
m_ratio[any_unseen] .= UNSEEN

fig = Figure()
ax = Axis(fig[1, 1], aspect=2)
img, mask, anymasked = mollweide(m_ratio)
hm = heatmap!(ax, img', show_axis = false)
hidespines!(ax)
hidedecorations!(ax)
# colat, long
coords_rot_lat = reduce(hcat, map(x -> collect(vec2ang(x...)), sph_to_cart.(eachcol(coords_rot))))
coords_rot_lat[1, :] = colat2lat.(coords_rot[1, :])
coords_rot_lat[2, :] .-= π


coords_proj = Point2f.(map(x -> collect(mollweideproj(x...)[[2,3]]), eachcol(coords_rot_lat)))
coords_proj ./= 2
coords_proj .+= Ref([0.5, 0.5])
relative_projection = Makie.camrelative(ax.scene);

coords_proj
scatter!(relative_projection, coords_proj, color=:blue)
Colorbar(fig[2, 1], hm, vertical = false, label="Acceptance")
fig


azimuth_hit = reduce(vcat, azimuth_hit)

bins = 0:0.05:2*π
fig, ax, h = hist(azimuth_hit, bins=bins, axis=(xlabel="Azimuth angle [rad]", ylabel="Counts"))


#vlines!(ax, coords_rot[2, :], color=(:red, 0.5))
fig



rand(size(azimuth_hit))
uni_hits = rand(size(azimuth_hit)[1]).*2 .*π

fig = Figure()
ax = Axis3(fig[1, 1], aspect = (1, 1, 1), azimuth = deg2rad(30))
ax2 = Axis3(fig[1, 2], aspect = (1, 1, 1), azimuth = deg2rad(60))
coords_cart = Point3f.(sph_to_cart.(eachcol(Matrix(coords_rot))))

coords_cart[1:8] .+= Ref([2, 0, 0])
coords_cart[9:16] .-= Ref([2, 0, 0])
dirs_cart = Vec3f.(coords_cart)
arrows!(ax, 5 .*coords_cart, dirs_cart)
arrows!(ax2, 5 .*coords_cart, dirs_cart)
fig

ax2 = Axis3(fig[1, 2], aspect = (1, 1, 1), azimuth = deg2rad(60))
for coord in eachcol(coords_rot[:, 1:8])
    cc = sph_to_cart(coord)
    line = hcat(cc, 1.2*cc)
    arrows!(ax, cc..., cc...)
end


for coord in eachcol(coords_rot[:, 9:16])
    cc = sph_to_cart(coord)
    line = hcat(cc, 1.2*cc)
    lines!(ax, line, color=:blue)
    lines!(ax2, line, color=:blue)
end
fig


wlsort = sortperm(wavelengths)

wavelengths = wavelengths[wlsort]
total_acc_1 = total_acc_1[wlsort]
total_acc_2 = total_acc_2[wlsort]
 
fig, ax, _ = lines(wavelengths, total_acc_1)
lines!(ax, wavelengths, total_acc_2)
fig

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

fname = joinpath(@__DIR__, "../assets/rel_pmt_acc.hd5")
h5open(fname, "w") do fid

    rel_acc = total_acc_1 ./ total_acc_2
    fid["rel_acc_pmt_grp_1"] = median(rel_acc)
    fid["acc_pmt_grp_2"] = total_acc_2 *  length(pmt_grp_2)
    fid["wavelengths"] = wavelengths
    fid["sigma_grp_1"] = d1.σ 
    fid["sigma_grp_2"] = d2.σ
end


fname = joinpath(@__DIR__, "../assets/pmt_acc.hd5")
h5open(fname, "r") do fid
    fig = Figure()
    ax = Axis(fig[1, 1], xticks = WilkinsonTicks(8), xminorticks = IntervalsBetween(10), xminorticksvisible=true)
    lines!(ax, fid["wavelengths"][:], fid["acc_pmt_grp_1"][:])
    lines!(ax, fid["wavelengths"][:], fid["acc_pmt_grp_2"][:])
    fig
end





using DataFrames
using StaticArrays
using Cthulhu
using Rotations
using Random
using LinearAlgebra
using NeutrinoTelescopes
using PhotonPropagation
n = 1000000
positions = rand(SVector{3, Float64}, n)
directions = rand(SVector{3, Float64}, n)
directions ./= norm.(directions)
total_weights = rand(n)



photons = DataFrame(position=positions, direction=directions, total_weight=total_weights, module_id=ones(n), wavelength=fill(400, n))

target = make_pone_module(SA_F32[0., 0., 10.], UInt16(1))
medium = make_cascadia_medium_properties(0.95f0)
source = PointlikeIsotropicEmitter(SA_F32[0., 0., 0.], 0f0, 10000)
spectrum = Monochromatic(4250f0)

seed = 1

# Setup propagation
setup = PhotonPropSetup([source], [target], medium, spectrum, seed)

@profview make_hits_from_photons(photons, setup, RotMatrix3(I))