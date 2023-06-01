using PhotonPropagation
using NeutrinoTelescopes
using PhysicsTools
using CSV
using StaticArrays
using DataFrames
using Interpolations
using GeoMakie
using Rotations
using LinearAlgebra
using Distributions
using StatsBase


# Setup target
targ_pos = SA_F64[0., 5., 20.]
pmt_area = (75e-3 / 2)^2 * π
target_radius = 0.21

detshape = Spherical(Float32.(targ_pos), Float32(target_radius))

PROJECT_ROOT = pkgdir(PhotonPropagation)
df = CSV.read(joinpath(PROJECT_ROOT, "assets/PMTAcc.csv"), DataFrame, header=["wavelength", "acceptance"])
acc_pmt_wl = linear_interpolation(df[:, :wavelength], df[:, :acceptance], extrapolation_bc=0.)


target = SphericalMultiPMTDetector(
    detshape,
    pmt_area,
    make_pom_pmt_coordinates(Float64),
    acc_pmt_wl,
    UInt16(1)
)

target2 = make_pone_module(targ_pos, UInt16(1))

# Setup source
position = SA_F32[0., 0., 0.]
energy = Float32(5E4)

dir_theta = deg2rad(20)
dir_phi = deg2rad(120)

direction = SVector{3, Float32}(sph_to_cart(dir_theta, dir_phi))
p = Particle(position, direction, 0f0, energy, 0f0, PEMinus)

# Setup medium
mean_sca_angle = 0.99f0
medium = make_cascadia_medium_properties(mean_sca_angle)

# Wavelength range for Cherenkov emission
wl_range = (200f0, 800f0)
source = ExtendedCherenkovEmitter(p, medium, wl_range)

spectrum = CherenkovSpectrum(wl_range, medium)

seed = 1

# Setup propagation
setup = PhotonPropSetup([source], [target], medium, spectrum, seed)
setup2 = PhotonPropSetup([source], [target2], medium, spectrum, seed)

# Run propagation
photons = propagate_photons(setup)
photons2 = propagate_photons(setup2)

hits = make_hits_from_photons(photons, setup)
hits2 = make_hits_from_photons(photons2, setup2)


function positions_to_lonlat(hits, target)
    pos = hits[hits[:, :pmt_id] .== 5, :position]
    #pos = hits[:, :position]
    pos_rel_sph = reduce(hcat, cart_to_sph.((pos .- Ref(target.shape.position)) ./ target.shape.radius))
    longitude = rad2deg.(pos_rel_sph[2, :]) .- 180
    latitude =  rad2deg.(pos_rel_sph[1, :]) .- 90

    return longitude, latitude
end


combine(groupby(hits, :pmt_id), nrow)
combine(groupby(hits2, :pmt_id), nrow)


fig = Figure()
ax = GeoAxis(fig[1,1]; dest = "+proj=moll")
lon, lat =positions_to_lonlat(hits, target)
scatter!(ax, lon, lat, color=(:blue, 0.1))
lon, lat =positions_to_lonlat(hits2, target2)
scatter!(ax, lon, lat, color=(:orange, 0.1))
fig
