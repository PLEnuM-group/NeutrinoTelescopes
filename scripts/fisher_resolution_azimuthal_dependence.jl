using NeutrinoTelescopes
using PhotonPropagation
using PhotonSurrogateModel
using NeutrinoSurrogateModelData
using PhysicsTools
using Flux
using CUDA
using Random
using StaticArrays
using CairoMakie
using Rotations
using LinearAlgebra
using DataFrames
using StatsBase
using JLD2
using Base.Iterators
using DataStructures
using PreallocationTools
using Rotations
using Distributions

workdir = ENV["ECAPSTOR"]
medium = make_cascadia_medium_properties(0.95f0)
model = PhotonSurrogate(em_cascade_time_model(2)...)
model = gpu(model)

targets = make_detector_line([0., 0., 0.], 20, 50)
d = LineDetector([targets], medium)


medium = make_cascadia_medium_properties(0.95)

rad2deg(cherenkov_angle(400., medium))

input_buffer = create_input_buffer(model, d, 1)
output_buffer = create_output_buffer(d, 100)
diff_cache = DiffCache(input_buffer, 13)
hit_generator = SurrogateModelHitGenerator(model, 200.0, input_buffer, output_buffer)

event = Event()


dir = sph_to_cart(0.7, 4.5)


pos = [-10, 10, -400]

function get_directional_uncertainty(dir_sph, cov)
    dir_cart = sph_to_cart(dir_sph)

    dist = MvNormal(dir_sph, cov)
    rdirs = rand(dist, 100)

    dangles = rad2deg.(acos.(dot.(sph_to_cart.(rdirs[1, :], rdirs[2, :]), Ref(dir_cart))))
    return mean(dangles)
end

azis = 0:0.1:2*π
res = []
for azi in azis
    dir = sph_to_cart(π/2, azi)
    event[:particles] = [Particle(pos, dir, 0., 5E4, 0., PEMinus)]
    fm, _ = calc_fisher_matrix(event, d, hit_generator, cache=diff_cache, n_samples=100)

    cov = inv(fm)
    cov = 0.5 * (cov + cov')

    uncert = get_directional_uncertainty(cart_to_sph(dir), cov[2:3, 2:3])
    push!(res, (uncert, sqrt(cov[2,2]), sqrt(cov[3,3])))
end

fig, ax, _ = lines(azis, Float64.(first.(res)), axis=(ylabel="Angular Resolution [deg]", xlabel="Cascade Azimuth"))
vlines!(ax, deg2rad(135), color=:black)
vlines!(ax, deg2rad(135+90+42.25), color=:black, linestyle=:dash)
vlines!(ax, deg2rad(135+90+2*42.25), color=:black, linestyle=:dash)
fig



pos = SA[0.0f0, 20.0f0, -500]
dir_theta = deg2rad(20f0)
dir_phi = deg2rad(50f0)
dir = sph_to_cart(dir_theta, dir_phi)





p = Particle(pos, dir, 0.0f0, Float32(1E5), Float32(1E4), PEPlus)

n_lines_max = 50





cylinder = NeutrinoTelescopes.Cylinder(SA[0., 0., -475.], 1100., radius)
