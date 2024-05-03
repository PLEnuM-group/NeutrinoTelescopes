using NeutrinoTelescopes
using PhotonPropagation
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

workdir = ENV["ECAPSTOR"]
medium = make_cascadia_medium_properties(0.95f0)
model = PhotonSurrogate(
    joinpath(workdir, "snakemake/time_surrogate_perturb/extended/amplitude_1_FNL.bson"),
    joinpath(workdir, "snakemake/time_surrogate_perturb/extended/time_uncert_0_2_FNL.bson")
)

model = gpu(model)

targets_hex = make_hex_detector(3, 50, 20, 50, truncate=1)
d = LineDetector(targets_hex, medium)

cylinder = get_bounding_cylinder(d)
surf = CylinderSurface(cylinder)
pdist = CategoricalSetDistribution(OrderedSet([PMuPlus, PMuMinus]), [0.5, 0.5])
edist = Pareto(1, 1E4)
ang_dist = LowerHalfSphere()
length_dist = Dirac(1E4)
time_dist = Dirac(0.0)
inj = SurfaceInjector(surf, edist, pdist, ang_dist, length_dist, time_dist)

input_buffer = create_input_buffer(model, 20*16*10, 1)
output_buffer = create_output_buffer(20*16*10, 100)
diff_cache = FixedSizeDiffCache(input_buffer, 6)
hit_generator = SurrogateModelHitGenerator(model, 200.0, input_buffer, output_buffer)

calc_fisher(d, inj, hit_generator)



pos = SA[0.0f0, 20.0f0, -500]
dir_theta = deg2rad(20f0)
dir_phi = deg2rad(50f0)
dir = sph_to_cart(dir_theta, dir_phi)





p = Particle(pos, dir, 0.0f0, Float32(1E5), Float32(1E4), PEPlus)

n_lines_max = 50





cylinder = NeutrinoTelescopes.Cylinder(SA[0., 0., -475.], 1100., radius)
