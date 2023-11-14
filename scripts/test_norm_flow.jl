using NeutrinoTelescopes
using Random
using CUDA
using Flux
using PhotonPropagation
using PhysicsTools
using DataStructures
using Distributions
using CairoMakie
using ArraysOfArrays


num_bins = 10
x = -5:0.1:5
params = randn(3 * num_bins + 1,  length(x))
x_pos, y_pos, knot_slopes = constrain_spline_params(params, -9.5, 9.5)
y, logdet = rqs_univariate(x_pos, y_pos, knot_slopes, x)
xrt, logdet_inv = inv_rqs_univariate(x_pos, y_pos, knot_slopes, y)

y, logdet = rqs_univariate(x_pos[:, 1], y_pos[:, 1], knot_slopes[:, 1], x)
xrt, logdet_inv = inv_rqs_univariate(x_pos[:, 1], y_pos[:, 1], knot_slopes[:, 1], y)


all(logdet .≈ -logdet_inv)
all(isapprox.(x, xrt; atol=1E-5))

x = CuArray(x)
params = CuMatrix(params)
x_pos, y_pos, knot_slopes = constrain_spline_params(params, -5, 5)
y, logdet = rqs_univariate(x_pos, y_pos, knot_slopes, x)

xrt, logdet_inv = inv_rqs_univariate(x_pos, y_pos, knot_slopes, y)

all(Vector(logdet) .≈ -Vector(logdet_inv))
all(isapprox.(Vector(x), Vector(xrt); atol=1E-5))


model = PhotonSurrogate(
    "/home/saturn/capn/capn100h/snakemake/time_surrogate/extended/amplitude_1_FNL.bson",
    "/home/saturn/capn/capn100h/snakemake/time_surrogate/extended/time_uncert_2.5_1_FNL.bson")

model = gpu(model)

medium = make_cascadia_medium_properties(0.95f0)
targets = make_n_hex_cluster_detector(7, 50, 20, 50, z_start=475)

d = Detector(targets, medium)
feat_buffer = create_input_buffer(d, 100)
output_buffer = create_output_buffer(d, 100)
cylinder = get_bounding_cylinder(d)
pdist = CategoricalSetDistribution(OrderedSet([PEMinus, PEPlus]), [0.5, 0.5])
edist = Dirac(1E4)
ang_dist = LowerHalfSphere()
length_dist = Dirac(0.0)
time_dist = Dirac(0.0)
inj = VolumeInjector(cylinder, edist, pdist, ang_dist, length_dist, time_dist)

hit_generator = SurrogateModelHitGenerator(model, 200.0, feat_buffer, output_buffer)
for _ in 1:1000
    event = rand(inj)
    generate_hit_times(event, d, hit_generator)
end



particles_vec = [rand(inj)[:particles] for _ in 1:100]

particles_vec_flat = reduce(vcat, particles_vec)


flow_params, n_hits_per_source_rs = ExtendedCascadeModel._prepare_flow_inputs(particles_vec_flat, targets, model, feat_buffer, gpu, Random.default_rng(), oversample=1)

times, n_hits_per_pmt_source = ExtendedCascadeModel._sample_times_for_particle(event[:particles], targets, model, output_buffer, Random.default_rng(), oversample=1, feat_buffer=feat_buffer, device=gpu)
empty!(output_buffer)

times = sample_multi_particle_event(event[:particles], targets, model, medium; feat_buffer=feat_buffer, output_buffer=output_buffer)