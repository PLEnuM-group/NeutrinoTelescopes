using NeutrinoTelescopes
using PhotonPropagation
using PhysicsTools
using Flux
using CUDA
using PreallocationTools
using BenchmarkTools
using Profile
using LinearAlgebra
using Random

model = PhotonSurrogate(
    "/home/saturn/capn/capn100h/snakemake/time_surrogate/extended/amplitude_1_FNL.bson",
    "/home/saturn/capn/capn100h/snakemake/time_surrogate/extended/time_uncert_2.5_1_FNL.bson")

model = gpu(model)

medium = make_cascadia_medium_properties(0.95f0)

args = Dict("nevents" => 5, "vert_spacing" => 28.57, "spacing" => 120, "li-file" =>  "/home/saturn/capn/capn100h/snakemake/leptoninjector-extended-1.hd5", "det" => "full", "type" => "extended")

n_samples = 50
n_events = args["nevents"]

results = []

targets = nothing
n_gaps = floor(Int64, 1000 / args["vert_spacing"])
n_modules = n_gaps + 1 
zstart = 0.5* n_gaps * args["vert_spacing"]

## HACK FOR P-ONE-1 lines
if args["vert_spacing"] == 50
    n_modules = 20
    zstart = 475
end

if args["det"] == "cluster"
    targets = make_hex_detector(3, args["spacing"], n_modules, args["vert_spacing"], truncate=1, z_start=zstart)
else
    targets = make_n_hex_cluster_detector(7, args["spacing"], n_modules, args["vert_spacing"], z_start=zstart)
end

d = Detector(targets, medium)
hit_buffer = create_input_buffer(d, 1)
output_buffer = create_output_buffer(d, 100)
diff_cache = FixedSizeDiffCache(hit_buffer, 6)
cylinder = get_bounding_cylinder(d)
inj = LIInjector(args["li-file"], drop_starting=(args["type"] == "lightsabre"), volume=cylinder)
hit_generator = SurrogateModelHitGenerator(model, 200.0, hit_buffer, output_buffer)

event = rand(inj)

m, evts = calc_fisher(d, inj, hit_generator, n_events, n_samples, use_grad=true, cache=diff_cache)



Profile.clear_malloc_data()
m, evts = calc_fisher(d, inj, hit_generator, n_events, n_samples, use_grad=true, cache=diff_cache)

#calc_fisher(d, inj, hit_generator, n_events, n_samples, use_grad=true, cache=diff_cache)

#=
model_cpu = cpu(model)
hit_generator_cpu = SurrogateModelHitGenerator(model_cpu, 200.0, hit_buffer)

m, evts = calc_fisher(d, inj, hit_generator_cpu, n_events, n_samples, use_grad=true, cache=diff_cache, device=cpu)
@benchmark m, evts = calc_fisher(d, inj, hit_generator_cpu, n_events, n_samples, use_grad=true, cache=diff_cache, device=cpu)=#