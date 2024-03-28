using NeutrinoTelescopes
using PhotonPropagation
using PhysicsTools
using Flux
using CUDA
using Random
using StaticArrays
using BSON: @save, @load
using BSON
using CairoMakie
using Rotations
using LinearAlgebra
using DataFrames
using BSON
using CairoMakie
using Rotations
using LinearAlgebra
using DataFrames
using PreallocationTools
using StatsBase
using JLD2
using Glob
using Base.Iterators


workdir = ENV["ECAPSTOR"]

model = PhotonSurrogate(
    joinpath(workdir, "snakemake/time_surrogate_perturb/extended/amplitude_1_FNL.bson"),
    joinpath(workdir, "snakemake/time_surrogate_perturb/extended/time_uncert_0_1_FNL.bson")
)

model = gpu(model)

pos = SA[0.0f0, 20.0f0, -500]
dir_theta = deg2rad(20f0)
dir_phi = deg2rad(50f0)
dir = sph_to_cart(dir_theta, dir_phi)


targets_hex = make_hex_detector(3, 50, 20, 50, truncate=1)
d = LineDetector(targets_hex, medium)

p = Particle(pos, dir, 0.0f0, Float32(1E5), Float32(1E4), PEPlus)

n_lines_max = 50

input_buffer = create_input_buffer(model, 20*16*n_lines_max, 1)
output_buffer = create_output_buffer(20*16*n_lines_max, 100)
diff_cache = FixedSizeDiffCache(input_buffer, 6)
hit_generator = SurrogateModelHitGenerator(model, 200.0, input_buffer, output_buffer)

medium = make_cascadia_medium_properties(0.95f0)



abs_scale = 1.
sca_scale = 1.
zeniths = rad2deg(dir_theta)-1:0.1:rad2deg(dir_theta)+1

fig = Figure()
ax = Axis(fig[1, 1])
#for nr in [0, 1E3, 1E4, 1E5, 1E6]
abs_scales = [0.95, 0.97, 1, 1.02, 1.05]
for abs_scale in abs_scales
    for i in 1:50
        hits, mask = generate_hit_times([p], d, hit_generator, noise_rate=1E4, abs_scale=abs_scale, sca_scale=sca_scale)
        all_lhs = Vector{Vector{Float64}}(undef, 0)
        lhs = Vector{Float64}(undef, 0)
        for zen in zeniths

            dir = sph_to_cart(Float32(deg2rad(zen)), dir_phi)
            p_reco = Particle(pos, dir, 0.0f0, Float32(1E5), Float32(1E4), PEPlus)
            push!(lhs, multi_particle_likelihood(
                [p_reco],
                data=hits,
                targets=get_detector_modules(d)[mask],
                model=model,
                medium=medium,
                feat_buffer=input_buffer,
                abs_scale=abs_scale,
                sca_scale=sca_scale,
                noise_rate=1E4)
            )
        end
        push!(all_lhs, lhs)
    end

    all_lhs = mean(reduce(hcat, all_lhs), dims=2)[:, 1]
    max_scan = argmax(all_lhs)
    dllh = -2*(all_lhs .- all_lhs[max_scan])

    lines!((zeniths), dllh, label="$(abs_scale)")
end
axislegend("Absorption Scale")
vlines!(ax, rad2deg(dir_theta))
fig

model = PhotonSurrogate(
joinpath(workdir, "snakemake/time_surrogate_perturb/extended/amplitude_1_FNL.bson"),
joinpath(workdir, "snakemake/time_surrogate_perturb/extended/time_uncert_0_1_FNL.bson")
)

model = gpu(model)

input_buffer = create_input_buffer(model, 16, 1)
output_buffer = create_output_buffer(16, 100)
diff_cache = FixedSizeDiffCache(input_buffer, 6)
hit_generator = SurrogateModelHitGenerator(model, 200.0, input_buffer, output_buffer)
medium = make_cascadia_medium_properties(0.95f0)

validation_data = jldopen("/home/wecapstor3/capn/capn100h/surrogate_validation_sets/cascades.jdl2")
target = POM(SA[0., 0., 0.], 1)

d = UnstructuredDetector([target], medium)

models_amp = glob("amplitude_*_FNL.bson", joinpath(workdir, "snakemake/time_surrogate_perturb/extended/"), )
models_time = glob("time_uncert_0*_FNL.bson", joinpath(workdir, "snakemake/time_surrogate_perturb/extended/"), )

p_values = DataFrame()

for key in keys(validation_data)
    hits_mc = validation_data[key]["hits"]
    particle = validation_data[key]["particle"]
    abs_scale = validation_data[key]["abs_scale"]
    sca_scale = validation_data[key]["sca_scale"]

    ts_mc = Float64[]
    for i in 1:100
        hits = dataframe_to_hit_list(resample_hits(hits_mc, true), [target])

        push!(ts_mc, multi_particle_likelihood(
            [particle],
            data=hits,
            targets=get_detector_modules(d),
            model=model,
            medium=medium,
            feat_buffer=input_buffer,
            abs_scale=abs_scale,
            sca_scale=sca_scale,
            noise_rate=0)
        )
    end

    for (mamp, mtime) in product(models_amp, models_time)
        model = gpu(PhotonSurrogate(mamp, mtime))
        ts_same = Float64[]
        for i in 1:500
            hits, mask = generate_hit_times([particle], d, hit_generator, noise_rate=0, abs_scale=abs_scale, sca_scale=sca_scale)
            push!(ts_same, multi_particle_likelihood(
                        [particle],
                        data=hits,
                        targets=get_detector_modules(d)[mask],
                        model=model,
                        medium=medium,
                        feat_buffer=input_buffer,
                        abs_scale=abs_scale,
                        sca_scale=sca_scale,
                        noise_rate=0)
            )
        end    
        p_value = 1-percentile_of_score(ts_same, median(ts_mc))
        push!(p_values, (p_value=p_value, amp=parse(Int64, mamp[end-9]), time=parse(Int64, mtime[end-9])))
    end
end

p_values

hist(p_values)

mean(p_values)

hist(.-log10.(p_values), bins=0:0.1:3)