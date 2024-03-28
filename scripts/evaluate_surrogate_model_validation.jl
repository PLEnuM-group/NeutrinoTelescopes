using NeutrinoTelescopes
using PhotonPropagation
using PhysicsTools
using Random
using StaticArrays
using CairoMakie
using DataFrames
using StatsBase
using JLD2
using Glob
using Flux
using PreallocationTools
using Base.Iterators

workdir = "/home/wecapstor3/capn/capn100h/"#ENV["ECAPSTOR"]

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
jldsave(joinpath(workdir, "surrogate_validation.jld2"), data=p_values)

data = jldopen(joinpath(workdir, "surrogate_validation.jld2"))["data"]


data

combine(groupby(data, [:amp, :time]), :p_value => median)