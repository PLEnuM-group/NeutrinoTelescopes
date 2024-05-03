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
using HypothesisTests
using Glob
using UMAP
using LinearAlgebra
using MultivariateStats
workdir = "/home/wecapstor3/capn/capn100h/"#ENV["ECAPSTOR"]

type = "extended"

model = PhotonSurrogate(
joinpath(workdir, "snakemake/time_surrogate_perturb/$type/amplitude_1_FNL.bson"),
joinpath(workdir, "snakemake/time_surrogate_perturb/$type/time_uncert_0_1_FNL.bson")
)

input_buffer = create_input_buffer(model, 16, 1)
output_buffer = create_output_buffer(16, 100)
diff_cache = FixedSizeDiffCache(input_buffer, 6)

medium = make_cascadia_medium_properties(0.95f0)

type == "lightsabre" ?  "tracks" : "cascades"
target = POM(SA[0., 0., 0.], 1)

d = UnstructuredDetector([target], medium)

models_amp = glob("amplitude_*_FNL.bson", joinpath(workdir, "snakemake/time_surrogate_perturb/$type/"), )
models_time = glob("time_uncert_0*_FNL.bson", joinpath(workdir, "snakemake/time_surrogate_perturb/$type/"), )

validation_sets = glob("*$type*", "/home/wecapstor3/capn/capn100h/snakemake/surrogate_validation_sets")
rex = r"set_([0-9]+)_"

likelihoods = DataFrame()

for vset in validation_sets[1:2]
    validation_data = jldopen(vset)
    seed = match(rex, vset)[1]

    for (mamp, mtime) in product(models_amp, models_time)
                
        model = gpu(PhotonSurrogate(mamp, mtime))
        hit_generator = SurrogateModelHitGenerator(model, 200.0, input_buffer, output_buffer)

        for key in keys(validation_data)
            hits_mc = validation_data[key]["hits"]
            particle = validation_data[key]["particle"]
            abs_scale = validation_data[key]["abs_scale"]
            sca_scale = validation_data[key]["sca_scale"]
            
            ts_mc = Float64[]
            ts_same = Float64[]

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
                
            for i in 1:100
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
            pval_ks = ApproximateTwoSampleKSTest(ts_same, ts_mc)

            amp = parse(Int64, mamp[end-9])
            time = parse(Int64, mtime[end-9])

            dir_sph = cart_to_sph(particle.direction)
            dist = norm(particle.position)
            pos_sph = cart_to_sph(particle.position ./ dist)

            push!(
                likelihoods,
                (
                    event=key,
                    p_value=p_value,
                    p_value_ks=pval_ks,
                    ts_model=ts_same,
                    ts_mc=ts_mc,
                    time=time,
                    amp=amp,
                    dset=seed,
                    particle_dir_theta=dir_sph[1],
                    particle_dir_phi=dir_sph[2],
                    particle_pos_theta=pos_sph[1],
                    particle_pos_phi=pos_sph[2],
                    particle_log10_energy=log10(particle.energy),
                    particle_log10_dist=log10(dist),
                    abs_scale=abs_scale,
                    sca_scale=sca_scale,
                    ))
        end
    end
end

jldsave(joinpath(workdir, "surrogate_validation.jld2"), data=likelihoods)
key = "dataset_1"
dset = "1"


mask = likelihoods.event .== key .&& likelihoods.dset .== dset .&& likelihoods.time .== 1 .&& likelihoods.time .== 1 .&& likelihoods.amp .== 1
mc_ts = first(likelihoods[mask, :ts_mc])
model_ts = first(likelihoods[mask, :ts_model])


fig = Figure()
ax = Axis(fig[1, 1])
edges = extrema(vcat(mc_ts..., model_ts...))
bins = LinRange(edges[1]-0.01, edges[2]+0.01, 20)
hist!(ax, mc_ts, bins=bins, normalization=:pdf)
hist!(ax, model_ts, bins=bins, normalization=:pdf)
fig

hist(mc_ts, normalization=:pdf)



likelihoods = jldopen(joinpath(workdir, "surrogate_validation.jld2"))["data"]
likelihoods[!, :p_value_ks_p] .= pvalue.(likelihoods[:, :p_value_ks] )


combine(groupby(likelihoods, [:amp, :time]), :p_value_ks_p => median)

hist(likelihoods[likelihoods.amp .== 3 .&& likelihoods.time .== 2, :p_value_ks_p])
dir_sph = cart_to_sph.(likelihoods.particle_dir)


model_mask = likelihoods.time .== 1 .&& likelihoods.amp .== 3
dataset = likelihoods[model_mask, :]

scatter(dataset.particle_log10_energy, log10.(dataset.p_value_ks_p))
scatter(dataset.particle_log10_dist, log10.(dataset.p_value_ks_p))
scatter(dataset.particle_dir_theta, log10.(dataset.p_value_ks_p))
scatter(dataset.particle_dir_phi, log10.(dataset.p_value_ks_p))
scatter(dataset.particle_pos_phi, log10.(dataset.p_value_ks_p))
scatter(dataset.particle_pos_theta, log10.(dataset.p_value_ks_p))

scatter(dataset.sca_scale, log10.(pvalue.(dataset.p_value_ks)))


features = dataset[:, [:particle_log10_energy, :particle_log10_dist, :particle_dir_theta, :particle_dir_phi, :particle_pos_phi, :particle_pos_theta, :abs_scale, :sca_scale]]


cor(Matrix(features), dataset.p_value_ks_p)

embedding = umap(likelii, n_components; n_neighbors, metric, min_dist, ...)

