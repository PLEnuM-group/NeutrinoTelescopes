using PhotonPropagation
using NeutrinoTelescopes
using PhysicsTools
using StaticArrays
using LinearAlgebra
using Random
using Flux
using BSON: @load
using DataFrames
using CairoMakie


workdir = ENV["ECAPSTOR"]

model = gpu(PhotonSurrogate(
    joinpath(workdir, "snakemake/time_surrogate/extended/amplitude_2_FNL.bson"),
    joinpath(workdir, "snakemake/time_surrogate/extended/time_uncert_0_1_FNL.bson")
))


pos = SA[-15.0, 0., 5.]
dir_theta = 0.6
dir_phi = 0.3
dir = sph_to_cart(dir_theta, dir_phi)
energy = 7e4
particle = Particle(pos, dir, 0.0, energy, 0.0, PEMinus)

rng = MersenneTwister(31338)


target = POM(SA_F32[0., 0., 0.], 1)

feat_buffer = create_input_buffer(model, 16, 1)
device = gpu



times = -10:0.1:100

abs_scales = 0.9:0.01:1.1
sca_scale = 1.
fig = Figure()
ax = Axis(fig[1, 1])
input = @view feat_buffer[:, 1:16]


for abs_scale in abs_scales
    log_expec_per_pmt, log_expec_per_src_pmt_rs = get_log_amplitudes(
        [particle], [target], model; feat_buffer=feat_buffer, device=device, abs_scale=abs_scale)
    @show log_expec_per_pmt
    calc_flow_input!([particle], [target], model.time_transformations, feat_buffer, abs_scale=abs_scale, sca_scale=sca_scale)
    @show input[:, 1]
    input[9, :] .= abs_scale
    input[10, :] .= sca_scale
    flow_params = cpu(model.time_model.embedding(device(input)))


    pmt_ix = 6

    pdf_eval = eval_transformed_normal_logpdf(
                        times,
                        flow_params[:, pmt_ix],
                        model.time_model.range_min,
                        model.time_model.range_max
                    )

    lines!(ax, times, exp.(pdf_eval .+ log_expec_per_pmt[pmt_ix, 1, 1]))
end

fig
log_expec_per_pmt