using HDF5
using DataFrames
using CairoMakie
using Distributions
using PhysicsTools
using LinearAlgebra
using PairPlots
using Polynomials

fname = "/home/wecapstor3/capn/capn100h/snakemake/photon_tables/extended/hits/photon_table_hits_extended_dmin_1_dmax_200_emin_100000.0_emax_5000000.0_0.hd5"

fname_photons = "/home/wecapstor3/capn/capn100h/snakemake/photon_tables/extended/photon_table_extended_dmin_1_dmax_200_emin_100000.0_emax_5000000.0_0.hd5"

fid = h5open(fname)


length(fid["/pmt_hits"])

grp_keys = keys(fid["/pmt_hits"])




gamma_params = []
for grp_k in grp_keys
    grp = fid["/pmt_hits/$grp_k"]
    grp_attrs = attrs(grp)
    df = DataFrame(grp[:, :], [:time, :pmt_id, :weights])
    
    offset = minimum(df[:, :time]) - 1E-10
    
    dist = fit_mle(Gamma, df[:, :time] .- offset, df[:, :weights])

    

    distance = grp_attrs["distance"]
    pos_theta = grp_attrs["pos_theta"]
    pos_phi = grp_attrs["pos_phi"]

    pos_cart = sph_to_cart(pos_theta, pos_phi)

    dir_theta = grp_attrs["dir_theta"]
    dir_phi = grp_attrs["dir_phi"]

    abs = grp_attrs["abs_scale"]
    sca = grp_attrs["sca_scale"]

    dir_cart = sph_to_cart(dir_theta, dir_phi)
    
    norm_vec = cross(pos_cart, dir_cart)
    ntheta, nphi = cart_to_sph(norm_vec)
   
    rel_ang = acos(dot(pos_cart, dir_cart))


    push!(gamma_params, 
        (
            rel_ang=rel_ang,
            logdistance=log10(distance),
            α=dist.α,
            logθ=log10(dist.θ),
            offset=offset,
            pos_theta=pos_theta,
            pos_phi=pos_phi,
            dir_theta=dir_theta,
            dir_phi=dir_phi,
            abs=abs,
            sca=sca,
            ntheta=ntheta,
            nphi=nphi
        )
    )
end

gamma_params = DataFrame(gamma_params)

fig = Figure()
ax = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])

mask = gamma_params[:, :rel_ang] .> π/2
scatter!(ax, (gamma_params[:, :logdistance]), (gamma_params[:, :logθ]), color=mask)
scatter!(ax2, (gamma_params[:, :logdistance]), log10.(gamma_params[:, :α]), color=mask)
fig


Polynomials.fit(10 .^(gamma_params[:, :logdistance]), 10 .^((gamma_params[:, :logθ])), 1)





fig = Figure()
ax = Axis(fig[1, 1])
ax2 = Axis(fig[1, 2])

scatter!(ax, (gamma_params[:, :logdistance]), (gamma_params[:, :θ]), color=gamma_params[:, :ntheta])
scatter!(ax2, (gamma_params[:, :logdistance]), log10.(gamma_params[:, :α]), color=gamma_params[:, :ntheta])
fig


mask = gamma_params[:, :distance] .> 50 .&& gamma_params[:, :distance] .< 60

gpsel = gamma_params[mask, :]

X = Matrix(gamma_params[:, [:θ, :α]])

embedding = umap(X', 2)

scatter(embedding, color=gamma_params[:, :rel_ang])


pairplot(gamma_params[mask, :], gamma_params[.!mask, :])








gamma_params

plot(Gamma(22, 2))

grp = fid["/pmt_hits/dataset_10002"]
grp_attrs = attrs(grp)

df_sel = df

fig, ax, _ = hist(df_sel[:, :time], weights=df_sel[:, :weights], normalization=:pdf,
bins=0:1:100)



plot!(ax, dist)
fig



