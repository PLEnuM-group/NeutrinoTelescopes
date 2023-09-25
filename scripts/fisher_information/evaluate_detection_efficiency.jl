using NeutrinoTelescopes
using PhotonPropagation
using PhysicsTools
using CairoMakie
using DataFrames
using StaticArrays
using Formatting
using StatsBase
using ColorSchemes
using JLD2
using LinearAlgebra
using Polyhedra
import GLPK
using Distributions
using CSV
using Glob
using HDF5

function get_polyhedron(spacing)
    targets = make_n_hex_cluster_detector(7, spacing, 20, 50)
    posv = [t.shape.position for t in targets]
    v = vrep(posv)
    v = removevredundancy(v, GLPK.Optimizer)
    p = polyhedron(v)
    return p
end

function orthonormal_basis(direction)
    if direction[3] > 0
        u = 1.
        v = 1.
        w = - (u*direction[1] + v*direction[2]) / direction[3]
        x1 = [u, v, w]
        x1 = x1 ./ norm(x1)
        x2 = cross(x1, direction)
        x2 = x2 ./ norm(x2)
    else
        # vec in in xy-plane
        x1 = [0., 0., 1.]
        x2 = cross(x1, direction)
        x2 = x2 ./ norm(x2)
    end
    return x1, x2
end


function get_geo_proj_area(p, direction)
    basis = hcat(orthonormal_basis(direction)...)
    p2 = project(p, basis)
    vol = 0.
    try
        vol = Polyhedra.volume(polyhedron(hrep(p2)))
    catch y
        @show p2, hrep(p2), direction
        vol = missing
    end
    return vol
end

function get_average_geo_proj_area(p)

    thetas = acos.(rand(Uniform(-1, 1), 1000))
    phis = rand(Uniform(0, 2*π), 1000)

    directions = sph_to_cart.(thetas, phis)

    return mean(skipmissing(get_geo_proj_area.(Ref(p), directions)))
end

function effective_area(above_thrsh, proj_area)
    eff_area = sum(proj_area[above_thrsh]) / length(above_thrsh)
    return eff_area
end

function neutrino_effective_area(above_thrsh, proj_area, total_prob)
    eff_area = sum(proj_area[above_thrsh] .* total_prob[above_thrsh]) / length(above_thrsh)
    return eff_area
end


function effective_volume(above_thrsh, eff_volume)
    return eff_volume[1] * sum(above_thrsh) / length(above_thrsh)
end

function _plot_effective_volume!(ax, df; n_mod_thresh=1, pmt_thresh=:n_mod_thrsh_two, add_colorbar=true)
    logenergies = unique(df[:, :log_energy])
    groups = groupby(df, :log_energy)
    is_above_thresh = df[:, pmt_thresh] .>= n_mod_thresh
    df[!, :is_above_thresh] = is_above_thresh

    cmap = cgrad(:viridis, length(groups)-1, categorical = true)

    for (groupn, group) in pairs(groups)
        eff = combine(groupby(group, :spacing), [:is_above_thresh, :sim_volume] => effective_volume => :effective_volume)
        sort!(eff, :spacing)

        mask = eff[:, :effective_volume] .!= 0
        x_vals = eff[mask, :spacing]
        y_vals = eff[mask, :effective_volume] ./ 1E9

        lines!(ax, x_vals, y_vals, label=format("{:.0d} TeV", round(10. ^groupn[1] / 1000)),
            color=groupn[1], colorrange=(minimum(logenergies), maximum(logenergies)), colormap=cmap)
    end

    spacings = sort(unique(df[:, :spacing]))
    volumes = Polyhedra.volume.(get_polyhedron.(spacings)) ./ 1E9
    lines!(ax, spacings, volumes, color=:black, linestyle=:dot)

    sim_vol = combine(groupby(df, :spacing), :sim_volume => first => :sim_volume)
    #lines!(sim_vol[:, :spacing], sim_vol[:, :sim_volume], color=:black, lw=3)

    return ax




end

function plot_effective_volume(df; n_mod_thresh=1, pmt_thresh=:n_mod_thrsh_two, add_colorbar=true)

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Spacing (m)", ylabel="Cascade Effective Volume (km³)",
              yscale=log10
              )

    _plot_effective_volume!(ax, df, n_mod_thresh=n_mod_thresh, pmt_thresh=pmt_thresh)
   
    tick_labels = Dict(3 => "1 TeV", 4 => "10 TeV", 5=> "100 TeV", 6 => "1 PeV")

    if add_colorbar
        Colorbar(fig[1, 2], limits = (minimum(logenergies), maximum(logenergies)), colormap = cmap,
            flipaxis = false, ticks=[3, 4, 5, 6], tickformat= xs -> [tick_labels[x] for x in xs])
    end

    ylims!(ax, 1E-2, 4)
    fig
end


function _plot_effective_area!(ax, df; n_mod_thresh=1, pmt_thresh=:n_mod_thrsh_two)
    
    logenergies = unique(df[:, :log_energy])
    groups = groupby(df, :log_energy)

    is_above_thresh = df[:, pmt_thresh] .>= n_mod_thresh
    df[!, :is_above_thresh] = is_above_thresh

    cmap = cgrad(:viridis, length(groups)-1, categorical = true)

    for (groupn, group) in pairs(groups)
        eff = combine(groupby(group, :spacing), [:is_above_thresh, :proj_area] => effective_area => :effective_area)
        sort!(eff, :spacing)

        mask = eff[:, :effective_area] .!= 0
        x_vals = eff[mask, :spacing]
        y_vals = eff[mask, :effective_area] ./ 1E6

        lines!(ax, x_vals, y_vals, label=format("{:.0d} TeV", round(10. ^groupn[1] / 1000)),
            color=groupn[1], colorrange=(minimum(logenergies), maximum(logenergies)), colormap=cmap)
    end

    #mean_proj = sort!(combine(groupby(df, :spacing), :proj_area => mean), :spacing)

    spacings = sort(unique(df[:, :spacing]))
    
    ps = get_polyhedron.(spacings)
    areas = get_average_geo_proj_area.(ps) ./ 1E6
    lines!(ax, spacings, areas, color=:black, linestyle=:dot)


    #lines!(ax, mean_proj[:, :spacing], mean_proj[:, :proj_area_mean] ./ 1E6, linestyle=:dot, color=:black)


    return ax
end

function plot_effective_area(df; n_mod_thresh=1, pmt_thresh=:n_mod_thrsh_two, add_colorbar=true)

    logenergies = unique(df[:, :log_energy])
    @show length(logenergies)
    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Spacing (m)", ylabel="Track Effective Area (km²)", yscale=log10)

    _plot_effective_area!(ax, df, n_mod_thresh=n_mod_thresh, pmt_thresh=pmt_thresh)

    tick_labels = Dict(3 => "1 TeV", 4 => "10 TeV", 5=> "100 TeV", 6 => "1 PeV")

    if add_colorbar
        cmap = cgrad(:viridis, length(logenergies)-1, categorical = true)
        Colorbar(fig[1, 2], limits = (minimum(logenergies), maximum(logenergies)), colormap = cmap,
            flipaxis = false, ticks=[3, 4, 5, 6], tickformat= xs -> [tick_labels[x] for x in xs])
    end

    ylims!(ax, 1E-2, 4)
    fig
end


function plot_effective_area_and_volume(df_track, df_casc; n_mod_thresh=1, pmt_thresh=:n_mod_thrsh_two, add_colorbar=true)

    fig = Figure(resolution=(1200, 600))
    ax = Axis(fig[1, 1], xlabel="Spacing (m)", ylabel="Track Effective Area (km²)", yscale=log10)
    ax2 = Axis(fig[1, 2], xlabel="Spacing (m)", ylabel="Cascade Effective Volume (km³)",
              yscale=log10
              )

    _plot_effective_area!(ax, df_track, n_mod_thresh=n_mod_thresh, pmt_thresh=pmt_thresh)
    _plot_effective_volume!(ax2, df_casc, n_mod_thresh=n_mod_thresh, pmt_thresh=pmt_thresh)

    tick_labels = Dict(3 => "1 TeV", 4 => "10 TeV", 5=> "100 TeV", 6 => "1 PeV")

    logenergies = unique(df_track[:, :log_energy])

    cmap = cgrad(:viridis, length(logenergies)-1, categorical = true)

    if add_colorbar
        Colorbar(fig[1, 3], limits = (minimum(logenergies), maximum(logenergies)), colormap = cmap,
            flipaxis = false, ticks=[3, 4, 5, 6], tickformat= xs -> [tick_labels[x] for x in xs])
    end

    ylims!(ax, 1E-2, 4)
    ylims!(ax2, 1E-2, 4)
    fig
end


function plot_max_eff_v_spacing(df)

    logenergies = unique(df[:, :log_energy])
    groups = groupby(df, :log_energy)
    
    max_vol_at = []

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Log10(Energy / GeV)", ylabel="Optimal Spacing (m)")

    for (groupn, group) in pairs(groups)
        eff = combine(groupby(group, :spacing), [:n_mod_thrsh, :sim_volume] => effective_volume => :effective_volume)
        
        max_r = argmax(eff[:, :effective_volume])
        push!(max_vol_at, [groupn[1], eff[max_r, :spacing]])
    end

    lines!(ax, reduce(hcat, max_vol_at))
    return fig
end


function _calc_neutrino_effective_area(df; n_mod_thresh=1, pmt_thresh=:n_mod_thrsh_two, cos_zenith_bins=[-1, 1])
    
    is_above_thresh = df[:, pmt_thresh] .>= n_mod_thresh
    df[!, :is_above_thresh] = is_above_thresh
    groups = groupby(df, :log_energy)
    dfs = []

    for (groupn, group) in pairs(groups)

        for zbix in 1:(length(cos_zenith_bins) - 1)
            ct = cos.(group[:, :dir_theta])
            lower = cos_zenith_bins[zbix]
            upper = cos_zenith_bins[zbix+ 1] 
            mask = (ct .>= lower) .&& (ct .< upper)
            grpmsk = group[mask, :]
            eff = combine(groupby(grpmsk, :spacing), [:is_above_thresh, :proj_area, :total_prob] => neutrino_effective_area => :effective_area)
            eff[!, :log10_energy] .= groupn[1]
            eff[!, :cos_theta] .= (upper+lower)/2
            push!(dfs, eff)
        end

    end
    
    return reduce(vcat, dfs)
end

function _plot_neutrino_effective_area!(ax, df; n_mod_thresh=1, pmt_thresh=:n_mod_thrsh_two)
    
    logenergies = unique(df[:, :log_energy])
    eff_a = _calc_neutrino_effective_area(df, n_mod_thresh=n_mod_thresh, pmt_thresh=pmt_thresh)
    groups = groupby(eff_a, :log10_energy)

    cmap = cgrad(:viridis, length(groups)-1, categorical = true)

    for (groupn, group) in pairs(groups)
      
        grp = sort(group, :spacing)

        mask = grp[:, :effective_area] .!= 0
        x_vals = grp[mask, :spacing]
        y_vals = grp[mask, :effective_area]

        lines!(ax, x_vals, y_vals, label=format("{:.0d} TeV", round(10. ^groupn[1] / 1000)),
            color=groupn[1], colorrange=(minimum(logenergies), maximum(logenergies)), colormap=cmap)
    end
    return ax
end


function plot_neutrino_effective_area(df; n_mod_thresh=1, pmt_thresh=:n_mod_thrsh_two, add_colorbar=true)

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Spacing (m)", ylabel="Neutrino Effective Area (m^2)", yscale=log10)

    _plot_neutrino_effective_area!(ax, df, n_mod_thresh=n_mod_thresh, pmt_thresh=pmt_thresh)

    tick_labels = Dict(3 => "1 TeV", 4 => "10 TeV", 5=> "100 TeV", 6 => "1 PeV")

    logenergies = unique(df[:, :log_energy])

    cmap = cgrad(:viridis, length(logenergies)-1, categorical = true)

    if add_colorbar
        Colorbar(fig[1, 2], limits = (minimum(logenergies), maximum(logenergies)), colormap = cmap,
            flipaxis = false, ticks=[3, 4, 5, 6], tickformat= xs -> [tick_labels[x] for x in xs])
    end

    ylims!(ax, 1E-2, 100)
    fig
end

PLOT_DIR = joinpath(pkgdir(NeutrinoTelescopes), "figures")
data_dir = joinpath(ENV["WORK"], "snakemake/fisher")


data_tracks = mapreduce(f ->load(f)["results"], vcat, glob("*det*extended*", data_dir)) 

pmt_thresh = :n_mod_thrsh_two
n_mod_thresh = 0

bins = 2:0.5:6
fig = Figure()
ax = Axis(fig[1, 1])
for (groupn, group) in pairs(groupby(data_tracks, :spacing))
    is_above_thresh = group[:, pmt_thresh] .>= n_mod_thresh


    w = group[is_above_thresh, :weight]
    e = group[is_above_thresh, :log_energy]

    h = zeros(length(bins)-1)
    ixs = searchsortedfirst.(Ref(bins), e) .- 1

    for bix in eachindex(h)
        mask = bix .== ixs
        h[bix] += sum(w[mask])
    end

    h ./= diff(10 .^bins) * 1E4 * 4 * π

    #h ./= diff(10 .^bins)

    #hist!(ax, group[is_above_thresh, :log_energy], weights=group[is_above_thresh, :weight], bins=bins)
    stairs!(ax, bins, push!(h, h[end]), step=:post)
    #lines!(ax, group[:, :log_energy], group[:, :weight])
end
fig


fid=  h5open("/home/saturn/capn/capn100h/snakemake/leptoninjector-lightsabre-0.hd5")
weights = fid["RangedInjector0"]["weights"][:]
energies = [r[:Energy] for r in fid["RangedInjector0"]["initial"][:]]

inj = LIInjector(
    "/home/saturn/capn/capn100h/snakemake/leptoninjector-lightsabre-0.hd5",
    drop_starting=true,
    volume=Cylinder(SA[])
)

make_hex_detector()




h = zeros(length(bins)-1)
ixs = searchsortedfirst.(Ref(bins), log10.(energies)) .- 1

for bix in eachindex(h)
    mask = bix .== ixs
    h[bix] += sum(weights[mask])
end
h ./= diff(10 .^bins) * 1E4 * 4 * π
push!(h, h[end])

stairs(bins, h, step=:post, axis=(yscale=log10, ))




data_tracks = load(joinpath(data_dir, "det_eff_track_full.jld2"), "results")
data_cascades = load(joinpath(data_dir, "det_eff_cascade_full.jld2"), "results")

poster_theme = Theme(
        fontsize = 30, linewidth=3,
        Axis=(xlabelsize=35, ylabelsize=35))
paper_theme = Theme(
    fontsize = 25, linewidth=2,
    #Axis=(xlabelsize=35, ylabelsize=35)
    )
set_theme!(paper_theme)
fig = plot_effective_area_and_volume(data_tracks, data_cascades; n_mod_thresh=3, pmt_thresh=:n_mod_thrsh_two, add_colorbar=true)
ax = contents(fig[1, 1])[1]
text!(ax, 0, 1,  space = :relative, text="PONE Preliminary",
    align=(:left, :top), offset=(4, -2))
fig
save(joinpath(PLOT_DIR, "aeff_veff.png"), fig)
save(joinpath(PLOT_DIR, "aeff_veff.svg"), fig)


fig = plot_effective_area(data_tracks; n_mod_thresh=3, pmt_thresh=:n_mod_thrsh_two, add_colorbar=true)
#plot_effective_area(data_tracks; n_mod_thresh=3, pmt_thresh=:n_mod_thrsh_three)
save(joinpath(PLOT_DIR, "lightsabre_eff_area.svg"), fig)

fig = plot_effective_volume(data_cascades; n_mod_thresh=3, pmt_thresh=:n_mod_thrsh_two)
save(joinpath(PLOT_DIR, "cascade_eff_volume.png"), fig)
plot_max_eff_v_spacing(data_cascades)

#data_cascades_new = load(joinpath(data_dir, "det_eff_cascade_full_new.jld2"), "results")
plot_neutrino_effective_area(data_cascades)

cos_zen_bins = -1:0.1:1
aeff = _calc_neutrino_effective_area(data_cascades, n_mod_thresh=3, cos_zenith_bins=cos_zen_bins)

aeff |> CSV.write(joinpath(PLOT_DIR, "aeff_cc_casc.csv"))

sel = groupby(aeff, :spacing)[1]


fig = plot_effective_area(data_tracks; n_mod_thresh=1, pmt_thresh=:n_mod_thrsh_three)
fig = plot_effective_area(data_tracks; n_mod_thresh=3, pmt_thresh=:n_mod_thrsh_two)


p = Particle(SA[0., 0., 0.], [0., 0., 1.], 0., 1E4, )

propagate_muon()



fig = Figure()
ax = Axis(fig[1, 1], xlabel="Energy (m)", ylabel="Detection Efficiency")




cmap = cgrad(:viridis, length(logenergies), categorical = true)

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Spacing (m)", ylabel="Muon Effective Area (m^2)", yscale=log10)

for (groupn, group) in pairs(groupby(eff_d, :log_energy))
    eff = combine(groupby(group, :spacing), [:n_mod_thrsh, :proj_area] => effective_area => :effective_area)
    sort!(eff, :spacing)

    mask = eff[:, :effective_area] .!= 0
    x_vals = eff[mask, :spacing]
    y_vals = eff[mask, :effective_area]

    lines!(ax, x_vals, y_vals, label=format("{:.0d} TeV", round(10. ^groupn[1] / 1000)),
        color=groupn[1], colorrange=(minimum(logenergies), maximum(logenergies)), colormap=cmap)
end

tick_labels = Dict(3 => "1 TeV", 4 => "10 TeV", 5=> "100 TeV", 6 => "1 PeV")

Colorbar(fig[1, 2], limits = (minimum(logenergies), maximum(logenergies)), colormap = cmap,
    flipaxis = false, ticks=[3, 4, 5, 6], tickformat= xs -> [tick_labels[x] for x in xs])

fig



save(joinpath(PLOT_DIR, "lightsabre_eff_area.png"), fig)





#fig = Figure()
ax = Axis(fig[1, 2], xlabel="Spacing (m)", ylabel="Detection Efficiency")
for (groupn, group) in pairs(groupby(eff_d, :spacing))

    eff = combine(groupby(group, :log_energy), :n_mod_thrsh => (x -> sum((x .>= 3)) / length(x)) => :eff)
    sort!(eff, :log_energy)

    lines!(ax, eff[:, :log_energy], eff[:, :eff], label=format("{:.0f} m", groupn[1]))
end
#axislegend("Energy")
fig



group=groups[4]
mask = group[:, :n_mod_thrsh] .> 2
sum(mask)
hist(group[mask, :log_energy], weights=fill(1/nrow(group), sum(mask)))


dfc = combine(groupby(eff_d, :spacing), :length => mean, :length => mean)

scatter(dfc[:, :spacing], dfc[:, :length_mean]) 




df_comb = combine(groupby(eff_d, [:spacing, :log_energy]),
 :n_total => mean,
 :n_mod_thrsh => median,
 :n_mod_thrsh => (v -> quantile(v, 0.9)) => :n_mod_thrsh_90,
 :variance_est => mean,
 :nrelpossq => mean,
 :length => mean,
 :max_lhit => mean,
 )
 hmask = (eff_d[:, :dir_theta] .> deg2rad(70)) .&& (eff_d[:, :dir_theta] .< deg2rad(110))
 df_comb_h = combine(groupby(eff_d[hmask, :], [:spacing, :log_energy]),
 :n_total => mean,
 :n_mod_thrsh => median,
 :n_mod_thrsh => (v -> quantile(v, 0.9)) => :n_mod_thrsh_90,
 :variance_est => mean,
 :nrelpossq => mean,
 :length => mean,
 :max_lhit => mean,
 )

 vmask = (eff_d[:, :dir_theta] .< deg2rad(30))
 df_comb_v = combine(groupby(eff_d[vmask, :], [:spacing, :log_energy]),
 :n_total => mean,
 :n_mod_thrsh => median,
 :n_mod_thrsh => (v -> quantile(v, 0.9)) => :n_mod_thrsh_90,
 :variance_est => mean,
 :nrelpossq => mean,
 :length => mean,
 :max_lhit => mean,
 )

 df_comb_v



 fig = Figure()
 ax = Axis(fig[1, 1], xlabel="Spacing (m)", ylabel=" LHit", yscale=Makie.pseudolog10,
 yminorticksvisible=true, yminorticks=IntervalsBetween(10))

log_energies = [3, 4, 5, 6]

 for (e, col) in zip(log_energies, Makie.wong_colors())
     dfs = sort(df_comb_v[df_comb_v[:, :log_energy] .== e, :], :spacing)
     lines!(ax, dfs[:, :spacing], (dfs[:, :max_lhit_mean]),  label=format("{:.0d} TeV vertical", 10^e/1000), color=col)
     dfs = sort(df_comb_h[df_comb_h[:, :log_energy] .== e, :], :spacing)
     lines!(ax, dfs[:, :spacing], (dfs[:, :max_lhit_mean]),  label=format("{:.0d} TeV horizontal", 10^e/1000), color=col, linestyle=:dash)
 end
 
egroup = [LineElement(color = c, linestyle = nothing) for c in Makie.wong_colors()[1:4]]
ori_grp = [LineElement(color = :black, linestyle = :solid), LineElement(color = :black, linestyle = :dash)]

axislegend(ax, [egroup, ori_grp],  [[format("{:.0d} TeV", 10^e/1000) for e in log_energies], ["Vertical", "Horizontal"]], ["Energy", "Direction"], position=:lb)
fig



fig = Figure()
ax = Axis(fig[1, 1], xlabel="Spacing (m)", ylabel=" Mean Number of Hits", yscale=log10,
yminorticksvisible=true, yminorticks=IntervalsBetween(10))
for (e, col) in zip(log_energies, Makie.wong_colors())
    dfs = sort(df_comb_v[df_comb_v[:, :log_energy] .== e, :], :spacing)
    lines!(ax, dfs[:, :spacing], (dfs[:, :n_total_mean]),  label=format("{:.0d} TeV vertical", 10^e/1000), color=col)
    dfs = sort(df_comb_h[df_comb_h[:, :log_energy] .== e, :], :spacing)
    lines!(ax, dfs[:, :spacing], (dfs[:, :n_total_mean]),  label=format("{:.0d} TeV horizontal", 10^e/1000), color=col, linestyle=:dash)
end

egroup = [LineElement(color = c, linestyle = nothing) for c in Makie.wong_colors()[1:3]]
ori_grp = [LineElement(color = :black, linestyle = :solid), LineElement(color = :black, linestyle = :dash)]

axislegend(ax, [egroup, ori_grp],  [[format("{:.0d} TeV", 10^e/1000) for e in logenergies], ["Vertical", "Horizontal"]], ["Energy", "Direction"], position=:lb)
fig

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Spacing (m)", ylabel=" LHit^2 * NHits", yscale=log10,
yminorticksvisible=true, yminorticks=IntervalsBetween(10))
for (e, col) in zip(log_energies, Makie.wong_colors())
    dfs = sort(df_comb_v[df_comb_v[:, :log_energy] .== e, :], :spacing)
    lines!(ax, dfs[:, :spacing], (dfs[:, :n_total_mean] .* dfs[:, :max_lhit_mean].^2),  label=format("{:.0d} TeV vertical", 10^e/1000), color=col)
    dfs = sort(df_comb_h[df_comb_h[:, :log_energy] .== e, :], :spacing)
    lines!(ax, dfs[:, :spacing], (dfs[:, :n_total_mean] .* dfs[:, :max_lhit_mean].^2),  label=format("{:.0d} TeV horizontal", 10^e/1000), color=col, linestyle=:dash)
end

egroup = [LineElement(color = c, linestyle = nothing) for c in Makie.wong_colors()[1:3]]
ori_grp = [LineElement(color = :black, linestyle = :solid), LineElement(color = :black, linestyle = :dash)]

axislegend(ax, [egroup, ori_grp],  [[format("{:.0d} TeV", 10^e/1000) for e in logenergies], ["Vertical", "Horizontal"]], ["Energy", "Direction"], position=:lb)
fig




fig = Figure()
ax = Axis(fig[1, 1], xlabel="Spacing (m)", ylabel=" Std. dev. Estimator", yscale=log10,
yminorticksvisible=true, yminorticks=IntervalsBetween(10))
for (e, col) in zip(log_energies, Makie.wong_colors())
    dfs = sort(df_comb[df_comb[:, :log_energy] .== e, :], :spacing)
    scatter!(ax, dfs[:, :spacing], sqrt.(dfs[:, :variance_est_mean]),  label=format("{:.0d} TeV", 10^e/1000), color=col)
end
fig


fig = Figure()
ax = Axis(fig[1, 1], xlabel="Spacing (m)", ylabel="Rel pos norm", yscale=Makie.pseudolog10,
yminorticksvisible=true, yminorticks=IntervalsBetween(10))
for (e, col) in zip(log_energies, Makie.wong_colors())
    dfs = sort(df_comb[df_comb[:, :log_energy] .== e, :], :spacing)
    lines!(ax, dfs[:, :spacing], sqrt.(dfs[:, :nrelpossq_mean]),  label=format("{:.0d} TeV", 10^e/1000), color=col)

    dfs = sort(df_comb_h[df_comb_h[:, :log_energy] .== e, :], :spacing)
    lines!(ax, dfs[:, :spacing], sqrt.(dfs[:, :nrelpossq_mean]),  label=format("{:.0d} TeV horiz", 10^e/1000),
           color=col, linestyle=:dash)
end
fig


fig = Figure()
ax = Axis(fig[1, 1], xlabel="Spacing (m)", ylabel="Median Modules with >= 2 Hits", yscale=Makie.pseudolog10,
yminorticksvisible=true, yminorticks=IntervalsBetween(10))
for (e, col) in zip(log_energies, Makie.wong_colors())
    dfs = sort(df_comb[df_comb[:, :log_energy] .== e, :], :spacing)
    scatter!(ax, dfs[:, :spacing], dfs[:, :n_mod_thrsh_median],  label=format("{:.0d} TeV", 10^e/1000), color=col)
    poly = Polynomials.fit(log.(dfs[:, :spacing]), log.(dfs[:, :n_mod_thrsh_median]), 1)

    @show poly
    low, hi = extrema(dfs[:, :spacing])
    xs = low:1:hi
    lines!(ax, xs, exp.(poly.(log.(xs))))
    #scatter!(ax, dfs[:, :spacing], dfs[:, :n_mod_thrsh_90],  label=format("{:.0d} TeV", 10^e/1000), marker='🐱', color=col)
end
fig



n_thrsh = 3:8
effs = [sum(df[:, :n_mod_thrsh] .>= n) / n_total for n in n_thrsh]

scatter(n_thrsh, effs)



hist(cos.(df[df[:, :n_mod_thrsh] .> 6, :dir_theta]), normalization=:probability)

sum(any((exp.(log_exp)) .>= 2, dims=1))


for le in logenergies
    edist = Dirac(10^le)
    inj = VolumeInjector(cylinder, edist, pdist, ang_dist, length_dist, time_dist)
end


p = Particle(SA[0., 0., 0.], SA[0., 0., 1.], 0., 1E3, 2000., PMuMinus)

energies = []
for i in 1:100
    p_prop, losses = propagate_muon(p)

    push!(energies, p_prop.energy)
end

mean(energies)


targets_cluster = make_n_hex_cluster_detector(7, 50, 20, 50)

d = Detector(targets_cluster, medium)
hit_buffer = create_input_buffer(d, 1)
cylinder = get_bounding_cylinder(d)
surface = CylinderSurface(cylinder)

buffer = (create_input_buffer(d, 1))
diff_cache = FixedSizeDiffCache(buffer, 6)

modules = get_detector_modules(d)
medium = get_detector_medium(d)

edist = Dirac(10^5.)
inj = SurfaceInjector(surface, edist, pdist, ang_dist, length_dist, time_dist)
ev = rand(inj)
isec = get_intersection(cylinder, ev[:particles][1])

length_in = isec.second - isec.first

particles = ev[:particles]
modules_range_mask = get_modules_in_range(particles, d, 200)
modules_range = (modules[modules_range_mask])
# npmt, 1, ntargets
log_exp_per_pmt, _ = get_log_amplitudes(particles, modules_range, gpu(model); feat_buffer=hit_buffer)

exp_per_mod = sum(exp.(log_exp_per_pmt), dims=1)[:]

exp_per_mod

mpos = reduce(hcat, [m.shape.position for m in modules_range])
mpos_all = reduce(hcat, [m.shape.position for m in modules])
fig, ax = scatter(mpos_all, color=(:black, 0.1))
scatter!(ax, mpos, color=exp_per_mod, colorrange=(1, 10))
fig
