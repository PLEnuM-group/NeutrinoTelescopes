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
    return Polyhedra.volume(polyhedron(hrep(p2)))
end

function get_average_geo_proj_area(p)

    thetas = acos.(rand(Uniform(-1, 1), 1000))
    phis = rand(Uniform(0, 2*π), 1000)

    directions = sph_to_cart.(thetas, phis)

    return mean(get_geo_proj_area.(Ref(p), directions))
end

function effective_area(above_thrsh, proj_area)
    eff_area = sum(proj_area[above_thrsh]) / length(above_thrsh)
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
    ax = Axis(fig[1, 1], xlabel="Spacing (m)", ylabel="Cascade Effective Volume (km^3)",
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

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel="Spacing (m)", ylabel="Track Effective Area (km^2)", yscale=log10)

    _plot_effective_area!(ax, df, n_mod_thresh=n_mod_thresh, pmt_thresh=pmt_thresh)

    tick_labels = Dict(3 => "1 TeV", 4 => "10 TeV", 5=> "100 TeV", 6 => "1 PeV")

    if add_colorbar
        Colorbar(fig[1, 2], limits = (minimum(logenergies), maximum(logenergies)), colormap = cmap,
            flipaxis = false, ticks=[3, 4, 5, 6], tickformat= xs -> [tick_labels[x] for x in xs])
    end

    ylims!(ax, 1E-2, 4)
    fig
end


function plot_effective_area_and_volume(df_track, df_casc; n_mod_thresh=1, pmt_thresh=:n_mod_thrsh_two, add_colorbar=true)

    fig = Figure(resolution=(1200, 600))
    ax = Axis(fig[1, 1], xlabel="Spacing (m)", ylabel="Track Effective Area (km^2)", yscale=log10)
    ax2 = Axis(fig[1, 2], xlabel="Spacing (m)", ylabel="Cascade Effective Volume (km^3)",
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



PLOT_DIR = joinpath(pkgdir(NeutrinoTelescopes), "figures")

data_dir = joinpath(ENV["WORK"], "fisher")
data_tracks = load(joinpath(data_dir, "det_eff_track_full.jld2"), "results")
data_cascades = load(joinpath(data_dir, "det_eff_cascade_full.jld2"), "results")

fontsize_theme = Theme(fontsize = 20)
set_theme!(fontsize_theme)

fig = plot_effective_area_and_volume(data_tracks, data_cascades; n_mod_thresh=3, pmt_thresh=:n_mod_thrsh_two, add_colorbar=true)
save(joinpath(PLOT_DIR, "aeff_veff.png"), fig)
fig = plot_effective_area(data_tracks; n_mod_thresh=3, pmt_thresh=:n_mod_thrsh_two, add_colorbar=false)
#plot_effective_area(data_tracks; n_mod_thresh=3, pmt_thresh=:n_mod_thrsh_three)
save(joinpath(PLOT_DIR, "lightsabre_eff_area.png"), fig)

fig = plot_effective_volume(data_cascades; n_mod_thresh=3, pmt_thresh=:n_mod_thrsh_two)
save(joinpath(PLOT_DIR, "cascade_eff_volume.png"), fig)
plot_max_eff_v_spacing(data_cascades)




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
