using CairoMakie
using NeutrinoTelescopes
using JLD2
using DataFrames
using PhysicsTools
using Distributions
using LinearAlgebra
using Formatting
using Glob
using CSV
using ForwardDiff
using PhotonPropagation


function calc_ang_std(fisher, dir_theta, dir_phi)
    try
        cov_za = inv(fisher)[2:3, 2:3]
            
        cov_za = 0.5 * (cov_za .+ cov_za')

        dir_sp = [dir_theta, dir_phi]
        dir_cart = sph_to_cart(dir_sp)

        dist = MvNormal(dir_sp, cov_za)
        rdirs = rand(dist, 1000)

        dangles = rad2deg.(acos.(dot.(sph_to_cart.(rdirs[1, :], rdirs[2, :]), Ref(dir_cart))))
        return std(dangles)
    catch y
        return missing
    end
end

function calc_e_std(fisher)
    try
        std_loge = sqrt(inv(fisher)[1, 1])
        return std_loge   
    catch y
        return missing
    end
end


function calc_pos_uncert(fisher, pos_x, pos_y, pos_z)
    try
        var_pos = inv(fisher)[4:6, 4:6]
        jac = ForwardDiff.gradient(norm, [pos_x, pos_y, pos_z])
        var_pos_t = (permutedims(jac) * var_pos * permutedims(jac)')[1]

        return sqrt(var_pos_t)
    catch y
        return missing
    end
end




function _make_resolution_plot!(ax, df; length_cut = 0)
    energies = [3, 4, 5, 6]
    for (e, col) in zip(energies, Makie.wong_colors())
        mask = ( df[:, :log_energy] .== e) .&& (df[:, :length_in] .> length_cut)
        df_sel = df[mask, :]

        df_comb = combine(groupby(df_sel, [:spacing, :model]), :dir_uncert => (x -> mean(skipmissing(x))) => :dir_uncert_mean)

        for (mname, group) in pairs(groupby(df_comb, :model))
            grp_srt = sort(group, :spacing)
            lines!(ax, grp_srt[:, :spacing], grp_srt[:, :dir_uncert_mean], color=col, label=format("{:.0d} TeV", 10^e / 1000), )#linestyle=ls)
        end

    end
    return ax

end

function _make_resolution_plot_model_avg!(ax, df; length_cut = 0)
    energies = [3, 4, 5, 6]

    for (e, col) in zip(energies, Makie.wong_colors())
        mask = ( df[:, :log_energy] .== e) .&& (df[:, :length_in] .> length_cut)
        df_sel = df[mask, :]

        df_comb = combine(groupby(df_sel, :spacing), :dir_uncert => (x -> mean(skipmissing(x))) => :dir_uncert_mean)
        grp_srt = sort(df_comb, :spacing)
        lines!(ax, grp_srt[:, :spacing], grp_srt[:, :dir_uncert_mean], color=col, label=format("{:.0d} TeV", 10^e / 1000), )#linestyle=ls)
    end
    return ax
end

function make_resolution_plot(df; length_cut = 0)
    fig = Figure()
    ax = Axis(fig[1, 1],
            xlabel="Spacing (m)", ylabel="Angular Resolution (deg)",
            yscale=log10,
            
            yminorticksvisible=true,
            yminorticks = IntervalsBetween(10),
            #yticks = [1E-3, 1E-2, 1E-1, 1]
            )
    ax = _make_resolution_plot_model_avg!(ax, df, length_cut=length_cut)
    axislegend("Energy", position=:lt, merge=true)
    fig
end

function _make_resolution_plot_zenith_split!(ax, df; length_cut = 0)
    energies = [3, 4, 5, 6]

    for (e, col) in zip(energies, Makie.wong_colors())
        mask = ( df[:, :log_energy] .== e) .&& (df[:, :length_in] .> length_cut)

        horizontal = mask .&& (df[:, :dir_theta] .> deg2rad(70)) .&& (df[:, :dir_theta] .< deg2rad(110))
        vertical = mask .&& ((df[:, :dir_theta] .< deg2rad(70)) .|| (df[:, :dir_theta] .> deg2rad(110)))

        df_comb_horz = combine(groupby( df[horizontal, :], [:spacing]), :dir_uncert => (x -> mean(skipmissing(x))) => :dir_uncert_mean)
        df_comb_vert = combine(groupby( df[vertical, :], [:spacing]), :dir_uncert => (x -> mean(skipmissing(x))) => :dir_uncert_mean)

        sort!(df_comb_horz, :spacing)
        sort!(df_comb_vert, :spacing)

        lines!(ax, df_comb_horz[:, :spacing], df_comb_horz[:, :dir_uncert_mean], color=col, label=format("{:.0d} TeV horiz.", 10^e / 1000), )
        lines!(ax, df_comb_vert[:, :spacing], df_comb_vert[:, :dir_uncert_mean], color=col, label=format("{:.0d} TeV vert.", 10^e / 1000), linestyle=:dash)
    end

    return ax
end


function make_resolution_plot_zenith_split(df; length_cut = 0)
    energies = [3, 4, 5, 6]
    fig = Figure()
    ax = Axis(fig[1, 1],
            xlabel="Spacing (m)", ylabel="Track Angular Resolution (deg)",
            yscale=log10,
            
            yminorticksvisible=true,
            yminorticks = IntervalsBetween(10),
            #yticks = [1E-3, 1E-2, 1E-1, 1]
            )

    _make_resolution_plot_zenith_split!(ax, df, length_cut=length_cut)

    egroup = [LineElement(color = c, linestyle = nothing) for c in Makie.wong_colors()[1:length(energies)]]
    ori_grp = [LineElement(color = :black, linestyle = :solid), LineElement(color = :black, linestyle = :dash)]

    l = Legend(fig[1, 2], [egroup, ori_grp],  [[format("{:.0d} TeV", round(10^e/1000)) for e in energies], ["Horizontal", "Vertical"]], ["Energy", "Direction"])
    fig
end


function make_combined_resolution_plot_zenith_split(df_casc, df_tracks; length_cut = 0)
    energies = [3, 4, 5, 6]
    fig = Figure(resolution=(1200, 600))
    ax = Axis(fig[1, 1],
            xlabel="Spacing (m)", ylabel="Cascade Angular Resolution (deg)",
            yscale=log10,            
            yminorticksvisible=true,
            yminorticks = IntervalsBetween(10),
            #yticks = [1E-3, 1E-2, 1E-1, 1]
            )
    ax2 = Axis(fig[1, 2],
        xlabel="Spacing (m)", ylabel="Track Angular Resolution (deg)",
        yscale=log10,

        yminorticksvisible=true,
        yminorticks = IntervalsBetween(10),
        #yticks = [1E-3, 1E-2, 1E-1, 1]
    )
    _make_resolution_plot_model_avg!(ax, df_casc)

    _make_resolution_plot_zenith_split!(ax2, df_tracks, length_cut=length_cut)

    egroup = [LineElement(color = c, linestyle = nothing) for c in Makie.wong_colors()[1:length(energies)]]
    ori_grp = [LineElement(color = :black, linestyle = :solid), LineElement(color = :black, linestyle = :dash)]

    l = Legend(fig[1, 3], [egroup, ori_grp],  [[format("{:.0d} TeV", round(10^e/1000)) for e in energies], ["Horizontal", "Vertical"]], ["Energy", "Direction"])
    
    fig
end

function make_binned_resolutions(df, cos_zenith_bins)
    groups = groupby(df, [:log_energy])
    avg_resos = []
    for (groupn, group) in pairs(groups)

        for zbix in 1:(length(cos_zenith_bins) - 1)
            ct = cos.(group[:, :dir_theta])
            lower = cos_zenith_bins[zbix]
            upper = cos_zenith_bins[zbix+ 1] 
            mask = (ct .>= lower) .&& (ct .< upper)
            grpmsk = group[mask, :]

            avg_reso = combine(groupby(grpmsk, :spacing), :dir_uncert => (x -> mean(skipmissing(x))) => :dir_uncert_mean, :log10e_uncert => (x -> mean(skipmissing(x))) => :log10e_uncert_mean)
            avg_reso[!, :log10_energy] .= groupn[1] 
            avg_reso[!, :cos_theta] .= (upper+lower)/2
            push!(avg_resos, avg_reso)
        end
    end
    return reduce(vcat, avg_resos)
end

function proc_data(files)

    rows = []
    for f in files
        res = load(f)["results"]
        cyl = res[:injection_volume]
        for (m, e) in zip(res[:fisher_matrices], res[:events])
            
            p = e[:particles][1]
            dir_theta, dir_phi = cart_to_sph(p.direction)
            pos_x, pos_y, pos_z = p.position
        
            intersection = get_intersection(cyl, p.position, p.direction)
            if isnothing(intersection.first)
                length_in = 0
            else
                length_in = intersection.second - intersection.first
            end

            cad = closest_approach_distance(p, cyl.center)
            cparam = closest_approach_param(p, cyl.center)
            cad_pos = p.position .+ cparam .* p.direction

            if haskey(res, :vert_spacing)
                vert_spacing = res[:vert_spacing]
            else
                vert_spacing = 50
            end

            push!(rows, (fisher=m, log_energy=log10(p.energy), dir_theta=dir_theta,
                dir_phi=dir_phi, pos_x=pos_x, pos_y=pos_y, pos_z=pos_z, length_in=length_in,
                spacing=res[:spacing], vert_spacing=vert_spacing, closest_approach_distance=cad, cad_x=cad_pos[1], cad_y=cad_pos[2],
                cad_z=cad_pos[3], time_uncert=parse(Float64, split(f, "-")[end-1]),
                cylinder=cyl.radius, cad_rho=sqrt(cad_pos[1]^2 + cad_pos[2]^2),
                in_cylinder=point_in_volume(cyl, p.position)))
        end
    end
       
    df = DataFrame(rows)
    df = transform(df, Cols(:fisher, :dir_theta, :dir_phi) => ByRow(calc_ang_std) => :dir_uncert)
    df = transform(df, Cols(:fisher) => ByRow(calc_e_std) => :log10e_uncert)
    df = transform(df, Cols(:fisher, :pos_x, :pos_y, :pos_z) => ByRow(calc_pos_uncert) => :pos_uncert)
end


function binned_average(x, y, bins)
    ixs = searchsortedfirst.(Ref(bins), x)
    averages = Vector{Float64}(undef, length(bins)+1)

    @inbounds for ix in eachindex(averages)
        sel = ixs .== ix
        averages[ix] = mean(skipmissing(y[sel]))
    end
    return averages[2:end-1]
end

bin_centers(bins) = 0.5*(bins[2:end] .+ bins[1:end-1])

outdir = joinpath(ENV["WORK"], "snakemake/fisher")
df_track = proc_data(glob("fisher-light*.jld2", outdir ))
df_track_in = df_track[df_track[:, :length_in] .> 0 .&& df_track[:, :time_uncert] .== 2.5, :] 

df_casc = proc_data(glob("fisher-ext*.jld2", outdir ))
df_casc_in = df_casc[df_casc[:, :in_cylinder] .&& df_casc[:, :vert_spacing] .== 50, :] 

begin 
    f = Figure(resolution=(1200, 1000))
    bins = 1:0.5:6
    f[1, 1] = grid = GridLayout()
    for (i, (groupn, group)) in enumerate(pairs(groupby(df_casc_in, :spacing)))
        row, col = divrem(i-1, 2)
        row +=1
        col +=1 

        subgrid = grid[row, col] = GridLayout()
        
        ax = Axis(subgrid[1, 1], yscale=log10,
                #xlabel="log10(Energy / GeV)",
                ylabel="Angular Resolution (deg)",
                title="Horizontal Spacing: $(groupn[1])"
                )

        sel50m = group[group[:, :vert_spacing] .== 50, :]
        avg_50 = binned_average(sel50m[:, :log_energy], sel50m[:, :dir_uncert], bins)

        ax_r = Axis(subgrid[2, 1], xlabel="log10(Energy / GeV)", ylabel="Ratio")
        for (ggroupn, ggroup) in pairs(groupby(group, :vert_spacing))
            avg = binned_average(ggroup[:, :log_energy], ggroup[:, :dir_uncert], bins)
            lines!(ax, bin_centers(bins), avg, label=string(ggroupn[1]) * "m")
            lines!(ax_r, bin_centers(bins), avg ./ avg_50, label=string(ggroupn[1]) * "m")
        end
        ylims!(ax, 1E-2, 30)
        rowsize!(subgrid, 1, Auto(2))
        rowsize!(subgrid, 2, Auto(1))

    end
    f[1, 2] = Legend(f, ax, "Vert Spacing", framevisible = false)
    f
end

begin
    f = Figure(resolution=(1200, 1000))
    bins = 1:0.5:6
    f[1, 1] = grid = GridLayout()
    for (i, (groupn, group)) in enumerate(pairs(groupby(df_track_in, :spacing)))
        row, col = divrem(i-1, 2)
        row +=1
        col +=1 

        subgrid = grid[row, col] = GridLayout()
        
        ax = Axis(subgrid[1, 1], yscale=log10,
                #xlabel="log10(Energy / GeV)",
                ylabel="Angular Resolution (deg)",
                title="Horizontal Spacing: $(groupn[1])"
                )

        sel50m = group[group[:, :vert_spacing] .== 50, :]
        avg_50 = binned_average(sel50m[:, :log_energy], sel50m[:, :dir_uncert], bins)

        ax_r = Axis(subgrid[2, 1], xlabel="log10(Energy / GeV)", ylabel="Ratio")
        for (ggroupn, ggroup) in pairs(groupby(group, :vert_spacing))
            avg = binned_average(ggroup[:, :log_energy], ggroup[:, :dir_uncert], bins)
            lines!(ax, bin_centers(bins), avg, label=string(ggroupn[1]) * "m")
            lines!(ax_r, bin_centers(bins), avg ./ avg_50, label=string(ggroupn[1]) * "m")
        end
        ylims!(ax, 1E-2, 2)
        rowsize!(subgrid, 1, Auto(2))
        rowsize!(subgrid, 2, Auto(1))

    end
    f[1, 2] = Legend(f, ax, "Vert Spacing", framevisible = false)
    f
end

bins = 1:0.5:6
f = Figure(resolution=(1000, 1000))
for (i, (groupn, group)) in enumerate(pairs(groupby(df_casc_in, [:time_uncert])))
    
    row, col = divrem(i-1, 2)
    row +=1
    col +=1 
    @show i, row, col
    ax = Axis(f[row, col], yscale=log10, title=string(groupn[1]))
    for (ggroupn, ggroup) in pairs(groupby(group, [:spacing]))
        avg = binned_average(ggroup[:, :log_energy], ggroup[:, :dir_uncert], bins)
        lines!(ax, bin_centers(bins), avg, label=string(ggroupn[1]))
    end
    axislegend()
    ylims!(ax, 1E-1, 100)
end
f


bins = 1:0.5:6
f = Figure(resolution=(1000, 1000))
for (i, (groupn, group)) in enumerate(pairs(groupby(df_track_in, [:time_uncert])))
    
    row, col = divrem(i-1, 2)
    row +=1
    col +=1 
    @show i, row, col
    ax = Axis(f[row, col], yscale=log10, title=string(groupn[1]))
    for (ggroupn, ggroup) in pairs(groupby(group, [:spacing]))
        avg = binned_average(ggroup[:, :log_energy], ggroup[:, :dir_uncert], bins)
        lines!(ax, bin_centers(bins), avg, label=string(ggroupn[1]))
    end
    axislegend()
    ylims!(ax, 1E-1, 100)
end
f



f = Figure()
ax = Axis(f[1, 1], yscale=log10)
for (groupn, group) in pairs(groupby(df_track_in, :spacing))
    scatter!(ax, group[:, :closest_approach_distance], group[:, :dir_uncert], label=string(groupn[1]))
end
axislegend(ax)
f


f = Figure()
for (i, (groupn, group)) in enumerate(pairs(groupby(df_track[df_track[:, :time_uncert] .== 1.5, :], :spacing)))
    ax = Axis(f[i, 1], limits=(-600, 600, -600, 600))
    color = copy(group[:, :dir_uncert])
    color[ismissing.(color)] .= NaN
    color = convert(Vector{Float64}, color)

    scatter!(ax, group[:, :cad_x], group[:, :cad_y], color=log10.(color), label=string(groupn[1]),
            )
    lines!(ax, Circle(Point2f((0, 0)), first(group[:, :cylinder])), linewidth=3)
end
f



f = Figure()
ax = Axis(f[1, 1],)
with_theme(
    Theme(
        Scatter = (cycle = [:marker],)
    )) do
    for (groupn, group) in pairs(groupby(df_track, :spacing))
        
        color = copy(group[:, :dir_uncert])
        color[ismissing.(color)] .= NaN
        color = convert(Vector{Float64}, color)

        scatter!(ax, group[:, :cad_x], group[:, :cad_y], color=log10.(color), label=string(groupn[1]),
                )
    end
    axislegend(ax)
    ylims!(-1000, 1000)
    xlims!(-1000, 1000)
    f
end



df_casc = proc_data(glob("fisher-extended*.jld2", outdir ))
bins = 1:0.5:6
f = Figure()
ax = Axis(f[1, 1], yscale=log10)
for (groupn, group) in pairs(groupby(df_casc, :spacing))
    avg = binned_average(group[:, :log_energy], group[:, :dir_uncert], bins)
    lines!(ax, bin_centers(bins), avg, label=string(groupn[1]))
end
axislegend(ax)
f

f = Figure()
for (i, (groupn, group)) in enumerate(pairs(groupby(df_casc, :spacing)))
    ax = Axis(f[i, 1], limits=(-600, 600, -600, 600))
    color = copy(group[:, :dir_uncert])
    color[ismissing.(color)] .= NaN
    color = convert(Vector{Float64}, color)

    scatter!(ax, group[:, :cad_x], group[:, :cad_y], color=log10.(color), label=string(groupn[1]),
            )
    lines!(ax, Circle(Point2f((0, 0)), first(group[:, :cylinder])), linewidth=3)
end
f





load(glob("fisher-lightsabre*.jld2", outdir )[1])["results"]

keys(load(glob("fisher-lightsabre*.jld2", outdir )[1])["results"])


fig = Figure()
ax = Axis(fig[1, 1], yscale=log10)
for (grpn, grp) in pairs(groupby(df_casc[df_casc[:, :model] .== "Model A", :], :log_energy))

    cmb = combine(groupby(grp, :spacing), :pos_uncert => mean)

    grp_sort = sort(cmb, :spacing)
    lines!(ax, cmb[:, :spacing], cmb[:, :pos_uncert_mean], label=grpn[1])
end
ylims!(1E-3, 10)
fig

axislegend(ax)
fig

poster_theme = Theme(
        fontsize = 30, linewidth=3,
        Axis=(xlabelsize=35, ylabelsize=35))
paper_theme = Theme(
    fontsize = 25, linewidth=2,
    #Axis=(xlabelsize=35, ylabelsize=35)
    )
set_theme!(paper_theme)


PLOT_DIR = joinpath(pkgdir(NeutrinoTelescopes), "figures")
fig, ax = scatter(make_n_hex_cluster_positions(7, 50), axis=(xlabel="x (m)", ylabel="y (m)"))
save(joinpath(PLOT_DIR, "cluster_geo.svg"), fig)


cos_zen_bins = -1:0.1:1
uncert_e_zen= make_binned_resolutions(df_casc, cos_zen_bins)
uncert_e_zen |> CSV.write(joinpath(PLOT_DIR, "fisher_reso_casc_full.csv"))

fig = make_resolution_plot(df_track)
save(joinpath(PLOT_DIR, "lightsabre_resolution_all.png"), fig)

fig = make_resolution_plot(df_casc)
save(joinpath(PLOT_DIR, "cascade_resolution_all.png"), fig)

fig = make_resolution_plot(df_casc, length_cut=700)
save(joinpath(PLOT_DIR, "lightsabre_resolution_700.png"), fig)


fig = make_resolution_plot_zenith_split(df_track)
save(joinpath(PLOT_DIR, "lightsabre_resolution_split.svg"), fig)

fig = make_combined_resolution_plot_zenith_split(df_casc, df_track)
ax = contents(fig[1, 1])[1]
text!(ax, 0, 1,  space = :relative, text="PONE Preliminary",
    align=(:left, :top), offset=(4, -2))
fig
save(joinpath(PLOT_DIR, "combined_resolution_split.svg"), fig)
save(joinpath(PLOT_DIR, "combined_resolution_split.png"), fig)

reso_jp = [1643.6006004398953  0.10077652031633488
3033.7180683525303  0.09920507226615771
4086.5215866793255  0.09512953367875654
5998.825080233566  0.08777542950640854
9113.14237165461  0.08618080174529597
13923.960510813695  0.08376465775838565
18648.595736732397  0.080921734387783
25554.92463986205  0.07890373602399786
36662.44583934608  0.07401349877283891
52295.6260888528  0.07158917371148082
76326.86539248186  0.06628988273793296
98771.77860716157  0.06385396782110725
159804.97613948726  0.06350013635124085
235919.96322494413  0.06025770384510509
379537.778187446  0.05661439869102819
563519.9455675046  0.05460594491409876
708617.0706356606  0.05463321516225805
]

reso_jp[:, 1] .= log10.(reso_jp[:, 1])

lengths = [0, 100, 200, 400, 700]
fig = Figure()
ax = Axis(fig[1, 1], xlabel="Log10(Track Energy / GeV)", ylabel="Angular Resolution (deg)")
for length in lengths
    mask= (df[:, :spacing] .== 93.75) .&& (df[:, :model] .== "A1S1") .&& (df[:, :length_in] .> length)

    df_comb = combine(groupby(df[mask, :], [:log_energy]), :dir_uncert => (x -> mean(skipmissing(x))) => :dir_uncert_mean)

    lines!(ax, df_comb[:, :log_energy], df_comb[:, :dir_uncert_mean], label=format("Track length > {:.0d}m", length))
end
lines!(ax, reso_jp, color=:black)
xlims!(ax, 3, 6)
ylims!(ax, 0, 0.4)

axislegend(gridshalign=:left)
fig


unique(df[:, :spacing])


lines( reso_jp')


fig = Figure()
ax = Axis(fig[1, 1], xlabel="Log10(Track Energy / GeV)", ylabel="Angular Resolution (deg)")


pos = make_n_hex_cluster_detector(7, 50, 20, 50)
d = Detector(pos, make_cascadia_medium_properties(0.95))
cyl = get_bounding_cylinder(d)



xy = make_n_hex_cluster_positions(7, 50)

fig, ax = scatter(xy)