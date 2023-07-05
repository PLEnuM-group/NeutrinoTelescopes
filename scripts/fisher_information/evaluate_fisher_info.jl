using CairoMakie
using NeutrinoTelescopes
using JLD2
using DataFrames
using PhysicsTools
using Distributions
using LinearAlgebra
using Formatting
using Glob

outdir = joinpath(ENV["WORK"], "fisher")

df = load(joinpath(outdir, "fisher_ext_casc_hex.jld2"), "results")
df = load(joinpath(outdir, "fisher_muon_hex.jld2"), "results")

files = glob("fisher_track*.jld2", outdir )

df = mapreduce(f -> load(f)["results"], vcat,  files)
#df = load(joinpath(outdir, "fisher_muon_full.jld2"), "results")


function unwrap_row(row)
    matrices = row[:matrices]
    events = row[:event_collection]
    cyl = events.injector.surface
    out = []
    for (m, e) in zip(matrices, events)
        p = e[:particles][1]
        dir_theta, dir_phi = sph_to_cart(p.direction)
        pos_x, pos_y, pos_z = p.position

        intersection = get_intersection(cyl, p.position, p.direction)
        length_in = intersection.second - intersection.first

        push!(out, (fisher=m, log_energy=log10(p.energy), dir_theta=dir_theta,
                dir_phi=dir_phi, pos_x=pos_x, pos_y=pos_y, pos_z=pos_z, length_in=length_in, spacing=row[:spacing],
                model=row[:model]))
    end
    return DataFrame(out)
end

new_rows = []
for row in eachrow(df)
    push!(new_rows, unwrap_row(row))
end
df =  reduce(vcat, new_rows)


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

df = transform(df, Cols(:fisher, :dir_theta, :dir_phi) => ByRow(calc_ang_std) => :dir_uncert)


fig = Figure()
ax = Axis(fig[1, 1], yscale=log10)

for (col, (gname, group)) in zip(Makie.wong_colors(), pairs(groupby(df, (:spacing))))

    for (gname, group2) in pairs(groupby(group, (:model)))
        scatter!(ax, Float64.(group2[:, :log_energy]), group2[:, :dir_uncert], color=col)  
    end
end
fig

fig = Figure()
ax = Axis(fig[1, 1], xlabel="Spacing (m)", ylabel="Angular Resolution (deg)",
    title="100 TeV Lightsabre")
for (gname, group) in pairs(groupby(df[df[:, :log_energy] .== 5, :], (:model)))

    sorted_g = sort(group, :spacing)

    lines!(ax, Float64.(sorted_g[:, :spacing]), sorted_g[:, :dir_uncert], label=gname[1])  
end
axislegend("Model", position=:lt)
fig


fontsize_theme = Theme(fontsize = 20)
set_theme!(fontsize_theme)


function make_resolution_plot(length_cut = 0)
    energies = [3, 4, 5]
    fig = Figure()
    ax = Axis(fig[1, 1],
            xlabel="Spacing (m)", ylabel="Angular Resolution (deg)",
            title="Lightsabre Muon", yscale=log10,
            
            yminorticksvisible=true,
            yminorticks = IntervalsBetween(10),
            #yticks = [1E-3, 1E-2, 1E-1, 1]
            )

    for (e, col, ls) in zip(energies, Makie.wong_colors(), [:solid, :dash, :dashdot])
        mask = ( df[:, :log_energy] .== e) .&& (df[:, :length_in] .> length_cut)
        df_sel = df[mask, :]



        df_comb = combine(groupby(df_sel, [:spacing, :model]), :dir_uncert => (x -> mean(skipmissing(x))) => :dir_uncert_mean)

        #=
        df_comb = sort(
            combine(groupby(df_sel, :spacing), :dir_uncert => minimum => :min, :dir_uncert => maximum => :max),
            :spacing)
        band!(ax, df_comb[:, :spacing], df_comb[:, :min], df_comb[:, :max],
            label=format("{:.0d} TeV", 10^e / 1000))
        =#

        for (mname, group) in pairs(groupby(df_comb, :model))
            grp_srt = sort(group, :spacing)
            lines!(grp_srt[:, :spacing], grp_srt[:, :dir_uncert_mean], color=col, label=format("{:.0d} TeV", 10^e / 1000), )#linestyle=ls)
        end
        @show ls


    end
    axislegend("Energy", position=:lt, merge=true)
    ylims!(1E-3, 1)
    fig
end


make_resolution_plot()

make_resolution_plot(700)



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
xlims!(ax, 3, 5)
ylims!(ax, 0, 0.15)

axislegend(gridshalign=:left)
fig


unique(df[:, :spacing])


lines( reso_jp')