using PhotonPropagation
using NeutrinoTelescopes
using PhysicsTools
using CairoMakie
using StaticArrays
using DataFrames
using Rotations
using Formatting
using LinearAlgebra
using Distributions
using Random
using StatsBase
using JSON
using Base.Iterators
using Arrow
using Glob
using CSV
using Interpolations
using Parquet2


function make_random_sources(
    n_pos::Integer,
    n_ph::Integer,
    trange::Real,
    radius::Real)
    sources = Vector{PointlikeTimeRangeEmitter}(undef, n_pos)


    radii = rand(Uniform(0, 1), n_pos) .^ (1 / 3) .* (radius - 0.3) .+ 0.3
    thetas = acos.(rand(Uniform(-1, 1), n_pos))
    phis = rand(Uniform(0, 2 * π), n_pos)

    pos = Float32.(radii) .* sph_to_cart.(Float32.(thetas), Float32.(phis))

    sources = PointlikeTimeRangeEmitter.(
        pos,
        Ref((0.0, trange)),
        Ref(Int64(n_ph))
    )

    return sources
end


function plot_sources(sources)

    scatter([0], [0], [0], marksersize=10, markercolor=:black,
        xlim=(-5, 5), ylim=(-5, 5), zlim=(-5, 5))

    scatter!(
        [src.position[1] for src in sources],
        [src.position[2] for src in sources],
        [src.position[3] for src in sources]
    )

    plot!([0, 0], [0, 0], [-5, 5])
end


function sim_biolumi(target, sources, seed)

    medium = make_cascadia_medium_properties(0.99f0)
    mono_spec = Monochromatic(420.0f0)
    orientation = RotMatrix3(I)

    setup = PhotonPropSetup(sources, [target], medium, mono_spec, seed)
    photons = propagate_photons(setup)

    hits = make_hits_from_photons(photons, setup, orientation)

    if nrow(hits) == 0
        return DataFrame()
    end
    calc_total_weight!(hits, setup)
    all_hits = resample_simulation(hits)
    all_hits[!, :time] = convert.(Float64, all_hits[:, :time])
    return all_hits

end


function read_sources(path, trange, nph)
    bio_pos = DataFrame(read_parquet(path))
    bio_sources = Vector{PointlikeTimeRangeEmitter}()
    for i in 1:nrow(bio_pos)
        position = SVector{3,Float32}(Vector{Float32}(bio_pos[i, [:x, :y, :z]]))
        sources = PointlikeTimeRangeEmitter(position, (0.0, trange), Int64(ceil(nph)))
        push!(bio_sources, sources)
    end
    bio_sources
end


function evaluate_sim(files)
    results = []
    for f in files

        tbl = Arrow.Table(f)
        meta = JSON.parse(Arrow.getmetadata(tbl)["metadata_json"])
        single_pmt_rate = meta["single_pmt_rate"] * 1E3 # rate in Hz

        nsources = length(meta["sources"])
        time_range = 1E7 # TODO READ FROM NEW FILES
        hits = DataFrame(tbl)
        all_hits = resample_simulation(hits)
        all_hits[!, :time] = convert.(Float64, all_hits[:, :time])

        downsampling = 10 .^ (0:0.1:3) # #1.0

        for ds in downsampling

            if ds ≈ 1
                hits = all_hits
            else
                n_sel = Int64(ceil(nrow(all_hits) / ds))
                hits = all_hits[shuffle(1:nrow(all_hits))[1:n_sel], :]
            end

            rate = nrow(hits) / time_range * 1E9 # Rate in Hz

            windows = [10, 15, 20, 30]
            sorted_hits = sort(hits, [:time])

            for window in windows
                coincs_trigger = calc_coincs_from_trigger(sorted_hits, window)
                coincs_fixed_w = count_coinc_in_tw(sorted_hits, window)
                ntup = (
                    ds_rate=ds,
                    hit_rate=rate,
                    hit_rate_1pmt=single_pmt_rate / ds,
                    time_window=window,
                    coincs_trigger=coincs_trigger,
                    coincs_fixed_w=coincs_fixed_w,
                    time_range=time_range,
                    n_sources=nsources)
                push!(results, ntup)
            end
        end

    end
    return DataFrame(results)
end


function make_all_coinc_rate_plot(ax, results_df, trange=1E7, lc_range=2:6)

    grouped_n_src = groupby(results_df, :n_sources)

    for (j, result_df) in enumerate(grouped_n_src)

        grouped_ds = groupby(result_df, :ds_rate)
        coinc_trigger = combine(grouped_ds, All() .=> mean)
       
        for (i, lc_level) in enumerate(lc_range)
            col_sym = Symbol(format("lc_{:d}_mean", lc_level))

            lines!(
                ax,
                coinc_trigger[:, :hit_rate_1pmt_mean],
                coinc_trigger[:, col_sym] .* (1E9 ./ trange),
                label=string(lc_level),
                linestyle=Cycled(j),
                color=Cycled(i)
            )

        end
    end

end

function count_lc_levels(a)
    return [counts(reduce(vcat, a), 2:10)]
end


files = glob("*.arrow", joinpath(@__DIR__, "../data/biolumi_sims/"))
results = evaluate_sim(files)
results = results[results[:, :n_sources] .> 10, :]
results[!, :coincs_trigger] .= Vector{Int64}.(results[!, :coincs_trigger])
Parquet2.writefile(joinpath(@__DIR__, "../data/biolumi_lc.parquet"), results)


Parquet2.readfile(joinpath(@__DIR__, "../data/biolumi_lc.parquet"))
results = DataFrame(Parquet2.readfile(joinpath(@__DIR__, "../data/biolumi_lc.parquet")))

results = results[results[:, :n_sources] .> 10, :]
results[!, :coincs_trigger] .= Vector{Int64}.(results[!, :coincs_trigger])

results[:, :coincs_trigger]

lc_levels = [Symbol("lc_$l") for l in 2:10]
result_proc = hcat(results[:, [:ds_rate, :hit_rate, :hit_rate_1pmt, :time_window, :time_range, :n_sources]], 
DataFrame(reduce(hcat, counts.(results[:, :coincs_trigger], Ref(2:10)))', lc_levels)
)
Parquet2.writefile(joinpath(@__DIR__, "../data/biolumi_lc_proc.parquet"), result_proc)


pkg_dir = @__DIR__

begin
lc_range = 2:6
sources_labels =  string.(getproperty.(keys(groupby(result_proc, :n_sources)), :n_sources))
straw_cum_rates = CSV.read(joinpath(pkg_dir, "../assets/straw_cumulative_rates.csv"), DataFrame, header=[:rate, :frac_above])

linestyles = ["-", ".", "-.", "-..", "-..."]

theme = Theme(
    palette=(color=Makie.wong_colors(), linestyle=linestyles),
    Lines=(cycle=Cycle([:color, :linestyle], covary=true),)
)

set_theme!(theme)


fig = Figure(resolution=(1200, 600))
for (i, tw) in enumerate([10, 20])
    mask = result_proc[:, :time_window] .== tw .&& result_proc[:, :n_sources] .> 10
    subsel = result_proc[mask, :]
   
    ax = Axis(fig[1, i],
        yscale=log10, xscale=log10,
        limits=(1E3, 1E6, 10, 1E6),
        xlabel="Single PMT Rate (Hz)",
        ylabel="LC Rate (Hz)",
        yminorticks=IntervalsBetween(8),
        yminorticksvisible=true,
        yminorgridvisible=true,
        title="LC Window: $tw ns")
    make_all_coinc_rate_plot(ax, subsel, 1E7, lc_range)
    #=
    ax2 = Axis(fig[1, i],
        backgroundcolor = :transparent,
        yaxisposition=:right,
        ylabel="Straw Rate Fraction Above",
        yminorticksvisible=true,
        xscale=log10)
    hidexdecorations!(ax2)
    hidespines!(ax2)
    xlims!(ax2, 1E3, 1E6)
    ylims!(ax2, 0, 1)
    make_all_coinc_rate_plot(ax, subsel, 1E7, lc_range)

    for (lc_level, col) in zip(lc_range, Makie.wong_colors())
        lines!(ax, single_pmt_rates, lc_rate.(single_pmt_rates, tw_size, lc_level), color=col)
    end

    lines!(ax2, straw_cum_rates[:, :rate], straw_cum_rates[:, :frac_above], color=:black)
        =#
end
group_color = [
    LineElement(linestyle=:solid, color=col) for col in Makie.wong_colors()[1:length(lc_range)]
]

group_linestyle = [LineElement(linestyle=ls, color=:black) for ls in linestyles]

legend = Legend(
    fig,
    [group_color, group_linestyle],
    [string.(lc_range), sources_labels],
    ["LC Level", "N-Sources"])

fig[1, 3] = legend
fig
end