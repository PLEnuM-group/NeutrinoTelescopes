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
using Parquet
using JSON
using Base.Iterators
using Arrow
using Glob
using CSV
using Interpolations


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


function evaluate_sim(files)
    trange = 1E7
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

            rate = nrow(hits) / trange * 1E9 # Rate in Hz

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


function count_lc_levels(a)
    return [counts(reduce(vcat, a), 2:10)]
end


function make_all_coinc_rate_plot(ax, results_df, trange=1E7, lc_range=2:6)

    grouped_n_src = groupby(results_df, :n_sources)

    for (j, result_df) in enumerate(grouped_n_src)

        grouped_ds = groupby(result_df, :ds_rate)
        coinc_trigger = combine(grouped_ds, :coincs_trigger => count_lc_levels => AsTable)
        #coinc_trigger = combine(grpd_tw, :coincs_fixed_w => count_lc_levels => AsTable)
        mean_hit_rate_1pmt = combine(grouped_ds, :hit_rate_1pmt => mean => :hit_rate_mean)
        coinc_trigger = innerjoin(coinc_trigger, mean_hit_rate_1pmt, on=:ds_rate)
        n_sim = combine(grouped_ds, nrow => :n_sim)
        coinc_trigger = innerjoin(coinc_trigger, n_sim, on=:ds_rate)

        for (i, lc_level) in enumerate(lc_range)
            col_sym = Symbol(format("x{:d}", i))

            lines!(
                ax,
                coinc_trigger[:, :hit_rate_mean],
                coinc_trigger[:, col_sym] .* (1E9 ./ (trange .* coinc_trigger[:, :n_sim])),
                label=string(lc_level),
                linestyle=Cycled(j),
                color=Cycled(i)
            )

        end
    end

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

theme = Theme(
    palette=(color=Makie.wong_colors(), linestyle=[:solid, :dash, :dot, :dashdot, :dashdotdot]),
    Lines=(cycle=Cycle([:color, :linestyle], covary=true),)
)

set_theme!(theme)

files = glob("*.arrow", joinpath(@__DIR__, "../data/biolumi_sims"))
results = evaluate_sim(files)
mask = results[:, :time_window] .== 20 .&& results[:, :n_sources] .> 10
subsel = results[mask, :]


begin
    lc_range = 2:6
    fig = Figure()
    ax = Axis(fig[1, 1],
        yscale=log10, xscale=log10,
        limits=(1E3, 1E6, 10, 1E6),
        xlabel="Single PMT Rate (Hz)",
        ylabel="LC Rate (Hz)", yminorticks=IntervalsBetween(8),
        yminorticksvisible=true,
        yminorgridvisible=true,)
    make_all_coinc_rate_plot(ax, subsel, 1E7, lc_range)

    group_color = [
        LineElement(linestyle=:solid, color=col) for col in Makie.wong_colors()[1:length(lc_range)]
    ]

    group_linestyle = [LineElement(linestyle=ls, color=:black) for ls in [:solid, :dash, :dot, :dashdot, :dashdotdot]]

    legend = Legend(
        fig,
        [group_color, group_linestyle],
        [string.(lc_range), string.(getproperty.(keys(groupby(subsel, :n_sources)), :n_sources))],
        ["LC Level", "N-Sources"])

    fig[1, 2] = legend
    fig
end

straw_cum_rates = CSV.read(joinpath(@__DIR__, "../assets/straw_cumulative_rates.csv"), DataFrame, header=[:rate, :frac_above])

straw_cum_rates_interp = linear_interpolation(straw_cum_rates[:, :rate], straw_cum_rates[:, :frac_above])

xs = 3.5:0.1:5.5

lines(10 .^ xs, straw_cum_rates_interp(10 .^ xs), axis=(; xscale=log10, xlabel="Straw Rate (Hz)", ylabel="Fraction Above Rate"))




n_sim = 5
all_res = []
#all_n_src = [5, 10, 30, 50, 80, 100]
all_n_src = [1, 10, 50, 100]
for n_sources in all_n_src

    n_ph = Int64(ceil((1E9 / n_sources)))

    Random.seed!(31338)
    bio_sources = [make_biolumi_sources(n_sources, n_ph, trange) for _ in 1:n_sim]
    rnd_sources = [make_random_sources(n_sources, n_ph * 7, trange, 5) for _ in 1:n_sim]


    bio_pos_df = Vector{Float64}.(JSON.parsefile(joinpath(@__DIR__, "../assets/relative_emission_positions.json")))
    bio_sources_fd = [sample(make_biolumi_sources_from_positions(bio_pos_df, n_ph * 3, trange), n_sources, replace=false) for _ in 1:n_sim]

    #=
    results_bio = reduce(
        vcat,
        [run_sim(target, target_1pmt, sources, trange, i) for (i, sources) in enumerate(bio_sources[1:n_sim])]
    )
    =#

    results_bio_fd = reduce(
        vcat,
        [run_sim(target, target_1pmt, sources, trange, i) for (i, sources) in enumerate(bio_sources_fd[1:n_sim])]
    )

    results_rnd = reduce(
        vcat,
        [run_sim(target, target_1pmt, sources, trange, i) for (i, sources) in enumerate(rnd_sources[1:n_sim])]
    )

    push!(all_res, (n_src=n_sources, bio_df=results_bio_fd, rng=results_rnd))
end




f = Figure()
grid = GridLayout()

for (i, res) in enumerate(all_res)

    row, col = divrem(i - 1, 2)
    @show i, row, col

    ax = Axis(f,
        title=format("{:d} sources", res[:n_src]),
        xlabel="Single PMT Rate",
        ylabel="LC Rate",
        yscale=log10,
        xscale=log10,
        yminorticks=IntervalsBetween(8),
        yminorticksvisible=true,
        yminorgridvisible=true,
    )

    ylims!(ax, (0.1, 1E7))
    xlims!(ax, (1E4, 5E6))

    make_all_coinc_rate_plot(ax, n_sim, res[:bio_df], res[:rng])

    grid[row+1, col+1] = ax
end

lc_range = 2:6
f.layout[1, 1] = grid
group_color = [
    LineElement(linestyle=:solid, color=col) for col in Makie.wong_colors()[1:length(lc_range)]
]

group_linestyle = [LineElement(linestyle=ls, color=:black) for ls in [:solid, :dash, :dot]]

legend = Legend(
    f,
    [group_color, group_linestyle],
    [string.(lc_range), ["Bio FD", "Random"]],
    ["LC Level", "Em. Pos."])

f[1, 2] = legend
f


all_res

#=
write_parquet(joinpath(@__DIR__, "../assets/bio_sources.parquet"),
    DataFrame([(x=src.position[1], y=src.position[2], z=src.position[3]) for sources in bio_sources for src in sources]))

bio_sources = collect(
    partition(
        read_sources(joinpath(@__DIR__, "../assets/bio_sources.parquet"), trange, 1E7),
        n_per_run
    )
)

write_parquet(joinpath(@__DIR__, "../assets/rnd_sources.parquet"),
    DataFrame([(x=src.position[1], y=src.position[2], z=src.position[3]) for sources in rnd_sources for src in sources]))

rnd_sources = collect(
    partition(
        read_sources(joinpath(@__DIR__, "../assets/rnd_sources.parquet"), trange, nph_rnd),
        n_per_run
    )
)

write_parquet(joinpath(@__DIR__, "../assets/bio_sources_fd.parquet"),
    DataFrame([(x=src.position[1], y=src.position[2], z=src.position[3]) for src in bio_sources_fd ]))
bio_sources_fd = collect(
    partition(
        read_sources(joinpath(@__DIR__, "../assets/bio_sources_fd.parquet"), trange, nph_rnd),
        n_per_run
    )
)
=#





grpd_tw = groupby(groupby(results_bio, :time_window)[3], :ds_rate)
coinc_trigger = combine(grpd_tw, :coincs_trigger => count_lc_levels => AsTable)
mean_hit_rate_1pmt = combine(groupby(results_bio, :ds_rate), :hit_rate_1pmt => mean)
coinc_trigger = innerjoin(coinc_trigger, mean_hit_rate_1pmt, on=:ds_rate)


rates_bio = groupby(results_bio, :ds_rate)[(1.0,)][:, :hit_rate_1pmt]
rates_rnd = groupby(results_rnd, :ds_rate)[(1.0,)][:, :hit_rate_1pmt]
rates_bio_fd = groupby(results_bio_fd, :ds_rate)[(1.0,)][:, :hit_rate_1pmt]
f = Figure()
ax = Axis(f[1, 1],
    title="Single PMT-Rates",
    xlabel="Log10(Rate)",
    ylabel="Count"
)

hist!(ax, log10.(rates_bio_fd), label="Bio FD")
hist!(ax, log10.(rates_bio), label="Bio")
hist!(ax, log10.(rates_rnd), label="Random")
axislegend(ax)
f


n_sources = 10
seed = 1
rng = Random.MersenneTwister(seed)
n_ph = Int64(ceil((1E9 / n_sources)))
bio_pos_df = Vector{Float64}.(JSON.parsefile(joinpath(@__DIR__, "../assets/relative_emission_positions.json")))
bio_sources_fd = sample(rng, make_biolumi_sources_from_positions(bio_pos_df, n_ph * 3, trange), n_sources, replace=false)
all_hits = sim_biolumi(target, bio_sources_fd, seed)
all_hits_1pmt = sim_biolumi(target_1pmt, bio_sources_fd, seed)

single_pmt_rate =
    all_hits

bio_sources = make






results_bio_1pmt = vcat(
    [run_sim(target_1pmt, sources, trange) for sources in bio_sources[1:n_sim]]...
)
results_rnd_1pmt = vcat(
    [run_sim(target_1pmt, sources, trange) for sources in rnd_sources[1:n_sim]]...
)
results_bio_fd_1pmt = vcat(
    [run_sim(target_1pmt, sources, trange) for sources in bio_sources_fd[1:n_sim]]...
)


mean_hit_rate_1pmt_bio = combine(groupby(results_bio_1pmt, :ds_rate), :hit_rate => mean)
mean_hit_rate_1pmt_rnd = combine(groupby(results_rnd_1pmt, :ds_rate), :hit_rate => mean)
mean_hit_rate_1pmt_bio_fd = combine(groupby(results_bio_fd_1pmt, :ds_rate), :hit_rate => mean)

scale_factor = mean_hit_rate_1pmt_bio_fd[1, :hit_rate_mean] / mean_hit_rate_1pmt_rnd[1, :hit_rate_mean]
scale_factor_bio = mean_hit_rate_1pmt_bio_fd[1, :hit_rate_mean] / mean_hit_rate_1pmt_bio[1, :hit_rate_mean]

n_sim = 50

results_bio = vcat(
    [run_sim(target, sources, trange) for sources in bio_sources[1:n_sim]]...
)

results_bio_df = vcat(
    [run_sim(target, sources, trange) for sources in bio_sources_fd[1:n_sim]]...
)

results_rnd = vcat(
    [run_sim(target, sources, trange) for sources in rnd_sources[1:n_sim]]...
)





make_all_coinc_rate_plot(
    (results_bio, mean_hit_rate_1pmt_bio),
    (results_bio_df, mean_hit_rate_1pmt_bio_fd),
    (results_rnd, mean_hit_rate_1pmt_rnd))




ax = Axis(f[1, 1])

make_coinc_rate_plot(results_rnd, mean_hit_rate_1pmt_rnd)


d = groupby(joined, :hit_rate)[15]
grpd = groupby(d, :time_window)

p = plot()
for key in keys(grpd)
    combined_coincs = vcat(grpd[key][:, :coincs_trigger]...)

    weights = fill(1E9 / trange, length(combined_coincs))

    scatterhist!(
        p,
        combined_coincs,
        weights=weights,
        label=string(key[1]),
        yscale=:log10,
        bins=1:9,
        yrange=(1E-1, 1E8),
        yticks=10 .^ (0:2:8)
    )
end

p
@df histogram(:coincs_trigger, groupby=:timewindow, yscale=:log10)



results_bio[1][1]


scatterhist(
    coinc_levels, weights=fill((1E9 / (trange * length(t))),
        length(coinc_levels)),
    yscale=:log10,
    bins=1:7,
    ylim=(1E-2, 1E7),
    label="Biolumi",
    xlabel="Coincidence Level",
    title=format("Single-PMT Rate: {:.2f} kHz", mean_rates[1, :bio] / 1000))
p = scatterhist!(
    coinc_levels_rnd,
    weights=fill((1E9 / (trange * length(trnd))),
        length(coinc_levels_rnd)),
    yscale=:log10,
    label="Random",
    bins=1:7,
    ylabel="Rate (Hz)")

savefig(p, joinpath(@__DIR__, "../figures/biolumi_coinc_exam.png"))


sorted_hits = sort(all_hits, [:time])

counts = combine(
    groupby(sorted_hits, :pmt_id),
    nrow => :counts
)

counts[!, :rates] = counts[:, :counts] / trange * 1E9
counts

triggers = lc_trigger(sorted_hits, 20)
coincs = []
for trigger in triggers
    push!(coincs, unique(trigger[:, :pmt_id]))
end
histogram(length.(coincs), yscale=:log10, weights=fill(1E9 / 1E9, length(coincs)))
