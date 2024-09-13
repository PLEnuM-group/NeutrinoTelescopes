using JLD2
using DataFrames
using ArgParse
using StatsBase
using CairoMakie
using Distributions


function make_hist(data, bins)
    return fit(Histogram, log10.(first(data)),  bins)
end

function make_kde(data)
    return kde!(log10.(first(data)))
end

function bin_table(args)
    table = jldopen(args["infile"])["table"]
    table[!, :energy_loss] .= broadcast((x,y) -> broadcast(-, x, y), table[:, :initial_energy], table[:, :final_energies])
    table[!, :dEE] .= table[:, :energy_loss] ./ ( table[:, :initial_energy])
    energy_at_dist_table_hist = combine(
        groupby(table, [:initial_energy, :distance]),
        :final_energies => make_kde => :kde_log_final_energy)
    @show names(energy_at_dist_table_hist)
    jldsave(args["outfile"], table=energy_at_dist_table_hist)
end


s = ArgParseSettings()

@add_arg_table s begin
    "--outfile"
    help = "Output filename"
    arg_type = String
    required = true
    "--infile"
    help = "Input filename"
    arg_type = String
    required = true
end

args = parse_args(s)

args = Dict("infile" => "/home/wecapstor3/capn/capn100h/muon_table.jld2")
table = load(args["infile"])["table"]

table[!, :mean_energy] .= mean.(table[:, :final_energies])
table[!, :energy_loss] .=broadcast((x,y) -> broadcast(-, x, y), table[:, :initial_energy], table[:, :final_energies])

table[!, :dEE] .= table[:, :energy_loss] ./ ( table[:, :initial_energy])

table[!, :z] .= table[:, :final_energies] ./ ( table[:, :initial_energy])

table

table[15, :]
ix = 130
fig, ax, _ = CairoMakie.hist(.-log10.((table[ix, :z])), bins=20, normalization=:pdf)
d = fit(Gamma, (table[ix, :dEE]))
lines!(ax, d)
fig

hist = fit(Histogram, (log10.(table[:, :initial_energy]), log10.(table[:, :distance])), Weights(log10.(table[:, :mean_energy])))

heatmap(hist)

