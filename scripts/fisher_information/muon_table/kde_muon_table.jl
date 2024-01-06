using JLD2
using DataFrames
using ArgParse
using StatsBase
using KernelDensityEstimate


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
bin_table(args)