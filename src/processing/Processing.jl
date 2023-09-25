module Processing

using DataFrames
using PhysicalConstants.CODATA2018
using PhotonPropagation
using PhysicsTools
using Unitful
using LinearAlgebra
using StatsBase
using PoissonRandom
using Reexport

include("triggering.jl")
@reexport using .Triggering

#=
function resample_simulation(hit_times, total_weights, downsample=1.0)
    wsum = sum(total_weights)

    mask = total_weights .> 0
    hit_times = hit_times[mask]
    total_weights = total_weights[mask]

    norm_weights = ProbabilityWeights(copy(total_weights), wsum)
    nhits = min(pois_rand(wsum * downsample), length(hit_times))
    try
        sample(hit_times, norm_weights, nhits; replace=false)
    catch e
        @show length(hit_times)
        error("error")
    end
end


function resample_simulation(df::AbstractDataFrame; downsample=1.0, per_pmt=true, time_col=:time)

    wrapped(hit_times, total_weights) = resample_simulation(hit_times, total_weights, downsample)

    if per_pmt
        groups = groupby(df, [:pmt_id, :module_id])
    else
        groups = groupby(df, :module_id)
    end
    resampled_hits = combine(groups, [time_col, :total_weight] => wrapped => time_col)
    return resampled_hits
end
=#


end
