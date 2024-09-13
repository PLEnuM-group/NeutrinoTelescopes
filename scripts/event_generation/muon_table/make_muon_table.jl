using PhysicsTools
using Base.Iterators
using DataFrames
using StatsBase
using CairoMakie
using ArgParse
using JLD2
using ProgressBars
import PhysicsTools.ProposalInterface: make_propagator

function propagate_for_distance_energy(energy, distance, nsims)

    position = [0., 0., 0.]
    direction = [1., 0., 0.]
    final_energies = Float64[]
    propagator = make_propagator(PMuMinus)

    p = Particle(position, direction, 0., energy, 0., PMuMinus)
    

    for _ in 1:nsims
        final_state, stochastic_losses, continuous_losses = propagate_muon(p, propagator=propagator, length=distance)
        push!(final_energies, final_state.energy)
    end
    return final_energies
end
    
function make_table(log_energies, log_distances, nsims)

    data = DataFrame(:initial_energy => Float64[], :distance => Float64[], :final_energies => Vector{Vector{Float64}}(undef, 0))
    for (log_energy, log_distance) in ProgressBar(product(log_energies, log_distances))
        final_energies = propagate_for_distance_energy(10^log_energy, 10^log_distance, nsims)
        push!(data, (initial_energy=10^log_energy, distance=10^log_distance, final_energies=final_energies))
    end
    return data

end

function make_table(args)
    log_energies = args["logE_min"]:args["logE_stepsize"]:args["logE_max"]
    log_distances = args["logdist_min"]:args["logdist_stepsize"]:args["logdist_max"]
    return make_table(log_energies, log_distances, args["nsims"])
end

s = ArgParseSettings()

@add_arg_table s begin
    "--outfile"
    help = "Output filename"
    arg_type = String
    required = true
    "--logE_min"
    help = "Lower log10 energy"
    arg_type = Float64
    required = true
    "--logE_max"
    help = "Upper log10 energy"
    arg_type = Float64
    required = true
    "--logE_stepsize"
    help = "Stepsize in log10E"
    arg_type = Float64
    required = true
    "--logdist_min"
    help = "Lower log10 distance"
    arg_type = Float64
    required = true
    "--logdist_max"
    help = "Upper log10 distance"
    arg_type = Float64
    required = true
    "--logdist_stepsize"
    help = "Stepsize in log10 distance"
    arg_type = Float64
    required = true
    "--nsims"
    help = "Simulations per setting"
    arg_type = Int64
    required = true
end

args = parse_args(s)
table = make_table(args)
jldopen(args["outfile"], "w") do file
    file["table"] = table
end

