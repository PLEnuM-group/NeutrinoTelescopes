using PhysicsTools
using DataFrames
using StatsBase
using CairoMakie
using ArgParse
using JLD2
using Base.Iterators
import ProposalInterface: make_propagator
using ProposalInterface

function overburden(costh)
    r = 6371E3
    d = 1950
    return sqrt(2*r*d + (r-d)^2*costh^2 -d^2) - (r-d)*costh
end

cos_thetas = -1:0.01:1

lines(cos_thetas, overburden.(cos_thetas), axis=(;yscale=log10))


function propagate_for_distance_energy(energy, distance, nsims; propagator = make_propagator(PMuMinus))

    position = [0., 0., 0.]
    direction = [1., 0., 0.]
    final_energies = Float64[]
    

    p = Particle(position, direction, 0., energy, 0., PMuMinus)
    

    for _ in 1:nsims
        final_state, stochastic_losses, continuous_losses = propagate_muon(p, propagator=propagator, length=distance)
        push!(final_energies, final_state.energy)
    end
    return final_energies
end

cos_thetas = 0:0.01:0.2
distances = overburden.(cos_thetas)
log_energies = 5:1:9

results = DataFrame()

propagator = make_propagator(PMuMinus)

@time propagate_for_distance_energy(1E8, 1000, 10, propagator=propagator)


length(cos_thetas) * length(distances)

for (le, ct) in product(log_energies, cos_thetas)
    dist = overburden(ct)
    push!(results, (log_energy=le, distance=dist, cos_theta=ct, final_energies = propagate_for_distance_energy(10^le, dist, 100, propagator=propagator)))
end



results[!, :final_energy_p16] .= percentile.(results[:, :final_energies], 16)
results[!, :final_energy_p50] .= percentile.(results[:, :final_energies], 50)
results[!, :final_energy_p84] .= percentile.(results[:, :final_energies], 84)


fig = Figure()
ax = Axis(fig[1, 1], yscale=log10, ylabel="Median Energy at Detector (GeV)", xlabel="cos(theta)")

for (groupn, group) in pairs(groupby(results, :log_energy))
    lines!(ax, group[:, :cos_theta], group[:, :final_energy_p50], label=string(groupn.log_energy))
end
axislegend("Log10(Energy)", position=:lt)

fig


