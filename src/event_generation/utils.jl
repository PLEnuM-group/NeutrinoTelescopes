export calc_muon_energy_at_entry
export PowerLogUniform
using StatsBase
using Distributions
using Random

function calc_muon_energy_at_entry(muon, t_entry, stoch, cont)
    energy_loss = 0
    for p in stoch
        t_stoch = mean((p.position - muon.position) ./ muon.direction)
        if t_stoch < t_entry
            energy_loss += p.energy
        else
            break
        end
    end

    for p in cont
        t_cont = mean((p.position - muon.position) ./ muon.direction)
        if t_cont < t_entry
            energy_loss += p.energy
        else
            break
        end
    end

    return muon.energy - energy_loss
end

struct PowerLogUniform{T} <: ContinuousUnivariateDistribution
    min::T
    max::T
end


Base.rand(rng::AbstractRNG, d::PowerLogUniform) = 10 .^ ((log10(d.max) - log10(d.min)) * rand(rng) + log10(d.min))
Distributions.logpdf(d::PowerLogUniform, x::Real) = -log(x) - log(log(d.max/d.min))
Distributions.sampler(d::PowerLogUniform) = d
Distributions.cdf(d::PowerLogUniform, x::Real) = log(x) - log(d.min) / (log(d.max) - log(d.min))
#Distributions.quantile(d::PowerLogUniform, q::Real) = quantile(d.dist, q)
Distributions.minimum(d::PowerLogUniform) = d.min
Distributions.maximum(d::PowerLogUniform) = d.max
Distributions.insupport(d::PowerLogUniform, x::Real) = (x >= minimum(d)) & (x <= maximum(d))

