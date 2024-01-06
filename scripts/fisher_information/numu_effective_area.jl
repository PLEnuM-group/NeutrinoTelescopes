using NeutrinoTelescopes
using Distributions
using CairoMakie

PKGDIR = pkgdir(NeutrinoTelescopes)
weighter = WeighterPySpline(joinpath(PKGDIR, "assets/transm_inter_splines.pickle"))

muon_maxrange(e, a=0.212/1.2, b=0.251E-3/1.2) = -(log(a) - log(a+b*e))/b

muon_e(e0, dist, a=0.268, b=0.470E-3) = (exp(-dist*b)*(a+b*e0) - a)/b
 
muon_e0(e, dist, a=0.268, b=0.470E-3) = ((a + b*e) / (exp(-dist * b)) - a) / b 

muon_e0(1E3, 1E4)

distances = 10 .^(1:0.1:5)
energy = 1E4
muon_e0.(energy, distances)
interaction_coeff = get_interaction_coeff(weighter, :NU_CC, log10(energy))
max_range = muon_maxrange(energy)
interaction_prob = interaction_coeff .* (max_range .- distances)

lines(distances, interaction_prob)



p_energy = Pareto(1, 100)

energy = rand(p_energy)

int_coeff = get_interaction_coeff(weighter, :NU_CC, log10(energy))

1/int_coeff



p_intpos = truncated(Exponential(1/int_coeff), upper= max_range)

intpos = rand(p_intpos)

prop_dist = max_range - intpos


e_at_det = muon_e(energy, prop_dist)
