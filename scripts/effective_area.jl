using NeutrinoTelescopes
using PyCall
using CairoMakie
using PhysicsTools

PKGDIR = pkgdir(NeutrinoTelescopes)

weighter = WeighterPySpline(joinpath(PKGDIR, "assets/transm_inter_splines.pickle"))

es = 2:0.1:10

fig = Figure()
ax = Axis(fig[1, 1], yscale=log10)

lines!(ax, es, get_total_xsec.(Ref(weighter), :NU_CC, es),)
fig


lines(es, get_transmission_prob.(Ref(weighter), es, Ref(-1)))


get_total_prob(weighter, PNuE(), :NU_CC, 4., 0.2, 100)



np = pyimport("numpy")
pickle = pyimport("pickle")


splines = pickle.load(py"open"(,"rb"))

tprob = splines["transmission_prob"]
xsec = splines["xsec"]
tprob(0.1, 4, grid=false)


# 1 / m^3


xs = 10^(xsec_spline_energy(logesmpl)) * 1E-4 # m^2
interaction_coeff = xs * n_nucleons

#taylor: 1 - exp(-x) ~ x
int_prob = interaction_coeff * chord_lengths

total_prob = int_prob * transmission_probs

eff_areas_contribs = total_prob * proj_areas



cs = -1:0.01:1
lines(cs, first.(tprob.(cs, Ref(4), grid=false)))
plt.plot(cs, )

splines