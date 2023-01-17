using CairoMakie
using NeutrinoTelescopes
using StaticArrays
using CSV
using DataFrames

log_energies = 2:0.1:8
zs = (0:0.1:20.0)# m
medium = make_cascadia_medium_properties(0.99)
wls = 200:1.0:800

p = plot(wls, frank_tamm.(wls, refractive_index.(wls, Ref(medium))) .* 1E9,
    xlabel="Wavelength (nm)", ylabel="Photons / (nm ⋅ m)", dpi=150, xlim=(200, 800),
)

water_abs = DataFrame(CSV.File(joinpath(@__DIR__, "../assets/water_absorption_wiki.csv");
    header=[:x, :y], delim=";", decimal=',', type=Float64))
p = plot!(twinx(p), water_abs[:, :x], 1 ./ water_abs[:, :y], color=:red, xticks=:none,
    yscale=:log10, ylabel="Absorption length (m)", label="Absorption", ylim=(1E-3, 1E2))

savefig(p, joinpath(@__DIR__, "../figures/ch_spectrum.png"))


# Plot longitudinal profile
plot(zs, longitudinal_profile.(Ref(1E3), zs, Ref(medium), Ref(PEMinus)), label="1E3 GeV",
    ylabel="PDF", title="Longitudinal Profile", dpi=150)
p = plot!(zs, longitudinal_profile.(Ref(1E6), zs, Ref(medium), Ref(PEMinus)), label="1E6 GeV",
    xlabel="Distance along axis (m)")
savefig(p, joinpath(@__DIR__, "../figures/long_profile_comp.png"))


# Show fractional contribution for a segment of shower depth
frac_contrib = fractional_contrib_long(1E5, zs, medium, PEMinus)


plot(zs, frac_contrib, linetype=:steppost, label="", ylabel="Fractional light yield")

ftamm_norm = frank_tamm_norm((200.0, 800.0), wl -> refractive_index(wl, medium))
light_yield = cascade_cherenkov_track_length.(1E5, PEMinus)

plot(zs, frac_contrib .* light_yield, linetype=:steppost, label="", ylabel="Light yield per segment")


# Calculate Cherenkov track length as function of energy
tlens = cascade_cherenkov_track_length.((10 .^ log_energies), PEMinus)
plot(log_energies, tlens, yscale=:log10, xlabel="Log10(E/GeV)", ylabel="Cherenkov track length")

total_lys = frank_tamm_norm((200.0, 800.0), wl -> refractive_index(wl, medium)) * tlens

p = plot(log_energies, total_lys, yscale=:log10, ylabel="Number of photons", xlabel="log10(Energy/GeV)",
    label="", dpi=150)
savefig(p, joinpath(@__DIR__, "../figures/photons_per_energy.png"))


# Calculate light yield for muons

rel_additional_track_length.(refractive_index(800.0, medium), 1E4)


plot(log_energies, rel_additional_track_length.(refractive_index(800.0, medium), 10 .^ log_energies) .* frank_tamm(450.0, refractive_index(450.0, medium)) .* 1E9 .* 1E2 / 10)

rel_additional_track_length.(refractive_index(800.0, medium), 10 .^ log_energies)


wl_range = (200.0, 800.0)
total_lys = total_lightyield.(Ref(Track()), 10 .^ log_energies, 1.0, Ref(medium), Ref(wl_range))
total_lys_simple = frank_tamm_norm(wl_range, wl -> refractive_index(wl, medium)) .* (1 .+ rel_additional_track_length.(refractive_index(400.0, medium), 10 .^ log_energies))

fig = Figure()
ax = Axis(fig[1, 1], xlabel="log10(E/GeV)", ylabel="Ratio")
lines!(ax, log_energies, total_lys ./ total_lys_simple)
#hlines!(total_lys_single, color=:red)
fig


lambdas = 200:1.0:800

plot(lambdas, refractive_index.(lambdas, Ref(medium)))
plot(lambdas, dispersion.(lambdas, Ref(medium)))
plot(group_velocity.(lambdas, Ref(medium)))
