using HDF5
using DataFrames
using Random
using StaticArrays
using PhysicsTools
using NeutrinoTelescopes
using PhotonPropagation
using JLD2
using StatsBase
using CairoMakie
using ProgressLogging
using ProposalInterface
import ProposalInterface: make_propagator
using Base.Iterators


function _primitive_not_one(e, gamma, phi_null, norm)
    return -(gamma-1) * phi_null/(norm)^-gamma * e^-(gamma-1)
end

function _primitive_one(e, gamma, phi_null, norm)
    return phi_null*norm*log(e)
end

function powerlaw_integral(gamma, phi_null, norm, emin, emax)
 
    primitive = gamma == 1 ? _primitive_one : _primitive_not_one
    
    return primitive(emax, gamma, phi_null, norm) - primitive(emin, gamma, phi_null, norm)
end


function make_table(fname)

    propagator = make_propagator(PMuMinus)

    hdl = h5open(fname)
    table = DataFrame(hdl["RangedInjector0/properties"][:])
    #energy_at_dist_kde = load("/home/hpc/capn/capn100h/.julia/dev/NeutrinoTelescopes/scripts/fisher_information/muon_table/test_muons_hist.jld2", "table")

    df = DataFrame(hdl["RangedInjector0/final_1"][:])
    df2 = DataFrame(hdl["RangedInjector0/final_2"][:])
    weights = hdl["RangedInjector0/flux_weights"][:]
    ow = hdl["RangedInjector0/one_weights"][:]

    propagation_table = DataFrame(initial_energy = Float64[], final_energy = Float64[],
    prop_distance = Float64[], weight = Float64[], cos_zen = Float64[], one_weight=Float64[])
    for (row, weight, one_weight) in take(zip(eachrow(df), weights, ow), 100000)
        # Direction in LI is direction where particle came from
        muon = Particle(row.Position, .-sph_to_cart(row.Direction .+ [0, π]), 0.0,
            row.Energy, 0.0, ptype_for_code(row.ParticleType))
        prop_length = closest_approach_param(muon, [0, 0, 0])

        if prop_length < 0 || muon.energy < 1
            e_final = -1
        else
            propagated_muon, secondaries, cont = propagate_muon(
                muon, length = prop_length, propagator = propagator)
        
            e_final = propagated_muon.energy
        end

        push!(
        propagation_table,
        (initial_energy = row.Energy,
            final_energy = e_final,
            prop_distance = prop_length,
            weight = weight,
            # Here we use the LI definition of zenith angle
            cos_zen = muon.direction[3],
            one_weight=one_weight)
        )
    end

    return propagation_table
end

fname = "/home/wecapstor3/capn/capn100h/leptoninjector/muons_gamma1.hd5"
muons_gamma_1 = make_table(fname)

#fname = "/home/wecapstor3/capn/capn100h/leptoninjector/muons_gamma2.hd5"
#muons_gamma_2 = make_table(fname)

epre = 1:0.3:8
epos = 2:0.2:8
cos_bins = -1:0.2:1

mask = muons_gamma_1.final_energy .>0



# per bin: Rate = sum_e_post_bins sum w = aeff(bin) * integral_bin dPhi/dEdOmega dE dOmega

# Rate(i) = sum_e_post sum_i w_i_post = aeff(bin) * integral_bin dPhi/dEdOmega dE dOmega
# Rate(e_pre, e_post) = sum_i w_pre_post_i
# Rate(e_pre) = sum_e_post Rate(e_pre, e_post) = sum_e_post Rate(e_pre, e_post) = aeff(e_pre) * integral_bin dPhi/dEdOmega dE dOmega
# a_eff(e_pre) = sum_e_post Rate(e_pre, e_post) / integral_bin dPhi/dEdOmega dE dOmega

# a_eff(e_pre, e_post) = sum_e_post Rate(e_pre, e_post) / integral_bin dPhi/dEdOmega dE dOmega

# a_eff(e_pre) = a_geo * e_nu * e_nu_mu

# phi_nu_earth -> phi_mu_det
# R_mu_det = int d phi_mu_det /dE/DOmega * a_eff_mu dEdOmega

# muon rate_per_bin: sum(e_pre)


h_muon_rate = fit(
        Histogram,
        (log10.(muons_gamma_1[mask, :initial_energy]),
            muons_gamma_1[mask, :cos_zen],
            log10.(muons_gamma_1[mask, :final_energy])),
        Weights(Float64.(muons_gamma_1[mask, :weight])), (epre, cos_bins, epos))
h_muon_rate = sum(h_muon_rate.weights, dims=1)[1, :, :]
    
h_muon_flux = h_muon_rate ./ gen_area

heatmap(h_muon_rate)



h_aeff = fit(
    Histogram,
    (log10.(muons_gamma_1[mask, :initial_energy]),
     muons_gamma_1[mask, :cos_zen]),
    Weights(Float64.(muons_gamma_1[mask, :one_weight])), (epre, cos_bins))

norms = Float64[]
for i in eachindex(epre)
    if i == length(epre)
        break
    end
    push!(norms, 1 / (4*π* powerlaw_integral(1, 1E-18, 1E5, 10^epre[i], 10^epre[i+1])) * gen_area)
end

h_aeff.weights ./= norms

fig, ax, h = heatmap(epre, cos_bins, h_aeff.weights)
Colorbar(fig[1,2], h)
fig

h_flux = fit(
        Histogram,
        (log10.(muons_gamma_1[mask, :initial_energy]),
            muons_gamma_1[mask, :cos_zen]),
        Weights(Float64.(muons_gamma_1[mask, :weight])), (epre, cos_bins))
    




h_ow_flux.weights ./ h_flux.weights







geo_radius = 1200
gen_area = geo_radius^2 * π

# weight has units 1/s

h_gamma1 = fit(
    Histogram,
    (log10.(muons_gamma_1[mask, :initial_energy]),
        log10.(muons_gamma_1[mask, :final_energy]),
        muons_gamma_1[mask, :cos_zen]),
    Weights(muons_gamma_1[mask, :weight] ./ gen_area), (epre, epos, cos_bins))




# Units: 1/(s*m^2)

mask = muons_gamma_2.final_energy .>0
h_gamma2 = fit(
    Histogram,
    (log10.(muons_gamma_2[mask, :initial_energy]),
        log10.(muons_gamma_2[mask, :final_energy]),
        muons_gamma_2[mask, :cos_zen]),
    Weights(muons_gamma_2[mask, :weight]), (epre, epos, cos_bins))

jldopen("/home/wecapstor3/capn/capn100h/leptoninjector/muon_hist_gamma1.2.jld2", "w") do hdl
    hdl["hist"] = h_gamma1
end

summed_hist = sum(h_gamma1.weights, dims=1)[1, :, :]

fig = Figure()
ax = Axis(
    fig[1, 1], xlabel = "log10(Final energy / GeV)", ylabel = "cos(zenith)")
heatmap!(ax, epos, cos_bins, replace(log10.(summed_hist), -Inf64 => NaN))
fig

norms = Float64[] # units: 1/m^2 * 1/(m^2 s sr) * 1/s * 1/sr

for i in eachindex(epre)
    if i == length(epre)
        break
    end
    push!(norms, 1 / (4*π* gen_area * powerlaw_integral(1, 1E-18, 1E5, 10^epre[i], 10^epre[i+1])))
end

h_gamma1.weights ./ norms

summed_hist = sum(h_gamma1.weights, dims=(2,3))[:]

reweighted = summed_hist .* norms

summed_hist2 = sum(h_gamma2.weights, dims=(2,3))[:]

fig, ax, _ = lines(summed_hist ./ summed_hist2)
ylims!(ax, 0,2)
fig

summed_hist ./ summed_hist2


powerlaw_integral(3, 1E-18, 1E5, 1E2, 1E8)

epre = 1:0.1:8
epos = 1:0.35:8
cos_zen = -1:0.2:1

h = fit(Histogram,
    (log10.(df[:, :Energy]),
    cos.(getindex.(df[:, :Direction], 1))
    ),
    Weights(weights),
    (epre, cos_zen))

heatmap(epre, cos_zen, h.weights)




h = fit(Histogram,
     (log10.(propagation_table[:, :initial_energy]),
      propagation_table[:, :cos_zen]),
    Weights(propagation_table[:, :weight]), (epre, cos_zen))
hsum = sum(h.weights, dims=1)[:]





propagation_table

fig, ax, h = hist(log10.(abs.(propagation_table.prop_distance[propagation_table.final_energy .<0])))
hist!(ax, log10.(abs.(propagation_table.prop_distance[propagation_table.final_energy .>0])))
fig
scatter(propagation_table.prop_distance, propagation_table.final_energy)





#h.weights ./= diff(10 .^epre) 

summed_hist = sum(h.weights, dims=1)[1, :, :]

fig = Figure()
ax = Axis(
    fig[1, 1], xlabel = "log10(Final energy / GeV)", ylabel = "cos(zenith)")
heatmap!(ax, epos, cos_zen, replace(log10.(summed_hist), -Inf64 => NaN))
fig

summed_hist = sum(h.weights, dims=2)[:, 1, :]

fig = Figure()
ax = Axis(
    fig[1, 1], xlabel = "log10(Final energy / GeV)", ylabel = "cos(zenith)")
heatmap!(ax, epre, cos_zen, replace(log10.(summed_hist), -Inf64 => NaN))
fig



fig = Figure()
ax = Axis(
    fig[1, 1], xlabel = "log10(Initial energy / GeV)", ylabel = "log10(Final energy / GeV)")
heatmap!(ax, epre, epos, replace(log10.(h.weights), -Inf64 => NaN))
fig

geo_radius = 1200
gen_area = geo_radius^2 * π

norm_final = sum(h.weights, dims = (1, 3))[:]

stairs(epos, [norm_final; norm_final[end]] / (gen_area * 4 * π))

sum(norm_final) * π * 1E7 * 1E5 / 10000

fig = Figure()
ax = Axis(
    fig[1, 1], xlabel = "log10(Initial energy / GeV)", ylabel = "log10(Final energy / GeV)")
heatmap!(ax, epre, epos, replace(log10.(norm_per_final), -Inf64 => NaN))
fig

norm = positions = df2[:, :Position]

targets = make_n_hex_cluster_detector(7, 50, 20, 50, z_start = 475)
medium = make_cascadia_medium_properties(0.95f0)
det = LineDetector(targets, medium)
cyl = get_bounding_cylinder(det, padding_top = 100, padding_side = 100)

mask = Bool[]
muons = Particle[]

weights = hdl["RangedInjector0/weights"][:]

distance_table = DataFrame(
    nu_energy = Float64[], mu_energy = Float64[], distance = Float64[],
    weight = Float64[], mu_energy_at_det = Float64[])
@progress for (row, weight) in zip(table, weights)
    p = Particle(SA_F64[row.x, row.y, row.z], sph_to_cart(row.zenith, row.azimuth), 0.0,
        row.totalEnergy * (1 - row.finalStateY), 0.0, ptype_for_code(row.finalType1))
    isec = get_intersection(cyl, p)
    isecs = !isnothing(isec.first) && (isec.first > 0)
    if !isecs
        continue
    end

    muon_propagated, losses_stoch, losses_cont = propagate_muon(
        p, length = isec.first, propagator = propagator)

    push!(distance_table,
        (nu_energy = row.totalEnergy, mu_energy = p.energy,
            mu_energy_at_det = muon_propagated.energy,
            distance = isec.first, weight = weight))
end

distance_table

function get_df_bin(df, initial_energy, distance)
    sorted_keys = [k.initial_energy for k in sort(keys(groupby(df, :initial_energy)))]
    bix = searchsortedfirst(sorted_keys, initial_energy) - 1
    if (bix == 0) | (bix == lastindex(sorted_keys))
        return nothing
    end
    key1 = sorted_keys[bix]

    sorted_keys = [k.distance for k in sort(keys(groupby(df, :distance)))]
    bix = searchsortedfirst(sorted_keys, distance) - 1
    if (bix == 0) | (bix == lastindex(sorted_keys))
        return nothing
    end
    key2 = sorted_keys[bix]

    return groupby(df, [:initial_energy, :distance])[(
        initial_energy = key1, distance = key2)]
end

muon = deepcopy(muons[4])
isec = get_intersection(cyl, muon)
dist = isec.first

e_lost = first(get_df_bin(energy_at_dist_kde, muon.energy, dist)[:, :kde_log_final_energy])

xs = 0:0.01:7
lines(xs, e_lost(collect(xs)))

cylinder = Cylinder(SA[0.0, 0, 0], 1000.0, 500.0)

inj = LIInjector(fname, drop_starting = true, volume = cylinder)

extrema(reduce(hcat, inj.states[:, :Position_final1])[3, :])

save("test.jld2", Dict("inj" => inj))

medium = make_cascadia_medium_properties(0.95f0)

targets = make_n_hex_cluster_detector(7, 200, 20, 50, z_start = 475)
d = Detector(targets, medium)
get_bounding_cylinder(d).radius

Base.rand(inj)