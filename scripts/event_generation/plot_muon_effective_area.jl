using CairoMakie
using DataFrames
using Glob
using NeutrinoTelescopes
using JLD2
using StatsBase
using HDF5


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


files = glob("muon_eff_area_*weights*", "/home/wecapstor3/capn/capn100h/snakemake/muon_eff_area/")



re = r".*muon_eff_area_([0-9]+)_([0-9]+)_([0-9]+)"
data = []

for fname in files
    jldopen(fname) do file
        m = match(re, fname)
        vert_spacing, hor_spacing, file_ix = m.captures
        
        d = file["data"]
        d[!, :vert_spacing] .= parse.(Int64, d[!, :vert_spacing])
        d[!, :hor_spacing] .= parse.(Int64, d[!, :hor_spacing])
        d[!, :file_ix] .= parse(Int64, file_ix)
        push!(data, d)

    end
end

data = reduce(vcat, data)

data_sel = data[data.vert_spacing .== 50 .&& data.hor_spacing .== 80, :]
triggered = data_sel.triggered_ls_20
n_files = length(unique(data_sel.file_ix))

weights = Weights(1 ./data_sel[:, :area_weight]./ n_files)

# area_weight weight has units 1 / GeV m^2 sr 
# weight has units GeV m^2 sr


ebins = 2:0.2:8
cos_bins = -1:0.2:1

h_muon_eff_area = fit(
    Histogram,
    (log10.(data_sel.e_entry[triggered]), data_sel.cos_zen[triggered]),
    weights[triggered],
    (ebins, cos_bins)
)

h_muon_eff_area.weights ./= diff(10 .^ebins);
h_muon_eff_area.weights ./= diff(cos_bins)';
#units: m^2
h_muon_eff_area.weights


# Units:  1/(s*m^2)
nu_flux_at_det = load("/home/wecapstor3/capn/capn100h/leptoninjector/muon_hist_gamma1.2.jld2")["hist"]

# Units: 1/s
nu_eff_area = nu_flux_at_det.weights .* reshape(h_muon_eff_area.weights, (1, size(h_muon_eff_area.weights)...))

# Divide by flux integral flux:

dPhi/dE = phi0 * (E/E_0)^-(2) # 1/GeV cm^s sr 

Rate = int (a_eff * dPhi/dE) dE

dRate/dE/dO / dPhi/dE = aeff 
epre = 1:0.3:8
summed_nu_eff_area = sum(nu_eff_area ,dims = 2)[:, 1, :]

norms = Float64[] # units: 1/m^2 * 1/(m^2 s sr) * 1/s * 1/sr
for i in eachindex(epre)
    if i == length(epre)
        break
    end
    push!(norms, powerlaw_integral(1, 1E-18, 1E5, 10^epre[i], 10^epre[i+1]))
end

aeff = sum(summed_nu_eff_area ./ norms, dims=2)[:]



stairs(epre, [aeff; aeff[end]])


 * π * 1E7

fig, ax, h = heatmap(ebins, cos_bins, sum(nu_eff_area, dims=2)[:, 1, :])
Colorbar(fig[1, 2], h)
fig
h_muon_eff_area.weights

ebins = 2:0.3:9
f = h5open("/home/wecapstor3/capn/capn100h/muon_gun_reco_spline_sim0005_water_normalized_fits.hdf5")

maximum(counts(mctruth[:, "Event"]))

aeff = DataFrame(f["MuonEffectiveArea"][:])
mctruth = DataFrame(f["MCMuon"][:])

energies = mctruth[:, "energy"]
weights = aeff[:, "value"]
zeniths = mctruth[:, "zenith"]
hmuongun = fit(Histogram, log10.(energies), Weights(weights), ebins)
hmuongun.weights ./= (diff(10 .^ebins) * 4*π * 1000)
close(f)

hist(cos.(zeniths))

stairs(ebins, [hmuongun.weights; hmuongun.weights[end]])


fig = Figure()
ax = Axis(fig[1, 1], yscale=log10)
colors = Makie.wong_colors()
ebins = 2:0.3:9

det_configs = groupby(data[data.hor_spacing.==80, :], [:vert_spacing])

for (i, (groupn, data_det)) in enumerate(pairs(det_configs))
    n_files = length(unique(data_det[:, :file_ix]))
    triggered_ls = data_det[:, :triggered_ls_20]
    triggered_full = data_det[:, :triggered_full_20]

    h = fit(Histogram, log10.(data_det[triggered_ls, :e_entry]), Weights(1 ./data_det[triggered_ls, :area_weight]./ (4*π * n_files)), ebins,)
    h2 = fit(Histogram, log10.(data_det[triggered_full, :e_entry]), Weights(1 ./data_det[triggered_full, :area_weight]./ (4*π * n_files)), ebins,)
    h.weights ./= diff(10 .^ebins);
    h2.weights ./= diff(10 .^ebins);

    #stairs!(ax, ebins, [h.weights; h.weights[end]], color=colors[i], label=groupn[1])
    stairs!(ax, ebins, [h2.weights; h2.weights[end]], color=colors[i], label=groupn[1])
    #hlines!(ax, average_proj_area, )
end
#axislegend("Vertical Spacing", position=:lt)
ylims!(1E4, 1E7)
stairs!(ax, ebins, [hmuongun.weights; hmuongun.weights[end]])
fig

data[!, :hor_spacing] .= parse.(Int, data[!, :hor_spacing])
data[!, :vert_spacing] .= parse.(Int, data[!, :vert_spacing])
det_configs = groupby(data[data.hor_spacing.=="60", :], [:vert_spacing])


function calculate_aeff_hist(data, ebins, cos_min, cos_max, n_files)

    cos_mask = (cos_min .<= data[:, :cos_zen]) .& (data[:, :cos_zen] .< cos_max)
    cos_range = cos_max - cos_min

    h = fit(Histogram, log10.(data[cos_mask, :e_entry]), Weights(1 ./data[cos_mask, :area_weight]./ (2*π*cos_range*n_files)), ebins,)
    h.weights ./= diff(10 .^ebins);
    return h
end


ebins = 2:0.2:8
fig = Figure(size=(1000, 1000))
for (hix, (hor, hor_group)) in enumerate(pairs(groupby(sort(data, :hor_spacing), :hor_spacing)))

    println(hor)
    ax = Axis(fig[hix, 1], yscale=log10, title="Horizontal Tracks ($(hor.hor_spacing))", xlabel="log10(Muon Energy)", ylabel="Effective Area (m^2)")
    ax2 = Axis(fig[hix, 2], yscale=log10, title="Vertical Tracks", xlabel="log10(Muon Energy)", ylabel="Effective Area (m^2)")
    colors = Makie.wong_colors()


    for (i, (groupn, data_det)) in enumerate(pairs(groupby(hor_group, :vert_spacing)))
        n_files = length(unique(data_det[:, :file_ix]))

        triggered_ls = data_det[:, :triggered_ls_20]
        triggered_full = data_det[:, :triggered_full_20]

        h_hor = calculate_aeff_hist(data_det[triggered_ls, :], ebins, -0.3, 0.3, n_files)
        h_vert_down = calculate_aeff_hist(data_det[triggered_ls, :], ebins, 0.9, 1, n_files)
        h_vert_up = calculate_aeff_hist(data_det[triggered_ls, :], ebins, -1, -0.9, n_files)
        h_vert = (h_vert_down.weights .+ h_vert_up.weights)

        stairs!(ax, ebins, [h_hor.weights; h_hor.weights[end]], color=colors[i], label=string(groupn[1]))
        stairs!(ax2, ebins, [h_vert; h_vert[end]], color=colors[i], )
        #hlines!(ax, average_proj_area, )
    end
    if hix == 1
        Legend(fig[:, 3], ax, "Vertical Spacing")
    end

    
    ylims!(ax, 1E3, 3E6)
    ylims!(ax2, 1E3, 3E6)
    linkaxes!(ax, ax2)

end
fig


ebins = 2:0.3:8
reference_data = data[data.hor_spacing .== 50 .&& data.vert_spacing .== 30, :]
trigger = reference_data[:, :triggered_ls_20]
n_files = length(unique(reference_data[:, :file_ix]))
h_hor_ref = calculate_aeff_hist(reference_data[trigger, :], ebins, -0.3, 0.3, n_files)
h_vert_down = calculate_aeff_hist(reference_data[trigger, :], ebins, 0.7, 1, n_files)
h_vert_up = calculate_aeff_hist(reference_data[trigger, :], ebins, -1, -0.7, n_files)
h_vert_ref = h_vert_down.weights .+ h_vert_up.weights


mask_dir_uncert = isfinite.(reference_data[!, :dir_uncert]) .&& trigger

reference_data[mask_dir_uncert, :dir_uncert]



fig = Figure(size=(1000, 1000))
for (hix, (hor, hor_group)) in enumerate(pairs(groupby(sort(data, :hor_spacing), :hor_spacing)))

    println(hor)
    ax = Axis(
        fig[hix, 1],
        yscale=log10,
        title="Horizontal Tracks ($(hor.hor_spacing)m)",
        xlabel="log10(Muon Energy)",
        ylabel="Ratio to baseline",
        ytickformat = "{:.2f}",
        yticks = [0.1, 0.2, 0.5, 1, 2, 5, 10] )
    ax2 = Axis(
        fig[hix, 2],
        yscale=log10,
        title="Vertical Tracks ($(hor.hor_spacing)m)",
        xlabel="log10(Muon Energy)",
        ytickformat = "{:.2f}",
        yticks = [0.1, 0.2, 0.5, 1, 2, 5, 10])
    colors = Makie.wong_colors()


    for (i, (groupn, data_det)) in enumerate(pairs(groupby(hor_group, :vert_spacing)))

        n_files = length(unique(data_det[:, :file_ix]))
        triggered_ls = data_det[:, :triggered_ls_20]
        triggered_full = data_det[:, :triggered_full_20]
        n_files

        h_hor = calculate_aeff_hist(data_det[triggered_ls, :], ebins, -0.3, 0.3, n_files)
        h_vert_down = calculate_aeff_hist(data_det[triggered_ls, :], ebins, 0.9, 1, n_files)
        h_vert_up = calculate_aeff_hist(data_det[triggered_ls, :], ebins, -1, -0.9, n_files)
        h_vert = (h_vert_down.weights .+ h_vert_up.weights)

        h_ratio_hor = h_hor.weights ./ h_hor_ref.weights
        h_ratio_vert = h_vert ./ h_vert_ref

        stairs!(ax, ebins, [h_ratio_hor; h_ratio_hor[end]], color=colors[i], label=string(groupn[1]))
        stairs!(ax2, ebins, [h_ratio_vert; h_ratio_vert[end]], color=colors[i], )
        #hlines!(ax, average_proj_area, )
    end
    if hix == 1
        #axislegend(ax, "Vertical Spacing", position=:rb)
        Legend(fig[:, 3], ax, "Vertical Spacing")
    end
    ylims!(ax, 0.2, 5)
    ylims!(ax2, 0.2, 5)
    linkaxes!(ax, ax2)


end
fig

sum(isfinite.(data[:, :dir_uncert]))





mask_horizontal = triggered_events_ls_20 .&& abs.(cos_zens) .< 0.3
mask_vertical = triggered_events_ls_20 .&& abs.(cos_zens) .> 0.7

function calc_average_proj_area(cylinder, cos_min, cos_max)
    return quadgk(ct -> projected_area(cylinder,ct), cos_min, cos_max)[1] / (cos_max-cos_min)
end


h = fit(Histogram, log10.(energies[mask_horizontal]), Weights(1 ./area_weight[mask_horizontal] ./ (2*π * (0.6))), bins,)
h2 = fit(Histogram, log10.(energies[mask_vertical]), Weights(1 ./area_weight[mask_vertical] ./ (2*π * (0.8))), bins,)
h.weights ./= diff(10 .^bins);
h2.weights ./= diff(10 .^bins);

fig, ax, _  = stairs(bins, [h.weights; h.weights[end]])
stairs!(ax, bins, [h2.weights; h2.weights[end]])

hlines!([
    calc_average_proj_area(cylinder, -0.3, 0.3),
    0.5*(calc_average_proj_area(cylinder,-1,  -0.7) + calc_average_proj_area(cylinder,0.7, 1))],
    colors=[Makie.wong_colors()[1:2]])
fig



h = fit(Histogram, log10.(energies[triggered_events_ls_10]), Weights(1 ./area_weight[triggered_events_ls_10] ./ (4*π)), bins,)
h2 = fit(Histogram, log10.(energies[triggered_events_ls_20]), Weights(1 ./area_weight[triggered_events_ls_20] ./ (4*π)), bins,)
h.weights ./= diff(10 .^bins);
h2.weights ./= diff(10 .^bins);

fig, ax, _  = stairs(bins, [h.weights; h.weights[end]])
stairs!(ax, bins, [h2.weights; h2.weights[end]])
#hlines!(ax, Injectors.acceptance(cylinder, -1, 1))
#hlines!(ax, avg_gen_area)
hlines!(ax, average_proj_area, )
#ylims!(1E2, 2E7)
fig

h_all = h = fit(Histogram, log10.(energies), bins,)
h_triggered_ls = fit(Histogram, log10.(energies[triggered_events_ls]), bins,)
h_triggered = fit(Histogram, log10.(energies[triggered_events_full]), bins,)

h_ratio = h_triggered.weights ./ h_all.weights
h_ratio_ls = h_triggered_ls.weights ./ h_all.weights
fig, ax, _ = stairs(bins, [h_ratio; h_ratio[end]])
stairs!(ax, bins .-0.3, [h_ratio_ls; h_ratio_ls[end]])
fig





quadgk(phi -> quadgk(costheta -> projected_area(rand_cyl, sph_to_cart(acos(costheta), phi)), -1, 1)[1], 0, 2*π)[1] / Injectors.acceptance(rand_cyl, -1, 1) 


maximum_proj_area(cylinder)

2*π*cylinder.radius^2 + 2*π*cylinder.radius * cylinder.height


cylinder.height

[ev[:generation_area] / (pdf(inj.e_dist, ev[:e_entry])* n_events) for ev in event_collection]


fieldnames(typeof(inj))


evs = [rand(cylinder) for _ in 1:100000]

nev = 100000

rand_theta = acos.(rand(Uniform(-1, 1), nev))
rand_phi = rand(Uniform(0, 2*π), nev)

rand_dir = sph_to_cart.(rand_theta, rand_phi)


rand_x = rand(Uniform(-1000, 1000), nev)
rand_y = rand(Uniform(-1000, 1000), nev)


rand_pos = [[x, y, 2000] for (x, y) in zip(rand_x, rand_y)]

rotated_pos = rot_from_ez_fast.(rand_dir, rand_pos)


isecs = get_intersection.(Ref(cylinder), rotated_pos, rand_dir)
is_isec = [!isnothing(isec.first) for isec in isecs]

sum(is_isec)
acceptance = sum(is_isec) / nev * 2000 * 2000 * 4*π


Injectors.acceptance(cylinder, -1, 1)


function get_polyhedron(detector)
    posv = [convert(Vector{Float64}, t.shape.position) for t in get_detector_modules(detector)]
    v = vrep(posv)
    v = removevredundancy(v, GLPK.Optimizer)
    p = polyhedron(v)
    return p
end

function orthonormal_basis(direction)
    if direction[3] > 0
        u = 1.
        v = 1.
        w = - (u*direction[1] + v*direction[2]) / direction[3]
        x1 = [u, v, w]
        x1 = x1 ./ norm(x1)
        x2 = cross(x1, direction)
        x2 = x2 ./ norm(x2)
    else
        # vec in in xy-plane
        x1 = [0., 0., 1.]
        x2 = cross(x1, direction)
        x2 = x2 ./ norm(x2)
    end
    return x1, x2
end
