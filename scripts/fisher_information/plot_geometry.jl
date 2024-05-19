using NeutrinoTelescopes
using PhotonPropagation
using CairoMakie
using Flux

spacing = 80.
n_modules = 20
vert_spacing = 50.
z_start = 0.

medium = make_cascadia_medium_properties(0.95f0, 1., 1.)

lines = make_n_hex_cluster_detector(7, spacing, n_modules, vert_spacing, z_start=z_start)
d = LineDetector(targets, medium)
#lines = get_detector_lines(d)

xyz = reduce(hcat, [first(line).shape.position for line in lines])

xyz = make_n_hex_cluster_positions(7, 50, cluster_rotation=3*π/4)

scatter(xyz[1, :], xyz[2, :])

