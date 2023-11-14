module Detectors
using StaticArrays
using Rotations

using PhotonPropagation
using PhysicsTools
using StatsBase
using LinearAlgebra
using ..Injectors
export make_detector_line, make_hex_detector
export make_n_hex_cluster_positions, make_n_hex_cluster_detector
export Detector, get_detector_modules, get_detector_medium, get_detector_pmts
export get_bounding_cylinder

struct Detector{T<:PhotonTarget, MP <: MediumProperties}
    modules::Vector{T}
    medium::MP
end

get_detector_modules(d::Detector) = d.modules
get_detector_medium(d::Detector) = d.medium

function get_detector_pmts(d::Detector)

    modules = get_detector_modules(d)
    T = @NamedTuple{pos::SVector{3, Float64}, mod_ix::Int64, pmt_ix::Int64}
    pmt_positions = Vector{T}(undef, 0)
    for mod in modules
        for (pmt_ix, coords) in enumerate(eachcol(mod.pmt_coordinates))
            pmt_pos_cart = sph_to_cart(coords)
            pos = mod.shape.position .+ mod.shape.radius .* pmt_pos_cart
            push!(pmt_positions, (pos=pos, mod_ix=Int64(mod.module_id), pmt_ix=pmt_ix))
        end
    end
    return pmt_positions
end

function get_bounding_cylinder(d::Detector; padding_top=50., padding_side=50.)
    modules = get_detector_modules(d)
    positions = reduce(hcat, [m.shape.position for m in modules])
    center_xyz = mean(positions, dims=2)[:]
    radius = maximum(norm.(positions[1:2, :] .- center_xyz[1:2]))
    height = first(diff(collect(extrema(positions[3, :] .- center_xyz[3]))))
    return Cylinder(SVector{3, Float64}(center_xyz), height.+padding_top, radius.+padding_side)
end





function make_detector_line(position, n_modules, vert_spacing, module_id_start=1, mod_constructor=POM)

    line = [
        mod_constructor(SVector{3}(position .- (i - 1) .* [0, 0, vert_spacing]), i + module_id_start)
        for i in 1:n_modules
    ]
    return line
end

function hex_grid_positions(n_side, dist; truncate=0 )
    positions = []
    for irow in 0:(n_side-truncate-1)
        i_this_row = 2 * (n_side - 1) - irow
        x_pos = LinRange(
            -(i_this_row - 1) / 2 * dist,
            (i_this_row - 1) / 2 * dist,
            i_this_row
        )

        y_pos = irow * dist * sqrt(3) / 2

        for x in x_pos
            push!(positions, [x, y_pos])
        end

        if irow != 0
            x_pos = LinRange(
                -(i_this_row - 1) / 2 * dist,
                (i_this_row - 1) / 2 * dist,
                i_this_row
            )
            y_pos = -irow * dist * sqrt(3) / 2

            for x in x_pos
                push!(positions, [x, y_pos])
            end
        end
    end
    return reduce(hcat, positions)
end

function make_detector_from_line_positions(positions, n_per_line, vert_spacing; z_start=0, mod_constructor=POM)
    modules = []
    for (line_id, (x, y)) in enumerate(eachcol(positions))
        mod = make_detector_line(
            [x, y, z_start],
            n_per_line,
            vert_spacing,
            (line_id - 1) * n_per_line + 1,
            mod_constructor)
        push!(modules, mod)
    end
    return reduce(vcat, modules)
end
function make_hex_detector(n_side, dist, n_per_line, vert_spacing; z_start=0, mod_constructor=POM, truncate=0)

    positions = hex_grid_positions(n_side, dist; truncate=truncate)
    return make_detector_from_line_positions(positions, n_per_line, vert_spacing, z_start=z_start, mod_constructor=mod_constructor)

end


function make_n_hex_cluster_positions(n_clusters, spacing)
    p0 = hex_grid_positions(3, spacing; truncate=1)
    all_pos = [p0]
    
    for phi in LinRange(0, 2*π, n_clusters)[1:end-1]
        center = 3.5*spacing .* [cos(phi), sin(phi)]

        rm = RotMatrix{2}(phi+ π/2)

        positions = hex_grid_positions(3, spacing; truncate=1)
       
        positions = reduce(hcat, [rm * pos for pos in eachcol(positions)])
        positions .+= center 

        push!(all_pos, positions)
    end

    return reduce(hcat, all_pos)

end

function make_n_hex_cluster_detector(n_clusters, spacing, n_per_line, vert_spacing; z_start=0, mod_constructor=POM)
    positions = make_n_hex_cluster_positions(n_clusters, spacing)
    make_detector_from_line_positions(positions, n_per_line, vert_spacing, z_start=z_start, mod_constructor=mod_constructor)
end
end