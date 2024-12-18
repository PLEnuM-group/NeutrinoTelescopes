module Detectors
using StaticArrays
using Rotations

using PhotonPropagation
using PhysicsTools
using StatsBase
using LinearAlgebra
using ..Injectors
using AbstractMediumProperties
export make_detector_line, make_hex_detector
export make_n_hex_cluster_positions, make_n_hex_cluster_detector
export Detector, LineDetector, get_detector_modules, get_detector_medium, get_detector_pmts
export UnstructuredDetector
export get_bounding_cylinder
export get_detector_lines


abstract type Detector{T<:PhotonTarget, MP <: MediumProperties} end

struct UnstructuredDetector{T<:PhotonTarget, MP <: MediumProperties} <: Detector{T, MP}
    modules::Vector{T}
    medium::MP
end


struct LineDetector{T<:PhotonTarget, MP <: MediumProperties} <: Detector{T, MP}
    modules::Vector{T}
    medium::MP
    line_mapping::Dict{Int64, Vector{Int64}}
end

function LineDetector(lines::AbstractArray{<:AbstractArray{<:PhotonTarget}}, medium)

    modules = reduce(vcat, lines)
    line_mapping = Dict{Int64, Vector{Int64}}()

    counter = 1
    for (line_id, mods) in enumerate(lines)
        line_mapping[line_id] = collect(counter:(counter+length(mods)-1))
        counter += length(mods)
    end

    return LineDetector(modules, medium, line_mapping)
end

get_detector_modules(d::Detector) = d.modules
get_detector_medium(d::Detector) = d.medium

function get_detector_lines(d::LineDetector)
    lines = [d.modules[l] for l in values(d.line_mapping)]
    return lines
end

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
    radius = maximum(norm.(eachcol(positions[1:2, :] .- center_xyz[1:2])))
    height = first(diff(collect(extrema(positions[3, :] .- center_xyz[3]))))
    return Cylinder(SVector{3, Float64}(center_xyz), height.+padding_top, radius.+padding_side)
end

"""
    make_detector_line(position, n_modules, vert_spacing, module_id_start=1, mod_constructor=POM)

Creates a detector line.
# Arguments
    - `position`: Position of the topmost module
    - `n_modules`: Number of modules on the line
    - `vert_spacing`: Vertical module spacing
    - `module_id_start=1`: Id of the first (topmost) module (optional)
    - `mod_constructor=POM`: Function to initialize the module (optical)
"""
function make_detector_line(position, n_modules, vert_spacing, module_id_start=1, mod_constructor=POM)

    line::Vector{mod_constructor} = [
        mod_constructor(SVector{3}(position .- (i - 1) .* [0, 0, vert_spacing]), i + module_id_start - 1)
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
    lines = Vector{Vector{mod_constructor}}(undef, 0)
    for (line_id, (x, y)) in enumerate(eachcol(positions))
        mod = make_detector_line(
            [x, y, z_start],
            n_per_line,
            vert_spacing,
            (line_id - 1) * n_per_line + 1,
            mod_constructor)
        push!(lines, mod)
    end
    return lines
end
function make_hex_detector(n_side, dist, n_per_line, vert_spacing; z_start=0, mod_constructor=POM, truncate=0)

    positions = hex_grid_positions(n_side, dist; truncate=truncate)
    return make_detector_from_line_positions(positions, n_per_line, vert_spacing, z_start=z_start, mod_constructor=mod_constructor)

end


function make_n_hex_cluster_positions(n_clusters, spacing; cluster_rotation=π/2)
    p0 = hex_grid_positions(3, spacing; truncate=1)
    all_pos = [p0]
    
    for phi in LinRange(0, 2*π, n_clusters)[1:end-1]
        center = 3.5*spacing .* [cos(phi), sin(phi)]

        rm = RotMatrix{2}(phi+ cluster_rotation)

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