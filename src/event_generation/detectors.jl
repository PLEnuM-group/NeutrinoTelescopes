module Detectors
using StaticArrays

using PhotonPropagation
using PhysicsTools
using StatsBase
using LinearAlgebra
using ..Injectors
export make_pom_pmt_coordinates, make_pone_module, make_detector_line, make_hex_detector
export Detector, get_detector_modules, get_detector_medium
export get_bounding_cylinder
struct Detector
    modules::Vector{<:PhotonTarget}
    medium::MediumProperties
end

get_detector_modules(d::Detector) = d.modules
get_detector_medium(d::Detector) = d.medium

function get_bounding_cylinder(d::Detector; padding_top=50., padding_side=50.)
    modules = get_detector_modules(d)
    positions = reduce(hcat, [m.position for m in modules])
    center_xyz = mean(positions, dims=2)[:]
    radius = maximum(norm.(positions[1:2, :] .- center_xyz[1:2]))
    height = maximum(abs.(extrema(positions[3, :] .- center_xyz[3])))
    return Cylinder(SVector{3, Float64}(center_xyz), height.+padding_top, radius.+padding_side)
end

function make_pom_pmt_coordinates(T::Type)

    coords = Matrix{T}(undef, 2, 16)

    #upper 
    coords[1, 1:4] .= deg2rad(90 - 25)
    coords[2, 1:4] = (range(0; step=π / 2, length=4))

    # upper 2
    coords[1, 5:8] .= deg2rad(90 - 57.5)
    coords[2, 5:8] = (range(π / 4; step=π / 2, length=4))

    # lower 2
    coords[1, 9:12] .= deg2rad(90 + 25)
    coords[2, 9:12] = [π/2, 0, 3*π/2, π]

    # lower
    coords[1, 13:16] .= deg2rad(90 + 57.5)
    coords[2, 13:16] = [π/4, 7/4*π, 5/4*π, 3/4*π]

    R = calc_rot_matrix(SA[0.0, 0.0, 1.0], SA[1.0, 0.0, 0.0])
    @views for col in eachcol(coords)
        cart = sph_to_cart(col[1], col[2])
        col[:] .= cart_to_sph((R * cart)...)
    end

    return SMatrix{2,16}(coords)
end


function make_pone_module(position, module_id)
    pmt_area = (75e-3 / 2)^2 * π
    target_radius = 0.21
    target = MultiPMTDetector(
        position,
        target_radius,
        pmt_area,
        make_pom_pmt_coordinates(Float64),
        UInt16(module_id)
    )

    return target
end

function make_detector_line(position, n_modules, vert_spacing, module_id_start=1, mod_constructor=make_pone_module)

    line = [
        mod_constructor(SVector{3}(position .- (i - 1) .* [0, 0, vert_spacing]), i + module_id_start)
        for i in 1:n_modules
    ]
    return line
end

function make_hex_detector(n_side, dist, n_per_line, vert_spacing; z_start=0, mod_constructor=make_pone_module, truncate=0)

    modules = []
    line_id = 1

    for irow in 0:(n_side-truncate-1)
        i_this_row = 2 * (n_side - 1) - irow
        x_pos = LinRange(
            -(i_this_row - 1) / 2 * dist,
            (i_this_row - 1) / 2 * dist,
            i_this_row
        )

        y_pos = irow * dist * sqrt(3) / 2

        for x in x_pos
            mod = make_detector_line(
                [x, y_pos, z_start],
                n_per_line,
                vert_spacing,
                (line_id - 1) * n_per_line + 1,
                mod_constructor)
            push!(modules, mod)
            line_id += 1
        end

        if irow != 0
            x_pos = LinRange(
                -(i_this_row - 1) / 2 * dist,
                (i_this_row - 1) / 2 * dist,
                i_this_row
            )
            y_pos = -irow * dist * sqrt(3) / 2

            for x in x_pos
                mod = make_detector_line(
                    [x, y_pos, z_start],
                    n_per_line,
                    vert_spacing,
                    (line_id - 1) * n_per_line + 1,
                    mod_constructor)
                push!(modules, mod)
                line_id += 1

            end
        end
    end
    return reduce(vcat, modules)
end




end
