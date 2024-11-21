using ..EventGeneration
using Functors
using Optimisers
using StatsBase

export OptimizationLayout, ModuleLayout, StringLayout, StringLayoutCart, StringLayoutPolar
export get_center_of_mass


abstract type OptimizationLayout{T} end

struct ModuleLayout{T, P} <: OptimizationLayout{T}
    positions::Vector{Vector{T}}
    pmt_positions::P
end

Functors.@functor ModuleLayout
Optimisers.trainable(x::ModuleLayout) = (; positions = x.positions)

function Base.iterate(layout::ModuleLayout, state=1) 
    return iterate(layout.positions, state)
end

Base.length(layout::ModuleLayout) = length(layout.positions)

function get_center_of_mass(layout::ModuleLayout)
    return mean(layout.positions)
end

abstract type StringLayout{T} <: OptimizationLayout{T} end

get_xy_positions(::StringLayout) = error("Not implemented")
get_z_positions(::StringLayout) = error("Not implemented")


function Base.iterate(layout::StringLayout, state=1) 
    if state > length(layout)
        return nothing
    end

    xy_pos = get_xy_positions(layout)
    z_pos = get_z_positions(layout)

    i, j = divrem(state-1, length(z_pos))

    return ([xy_pos[i+1]; z_pos[j+1]], state+1)
end

function get_center_of_mass(layout::StringLayout)
    return mean(get_xy_positions(layout))
end

Base.length(layout::StringLayout) = length(get_xy_positions(layout)) * length(get_z_positions(layout))

struct StringLayoutCart{T,U, PX <: AbstractVector{<:AbstractVector{T}}, PZ <: AbstractVector{U}, AP <: AbstractVector{<:Real}, P} <: StringLayout{T}
    positions_xy::PX
    positions_z::PZ
    pmt_positions::P
    anchor_position::AP
    scaling_factor::Float64
end

Functors.@functor StringLayoutCart
Optimisers.trainable(x::StringLayoutCart) = (; positions_xy = x.positions_xy)


get_xy_positions(layout::StringLayoutCart) = [layout.positions_xy .* layout.scaling_factor; [layout.anchor_position .* layout.scaling_factor]]
get_z_positions(layout::StringLayoutCart) = layout.positions_z

#=
struct StringLayoutPolar{T,U, PX <: AbstractVector{<:AbstractVector{T}}, PZ <: AbstractVector{U}, P} <: StringLayout{T}
    positions_rphi::PX
    positions_z::PZ
    pmt_positions::P
end

Functors.@functor StringLayoutPolar
Optimisers.trainable(layout::StringLayoutPolar) = (; positions_rphi = layout.positions_rphi)

function get_xy_positions(layout::StringLayoutPolar) 
    rphi_pos = layout.positions_rphi
    return [rphi[1] .* [cos(rphi[2]), sin(rphi[2])] for rphi in rphi_pos]
end
    
get_z_positions(layout::StringLayoutPolar) = layout.positions_z
=#
