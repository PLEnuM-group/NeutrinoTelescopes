module Injectors

using Random
using StaticArrays
using Distributions
using PhysicsTools
using UUIDs
using StructTypes
import Base.rand
import ..Event

export sample_volume, inject
export Cylinder, Cuboid, VolumeType
export SurfaceType, CylinderSurface
export VolumeInjector, Injector
export SurfaceInjector
export ParticleTypeDistribution
export AngularDistribution, UniformAngularDistribution
export LowerHalfSphere
export get_intersection

"""
    VolumeType

Abstract type for volumes
"""
abstract type VolumeType end

"""
    Cylinder{T} <: VolumeType

Type for cylindrical volumes.
"""
struct Cylinder{T} <: VolumeType
    center::SVector{3,T}
    height::T
    radius::T
end

Base.:(==)(a::Cylinder, b::Cylinder) = (a.center == b.center) && (a.height == b.height) && (a.radius == b.radius)
StructTypes.StructType(::Type{<:Cylinder}) = StructTypes.Struct()

"""
    Cylinder{T} <: VolumeType

Type for cuboid volumes.
"""
struct Cuboid{T} <: VolumeType
    center::SVector{3,T}
    l_x::T
    l_y::T
    l_z::T
end

Base.:(==)(a::Cuboid, b::Cuboid) = (a.l_x == b.l_x) && (a.l_y == b.l_y) && (a.l_z == b.l_z)

StructTypes.StructType(::Type{<:Cuboid}) = StructTypes.Struct()

struct FixedPosition{T} <: VolumeType
    position::SVector{3,T}
end

Base.:(==)(a::FixedPosition, b::FixedPosition) = all(a.position .== b.position)

StructTypes.StructType(::Type{<:FixedPosition}) = StructTypes.Struct()
"""
    rand(::VolumeType)

Sample a random point in volume
"""
rand(::VolumeType) = error("Not implemented")
rand(vol::FixedPosition) = vol.position

function rand(vol::Cylinder{T}) where {T}
    uni = Uniform(-vol.height / 2, vol.height / 2)
    rng_z = rand(uni)

    rng_r = sqrt(rand(T) * vol.radius)
    rng_phi = rand(T) * 2 * π
    rng_x = rng_r * cos(rng_phi)
    rng_y = rng_r * sin(rng_phi)

    return SA{T}[rng_x, rng_y, rng_z] + vol.center

end

function rand(vol::Cuboid{T}) where {T}
    uni_x = Uniform(-vol.l_x / 2, vol.l_x / 2)
    uni_y = Uniform(-vol.l_y / 2, vol.l_y / 2)
    uni_z = Uniform(-vol.l_z / 2, vol.l_z / 2)
    return SA{T}[rand(uni_x), rand(uni_y), rand(uni_z)] + vol.center

end



abstract type SurfaceType end
struct CylinderSurface{T} <: SurfaceType
    center::SVector{3,T}
    height::T
    radius::T
end

Base.:(==)(a::CylinderSurface, b::CylinderSurface) = (a.center == b.center) && (a.height == b.height) && (a.radius == b.radius)

CylinderSurface(c::Cylinder) = CylinderSurface(c.center, c.height, c.radius)
Cylinder(c::CylinderSurface) = Cylinder(c.center, c.height, c.radius)

function Base.rand(c::CylinderSurface{T}) where {T}

    cap_area = π*c.radius^2
    mantle_area = 2*π*c.radius*c.height

    cap_prob = cap_area / (cap_area + mantle_area)

    if rand() < cap_prob
        
        # Uniform in one of the caps
        pos_z = c.center[3] + rand([-1, 1]) * c.height/2

        radius = sqrt(rand()*c.radius^2)
        phi = rand(Uniform(0, 2*π))
        pos_x = cos(phi) * radius
        pos_y = sin(phi) * radius

        return SA{T}[pos_x, pos_y, pos_z]

    else
        # Mantle
        pos_z = rand(Uniform(-c.height/2, c.height/2)) + c.center[3]
        phi = rand(Uniform(0, 2 * π))
        pos_x = c.radius * cos(phi)
        pos_y = c.radius * sin(phi)
        return SA{T}[pos_x, pos_y, pos_z]
    end

end

get_surface_normal(::SurfaceType, pos) = error("Not defined")

function get_surface_normal(c::CylinderSurface{T}, pos) where {T <: Real}
    if abs(pos[3]) == c.height /2 + c.center[3]
        # endcap
        return  SA{T}[0, 0, sign(pos[3] - c.center[3])]
    else
        # mantle
        _, phi, _ = cart_to_cyl(pos)
        return SA{T}[cos(phi), sin(phi), 0]
    end
end


mutable struct Intersection{T}
    first::Union{Nothing, T}
    second::Union{Nothing, T}

    function Intersection(first, second)
        T = promote_type(typeof(first), typeof(second))

        if isnothing(first) || isnothing(second)
            return new{T}(first, second)
        else
            if first <=second
                return new{T}(first, second)
            else
                return new{T}(second, first)
            end
        end        
    end
end

get_intersection(::VolumeType, position) = error("Not implemented")
function get_intersection(c::Cylinder{T}, position, direction) where {T <: Real}
    
    x, y, z = position .- c.center

    dir_sph = cart_to_sph(.-direction)
    sinth = sin(dir_sph[1])
    costh = cos(dir_sph[1])
    sinph = sin(dir_sph[2])
    cosph = cos(dir_sph[2])     
   
	
	b = x * cosph + y * sinph
    d = b^2 + c.radius^2 - x^2 - y^2

    i1 = Intersection(nothing, nothing)
    i2 = Intersection(nothing, nothing)

    if d > 0
        d = sqrt(d)
        # down-track distance to the endcaps
		if (costh != 0) 
            i1 = Intersection((z - c.height/2)/costh, (z + c.height/2)/costh)
        end
		
		# down-track distance to the side surfaces
        if (sinth != 0)
            i2 = Intersection((b - d)/sinth, (b + d)/sinth)
        end

        # Perfectly horizontal tracks never intersect the endcaps
        if (costh == 0)
            if ((z > -c.height/2) && (z < c.height/2))
                i1 = i2
            else
                i1 = Intersection(nothing, nothing)
            end
        # Perfectly vertical tracks never intersect the sides
		elseif  (sinth == 0)
            if (hypot(x, y) >= c.radius)
                i1 = Intersection(nothing, nothing)
            end
		# For general tracks, take the last entrace and first exit
		else
		    if (i1.first >= i2.second || i1.second <= i2.first)
				i1 = Intersection(nothing, nothing)
			else
                i1 = Intersection(max(i2.first, i1.first), min(i2.second, i1.second))
            end
		end
	
		return i1
    end

end

get_intersection(c::CylinderSurface, position, direction) = get_intersection(Cylinder(c), position, direction)
get_intersection(volume::VolumeType, particle::Particle) = get_intersection(volume, particle.position, particle.direction)




abstract type AngularDistribution end
abstract type HalfSphereAngularDistribution  <: AngularDistribution end
struct UniformAngularDistribution <: AngularDistribution end

struct LowerHalfSphere <: HalfSphereAngularDistribution end



StructTypes.StructType(::Type{UniformAngularDistribution}) = StructTypes.Struct()

function Base.rand(::UniformAngularDistribution)
    phi = rand() * 2 * π
    theta = acos(2 * rand() - 1)
    return sph_to_cart(theta, phi)
end

function Base.rand(::LowerHalfSphere)
    phi = rand() * 2 * π
    theta = acos(rand() - 1)
    return sph_to_cart(theta, phi)
end


abstract type Injector end
struct VolumeInjector{
    V<:VolumeType,
    E<:UnivariateDistribution,
    A<:AngularDistribution,
    L<:UnivariateDistribution,
    T<:UnivariateDistribution} <: Injector
    volume::V
    e_dist::E
    type_dist::CategoricalSetDistribution
    angular_dist::A
    length_dist::L
    time_dist::T
end

StructTypes.StructType(::Type{VolumeInjector}) = StructTypes.Struct()

function Base.:(==)(a::VolumeInjector, b::VolumeInjector)
    return (
        (a.volume == b.volume) &&
        (a.e_dist == b.e_dist) &&
        (a.type_dist == b.type_dist) &&
        (a.angular_dist == b.angular_dist) &&
        (a.length_dist == b.length_dist) &&
        (a.time_dist == b.time_dist)
    )
end


function Base.rand(inj::VolumeInjector)
    pos = rand(inj.volume)
    energy = rand(inj.e_dist)
    ptype = rand(inj.type_dist)
    dir = rand(inj.angular_dist)
    length = rand(inj.length_dist)
    time = rand(inj.time_dist)

    event = Event()
    event[:particles] = [Particle(pos, dir, time, energy, length, ptype)]

    return event

end


struct SurfaceInjector{
    S<:SurfaceType,
    E<:UnivariateDistribution,
    A<:HalfSphereAngularDistribution,
    L<:UnivariateDistribution,
    T<:UnivariateDistribution} <: Injector
    surface::S
    e_dist::E
    type_dist::CategoricalSetDistribution
    angular_dist::A
    length_dist::L
    time_dist::T
end

function Base.:(==)(a::SurfaceInjector, b::SurfaceInjector)
    return (
        (a.surface == b.surface) &&
        (a.e_dist == b.e_dist) &&
        (a.type_dist == b.type_dist) &&
        (a.angular_dist == b.angular_dist) &&
        (a.length_dist == b.length_dist) &&
        (a.time_dist == b.time_dist)
    )
end


function Base.rand(inj::SurfaceInjector)
    pos = rand(inj.surface)
    energy = rand(inj.e_dist)
    ptype = rand(inj.type_dist)
    length = rand(inj.length_dist)
    time = rand(inj.time_dist)

    # This draws a direction uniformly on a half sphere. Need to rotate z-axis of half sphere onto cylinder tangent plane

    in_surface_normal = get_surface_normal(inj.surface, pos)

    dir = rand(inj.angular_dist)

    dir = rot_from_ez_fast(in_surface_normal, dir)
    
    event = Event()
    event[:particles] = [Particle(pos, dir, time, energy, length, ptype)]

    return event

end

end
